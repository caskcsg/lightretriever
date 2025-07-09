import time
from typing import Optional

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx

class MaxLinearMapperFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, attention_mask: Optional[Tensor] = None, fused: bool = False):
        """
        Args:
            input (Tensor): Shape [batch_size, seq_len, hidden_dim].
            weight (Tensor): Shape [hidden_dim, vocab_size].
            bias (Optional[Tensor]): Shape [vocab_size].
            attention_mask (Optional[Tensor]): Shape [batch_size, seq_len], where True means valid positions.

        Returns:
            max_values (Tensor): Shape [batch_size, vocab_size].
        """
        device, dtype = input.device, input.dtype

        batch_size, seq_len, hidden_dim = input.shape
        _, vocab_size = weight.shape

        # Allocate storage for max values
        max_values = torch.full((batch_size, vocab_size), torch.finfo(dtype).min, device=device, dtype=dtype)
        max_indices = None  # Only compute when needed

        # Check whether requires grad
        input_require_grad = ctx.needs_input_grad[0]
        weight_require_grad = ctx.needs_input_grad[1]
        bias_require_grad = ctx.needs_input_grad[2]
        if input_require_grad or weight_require_grad or bias_require_grad:
            # Initialize max_indices only if needed
            max_indices = torch.zeros((batch_size, vocab_size), device=device, dtype=torch.long)
        
        # TODO: t (int) will incur one recompilation when it changes. But this is the fastest way of compiled forward. 
        def _sliced_forward(
            input_sliced: Tensor,                       # [batch_size, hidden_size]
            attention_mask_sliced: Optional[Tensor],    # [batch_size, 1]
            weight: Tensor,                             # [hidden_size, vocab_size]
            bias: Optional[Tensor],                     # [vocab_size]
            max_values: Tensor,                         # [batch_size, vocab_size]
            max_indices: Optional[Tensor],              # [batch_size, vocab_size]
            t: int,
        ):
            """ Perform a sliced forward `input_sliced @ weight + bias`, mask with `attention_mask_sliced`.
                Update the results of current step `t` on `max_values` and `max_indices` **in-place**.
            """
            # Compute logits for the current step: Shape [batch_size, vocab_size]
            logits = input_sliced @ weight
            if bias is not None:
                logits.add_(bias)

            # Apply attention mask if provided
            if attention_mask_sliced is not None:   # Shape [batch_size, 1]
                logits.masked_fill_(~attention_mask_sliced, torch.finfo(logits.dtype).min)   # In-place mask

            # Update max values and optionally max_indices
            max_mask = logits > max_values
            torch.where(max_mask, logits, max_values, out=max_values)

            if max_indices is not None:
                max_indices[max_mask] = t
        
        if fused:
            slice_forward_func = torch.compile(_sliced_forward, fullgraph=True)
        else:
            slice_forward_func = _sliced_forward

        for t in range(seq_len):
            input_sliced = input[:, t]
            attention_mask_sliced = attention_mask[:, t:t+1] if attention_mask is not None else None
            if (t == 0) and fused:
                torch._dynamo.mark_dynamic(input_sliced, 0)
                if attention_mask_sliced is not None:
                    torch._dynamo.mark_dynamic(attention_mask_sliced, 0)
                torch._dynamo.mark_dynamic(max_values, 0)
                if max_indices is not None:
                    torch._dynamo.mark_dynamic(max_indices, 0)
            
            slice_forward_func(
                input_sliced, attention_mask_sliced, weight, bias, max_values, max_indices, t
            )

        # Save tensors only if needed for backward
        ctx.save_for_backward(input, weight, bias, max_indices, torch.tensor(fused, device=input.device, dtype=torch.bool))

        return max_values
    
    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor):
        """
        Args:
            grad_output (Tensor): Gradient of the loss with respect to the output, shape [batch_size, vocab_size].

        Returns:
            Gradients with respect to input, weight, bias, and attention_mask (set to None as it's not differentiable).
        """
        input, weight, bias, max_indices, fused = ctx.saved_tensors
        batch_size, seq_len, hidden_dim = input.shape
        _, vocab_size = weight.shape

        grad_output_f32 = grad_output.to(torch.float32)
        input_f32 = input.to(torch.float32)
        weight_f32 = weight.to(torch.float32)
        bias_f32 = bias.to(torch.float32) if bias is not None else None

        # Check whether requires grad
        input_require_grad = ctx.needs_input_grad[0]
        weight_require_grad = ctx.needs_input_grad[1]
        bias_require_grad = ctx.needs_input_grad[2]

        # Initialize gradients
        grad_input = torch.zeros_like(input, dtype=torch.float32) if input_require_grad else None
        grad_weight = torch.zeros_like(weight, dtype=torch.float32) if weight_require_grad else None
        grad_bias = torch.zeros_like(bias_f32, dtype=torch.float32) if bias_require_grad and (bias_f32 is not None) else None

        # TODO: t (int) will incur one recompilation when it changes. But this is the fastest way of compiled backward.
        def _sliced_backward(
            max_indices: Tensor,            # [batch_size, vocab_size]
            grad_output_f32: Tensor,        # [batch_size, vocab_size]
            input_f32_sliced: Tensor,       # [batch_size, hidden_dim]
            weight_f32: Tensor,             # [hidden_dim, vocab_size]
            grad_input_sliced: Optional[Tensor],      # [batch_size, hidden_dim]
            grad_weight: Optional[Tensor],            # [hidden_dim, vocab_size]
            grad_bias: Optional[Tensor],    # [vocab_size]
            t: int,
        ):
            """ Perform a sliced backward on step `t`, compute and update grad of input, weight and bias **in-place** """
            # Get mask for time step t
            mask_t = max_indices.eq(t)  # Shape [batch_size, vocab_size]

            # Gradient of logits with respect to input, weight, and bias
            grad_logits = grad_output_f32 * mask_t  # Shape [batch_size, vocab_size]
            if grad_input_sliced is not None:
                grad_input_sliced.add_(grad_logits @ weight_f32.T)
            if grad_weight is not None:
                grad_weight.add_(input_f32_sliced.T @ grad_logits)
            if grad_bias is not None:
                grad_bias.add_(grad_logits.sum(dim=0))  # Shape [vocab_size]

        if fused:
            sliced_backward_func = torch.compile(_sliced_backward, fullgraph=True)
        else:
            sliced_backward_func = _sliced_backward

        if max_indices is not None:  # Check if max_indices was computed
            for t in range(seq_len):
                input_f32_sliced = input_f32[:, t]      # [batch_size, hidden_dim]
                grad_input_sliced = grad_input[:, t] if grad_input is not None else None  # [batch_size, hidden_dim]
                if (t == 0) and fused:
                    torch._dynamo.mark_dynamic(max_indices, 0)
                    torch._dynamo.mark_dynamic(grad_output_f32, 0)
                    torch._dynamo.mark_dynamic(input_f32_sliced, 0)
                    if grad_input_sliced is not None:
                        torch._dynamo.mark_dynamic(grad_input_sliced, 0)
                
                sliced_backward_func(
                    max_indices, grad_output_f32, input_f32_sliced, weight_f32, grad_input_sliced, grad_weight, grad_bias, t
                )

        # Convert gradients back to the original dtype of input, weight, and bias
        if grad_input is not None:
            grad_input = grad_input.to(input.dtype)
        if grad_weight is not None:
            grad_weight = grad_weight.to(weight.dtype)
        if grad_bias is not None:
            grad_bias = grad_bias.to(bias.dtype)

        return grad_input, grad_weight, grad_bias, None  # attention_mask is not differentiable


def max_linear_mapping(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, attention_mask: Optional[Tensor] = None) -> Tensor:
    """
    Memory-efficient computation of max values along the sequence dimension using a custom autograd function.

    Args:
        input (Tensor): Shape [batch_size, seq_len, hidden_dim].
        weight (Tensor): Shape [hidden_dim, vocab_size].
        bias (Tensor, optional): Shape [vocab_size].
        attention_mask (Tensor, optional): Shape [batch_size, seq_len], where True means valid positions.

    Returns:
        max_values (Tensor): Shape [batch_size, vocab_size].
    """
    return MaxLinearMapperFunction.apply(input, weight, bias, attention_mask)


# Set tolerance based on dtype
dtype_atol_rtol = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (1e-3, 1e-2),
    torch.bfloat16: (1e-2, 1e-2),
}

def check_close(tensor1: Optional[Tensor], tensor2: Optional[Tensor]):
    if tensor1 is None and tensor2 is None:
        return True
    
    if isinstance(tensor1, Tensor) and isinstance(tensor2, Tensor):
        assert tensor1.dtype == tensor2.dtype
        atol, rtol = dtype_atol_rtol[tensor1.dtype]
        return torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol)
    else:
        return False

# Test 1: Verify consistency between forward and backward results
def test_forward_backward_consistency(input_tensor, weight_tensor, bias, attention_mask):
    output_custom = max_linear_mapping(input_tensor, weight_tensor, bias, attention_mask)

    logits = input_tensor @ weight_tensor
    if bias is not None:
        logits += bias
    if attention_mask is not None:
        logits = torch.where(
            attention_mask.unsqueeze(-1), logits, torch.full_like(logits, torch.finfo(logits.dtype).min)
        )
    output_direct, _ = logits.max(dim=1)

    # Verify the consistency of forward results
    forward_match = check_close(output_custom, output_direct)
    print(f"Forward results match: {forward_match}")

    # Back propagation verification
    output_custom.mean().backward(retain_graph=True)
    input_grad_custom = input_tensor.grad.clone() if input_tensor.grad is not None else None
    weight_grad_custom = weight_tensor.grad.clone() if weight_tensor.grad is not None else None
    if bias is not None:
        bias_grad_custom = bias.grad.clone()

    if input_tensor.grad is not None:
        input_tensor.grad.zero_()
    if weight_tensor.grad is not None:
        weight_tensor.grad.zero_()
    if bias is not None:
        bias.grad.zero_()
    output_direct.mean().backward()
    input_grad_direct = input_tensor.grad
    weight_grad_direct = weight_tensor.grad
    if bias is not None:
        bias_grad_direct = bias.grad

    backward_input_match = check_close(input_grad_custom, input_grad_direct)
    backward_weight_match = check_close(weight_grad_custom, weight_grad_direct)
    if bias is not None:
        backward_bias_match = check_close(bias_grad_custom, bias_grad_direct)

    print(f"Backward input gradients match: {backward_input_match}")
    print(f"Backward weight gradients match: {backward_weight_match}")
    if bias is not None:
        print(f"Backward bias gradients match: {backward_bias_match}")

    if input_grad_custom is not None:
        print(f"Max of input_grad_custom - input_grad_direct: {torch.max(torch.abs(input_grad_custom - input_grad_direct))}")
    if weight_grad_custom is not None:
        print(f"Max of weight_grad_custom - weight_grad_direct: {torch.max(torch.abs(weight_grad_custom - weight_grad_direct))}")
    if bias is not None:
        print(f"Max of bias_grad_custom - bias_grad_direct: {torch.max(torch.abs(bias_grad_custom - bias_grad_direct))}")

# Test 2: Verify the consistency of the forward result of no_grad
def test_no_grad_forward_consistency(input_tensor, weight_tensor, bias, attention_mask):
    with torch.no_grad():
        output_custom = max_linear_mapping(input_tensor, weight_tensor, bias, attention_mask)

        logits = input_tensor @ weight_tensor
        if bias is not None:
            logits += bias
        if attention_mask is not None:
            logits = torch.where(
                attention_mask.unsqueeze(-1), logits, torch.full_like(logits, torch.finfo(logits.dtype).min)
            )
        output_direct, _ = logits.max(dim=1)

        forward_match = check_close(output_custom, output_direct)
        print(f"no_grad forward results match: {forward_match}")

# Test 3: Display and compare GPU memory usage
def test_memory_usage(input_tensor, weight_tensor, bias, attention_mask):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_mem = torch.cuda.memory_allocated()

    output_custom = max_linear_mapping(input_tensor, weight_tensor, bias, attention_mask)
    torch.cuda.synchronize()
    custom_mem_usage = torch.cuda.memory_allocated() - start_mem
    print(f"Custom function memory usage: {custom_mem_usage / 1024**2:.2f} MB")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_mem = torch.cuda.memory_allocated()

    logits = input_tensor @ weight_tensor
    if bias is not None:
        logits += bias
    if attention_mask is not None:
        logits = torch.where(
            attention_mask.unsqueeze(-1), logits, torch.full_like(logits, torch.finfo(logits.dtype).min)
        )
    output_direct, _ = logits.max(dim=1)
    torch.cuda.synchronize()
    direct_mem_usage = torch.cuda.memory_allocated() - start_mem
    print(f"Direct computation memory usage: {direct_mem_usage / 1024**2:.2f} MB")

# Test 4: Display and compare forward and backward speeds
def test_execution_time(input_tensor, weight_tensor, bias, attention_mask):
    torch.cuda.synchronize()
    start_time = time.time()
    output_custom = max_linear_mapping(input_tensor, weight_tensor, bias, attention_mask)
    torch.cuda.synchronize()
    custom_forward_time = time.time() - start_time

    torch.cuda.synchronize()
    start_time = time.time()
    logits = input_tensor @ weight_tensor
    if bias is not None:
        logits += bias
    if attention_mask is not None:
        logits = torch.where(
            attention_mask.unsqueeze(-1), logits, torch.full_like(logits, torch.finfo(logits.dtype).min)
        )
    output_direct, _ = logits.max(dim=1)
    torch.cuda.synchronize()
    direct_forward_time = time.time() - start_time

    print(f"Custom function forward time: {custom_forward_time:.4f} seconds")
    print(f"Direct computation forward time: {direct_forward_time:.4f} seconds")

    # Backward pass timing
    torch.cuda.synchronize()
    start_time = time.time()
    output_custom.mean().backward(retain_graph=True)
    torch.cuda.synchronize()
    custom_backward_time = time.time() - start_time

    if input_tensor.grad is not None:
        input_tensor.grad.zero_()
    if weight_tensor.grad is not None:
        weight_tensor.grad.zero_()
    torch.cuda.synchronize()
    start_time = time.time()
    output_direct.mean().backward()
    torch.cuda.synchronize()
    direct_backward_time = time.time() - start_time

    print(f"Custom function backward time: {custom_backward_time:.4f} seconds")
    print(f"Direct computation backward time: {direct_backward_time:.4f} seconds")

def test_repeated_forward_backwards(input_tensor, weight_tensor, bias, attention_mask):
    # Forward and backward loop with n/2 batch_sizes
    bs = input_tensor.shape[0]
    while bs >= 2:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated()
        start_time = time.time()
        output_custom = max_linear_mapping(input_tensor[:bs], weight_tensor, bias, attention_mask[:bs])
        torch.cuda.synchronize()
        custom_forward_time = time.time() - start_time
        print(f"[bs{bs}] Custom function forward time: {custom_forward_time:.4f} seconds")
        custom_mem_usage = torch.cuda.memory_allocated() - start_mem
        print(f"[bs{bs}] Custom function forward memory usage: {custom_mem_usage / 1024**2:.2f} MB")

        # Backward pass timing
        torch.cuda.synchronize()
        start_time = time.time()
        output_custom.mean().backward()
        torch.cuda.synchronize()
        custom_backward_time = time.time() - start_time
        print(f"[bs{bs}] Custom function backward time: {custom_backward_time:.4f} seconds")

        bs = bs // 2


def execute_one_forward_backward(input_tensor, weight_tensor, bias, attention_mask):
    """ For warmup """
    output_custom = max_linear_mapping(input_tensor, weight_tensor, bias, attention_mask)
    output_custom.mean().backward()
    if input_tensor.grad is not None:
        input_tensor.grad.zero_()
    if weight_tensor.grad is not None:
        weight_tensor.grad.zero_()
    if bias is not None:
        bias.grad.zero_()
    del output_custom
    torch.cuda.empty_cache()

if __name__ == '__main__':
    # Hyperparameters
    batch_size = 32
    seq_len = 512
    hidden_dim = 768
    vocab_size = 32000
    device = torch.device("cuda:0")
    dtype = torch.float32

    # Create input tensor, weight tensor, and attention_mask
    input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype, requires_grad=True)
    weight_tensor = torch.randn(hidden_dim, vocab_size, device=device, dtype=dtype, requires_grad=False)
    # bias = torch.randn(vocab_size, device=device, dtype=dtype, requires_grad=True)
    bias = None
    attention_mask = torch.rand(batch_size, seq_len, device=device, dtype=dtype) > 0.5
    # attention_mask = None

    # Warmup
    execute_one_forward_backward(input_tensor, weight_tensor, bias, attention_mask) 
    # Run test functions separately
    test_forward_backward_consistency(input_tensor, weight_tensor, bias, attention_mask)
    test_no_grad_forward_consistency(input_tensor, weight_tensor, bias, attention_mask)
    test_memory_usage(input_tensor, weight_tensor, bias, attention_mask)
    test_execution_time(input_tensor, weight_tensor, bias, attention_mask)


    # # Test very large tensor (e.g., Gemma vocab)
    # # Hyperparameters
    # batch_size = 128
    # seq_len = 512
    # hidden_dim = 3072
    # vocab_size = 256000
    # device = torch.device("cuda:0")
    # dtype = torch.float32

    # # Create input tensor, weight tensor, and attention_mask
    # input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype, requires_grad=True)
    # weight_tensor = torch.randn(hidden_dim, vocab_size, device=device, dtype=dtype, requires_grad=True)
    # # bias = torch.randn(vocab_size, device=device, dtype=dtype, requires_grad=True)
    # bias = None
    # attention_mask = torch.rand(batch_size, seq_len, device=device, dtype=dtype) > 0.7
    # # attention_mask = None

    # # Warmup
    # execute_one_forward_backward(input_tensor, weight_tensor, bias, attention_mask)
    # # Test
    # test_repeated_forward_backwards(input_tensor, weight_tensor, bias, attention_mask)
