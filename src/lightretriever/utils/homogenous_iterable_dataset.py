#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Interleave several iterable datasets (sources) into a single iterable dataset.
This dataset yeilds a homogenous batch from one dataset at each iteration.

@Time    :   2023/04/06
@Author  :   Ma (Ma787639046@outlook.com)
'''
import numpy as np
from copy import deepcopy
from typing import List, Optional, TypeVar, Iterator
from datasets.info import DatasetInfo
from datasets.splits import NamedSplit
from datasets.utils.py_utils import Literal
from datasets import DatasetDict, IterableDatasetDict
from datasets.arrow_dataset import Dataset
from datasets.iterable_dataset import IterableDataset, _BaseExamplesIterable, CyclingMultiSourcesExamplesIterable, RandomlyCyclingMultiSourcesExamplesIterable
from datasets.features.features import Features, _align_features, _check_if_features_can_be_aligned

DatasetType = TypeVar("DatasetType", Dataset, IterableDataset)

def interleave_datasets_homologenous(
    datasets: List[DatasetType],
    batch_size: int,
    probabilities: Optional[List[float]] = None,
    seed: Optional[int] = None,
    info: Optional[DatasetInfo] = None,
    split: Optional[NamedSplit] = None,
    stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "first_exhausted",
) -> DatasetType:
    """
    Interleave several datasets (sources) into a single dataset.
    The new dataset is constructed by alternating between the sources to get the examples.

    You can use this function on a list of [`Dataset`] objects, or on a list of [`IterableDataset`] objects.

        - If `probabilities` is `None` (default) the new dataset is constructed by cycling between each source to get the examples.
        - If `probabilities` is not `None`, the new dataset is constructed by getting examples from a random source at a time according to the provided probabilities.

    The resulting dataset ends when one of the source datasets runs out of examples except when `oversampling` is `True`,
    in which case, the resulting dataset ends when all datasets have ran out of examples at least one time.

    Note for iterable datasets:

    In a distributed setup or in PyTorch DataLoader workers, the stopping strategy is applied per process.
    Therefore the "first_exhausted" strategy on an sharded iterable dataset can generate less samples in total (up to 1 missing sample per subdataset per worker).

    Args:
        datasets (`List[Dataset]` or `List[IterableDataset]`):
            List of datasets to interleave.
        probabilities (`List[float]`, *optional*, defaults to `None`):
            If specified, the new dataset is constructed by sampling
            examples from one source at a time according to these probabilities.
        seed (`int`, *optional*, defaults to `None`):
            The random seed used to choose a source for each example.
        info ([`DatasetInfo`], *optional*):
            Dataset information, like description, citation, etc.
            <Added version="2.4.0"/>
        split ([`NamedSplit`], *optional*):
            Name of the dataset split.
            <Added version="2.4.0"/>
        stopping_strategy (`str`, defaults to `first_exhausted`):
            Two strategies are proposed right now, `first_exhausted` and `all_exhausted`.
            By default, `first_exhausted` is an undersampling strategy, i.e the dataset construction is stopped as soon as one dataset has ran out of samples.
            If the strategy is `all_exhausted`,  we use an oversampling strategy, i.e the dataset construction is stopped as soon as every samples of every dataset has been added at least once.
            Note that if the strategy is `all_exhausted`, the interleaved dataset size can get enormous:
            - with no probabilities, the resulting dataset will have `max_length_datasets*nb_dataset` samples.
            - with given probabilities, the resulting dataset will have more samples if some datasets have really low probability of visiting.
    Returns:
        [`Dataset`] or [`IterableDataset`]: Return type depends on the input `datasets`
        parameter. `Dataset` if the input is a list of `Dataset`, `IterableDataset` if the input is a list of
        `IterableDataset`.

    """

    if not datasets:
        raise ValueError("Unable to interleave an empty list of datasets.")
    for i, dataset in enumerate(datasets):
        if not isinstance(dataset, (Dataset, IterableDataset)):
            if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
                if not dataset:
                    raise ValueError(
                        f"Expected a list of Dataset objects or a list of IterableDataset objects, but element at position {i} "
                        "is an empty dataset dictionary."
                    )
                raise ValueError(
                    f"Dataset at position {i} has at least one split: {list(dataset)}\n"
                    f"Please pick one to interleave with the other datasets, for example: dataset['{next(iter(dataset))}']"
                )
            raise ValueError(
                f"Expected a list of Dataset objects or a list of IterableDataset objects, but element at position {i} is a {type(dataset).__name__}."
            )
        if i == 0:
            dataset_type, other_type = (
                (Dataset, IterableDataset) if isinstance(dataset, Dataset) else (IterableDataset, Dataset)
            )
        elif not isinstance(dataset, dataset_type):
            raise ValueError(
                f"Unable to interleave a {dataset_type.__name__} (at position 0) with a {other_type.__name__} (at position {i}). Expected a list of Dataset objects or a list of IterableDataset objects."
            )
    if stopping_strategy not in ["first_exhausted", "all_exhausted"]:
        raise ValueError(f"{stopping_strategy} is not supported. Please enter a valid stopping_strategy.")
    
    assert dataset_type is IterableDataset, "Currently only `IterableDataset` is supported for homogenous batch sampling."

    return _interleave_iterable_datasets_homologenous(
        datasets, batch_size=batch_size, probabilities=probabilities, seed=seed, info=info, split=split, stopping_strategy=stopping_strategy
    )


def _interleave_iterable_datasets_homologenous(
    datasets: List[IterableDataset],
    batch_size: int,
    probabilities: Optional[List[float]] = None,
    seed: Optional[int] = None,
    info: Optional[DatasetInfo] = None,
    split: Optional[NamedSplit] = None,
    stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "first_exhausted",
) -> IterableDataset:
    """
    Interleave several iterable datasets (sources) into a single iterable dataset.
    The new iterable dataset alternates between the sources to yield examples.
    If `probabilities = None` (default) the iterable dataset will cycles through the sources in order for each next example in the iteration.
    If `probabilities` is not `None, the iterable dataset will sample a random source according to the provided probabilities for each next examples in the iteration.

    <Added version="2.4.0"/>

    Args:
        datasets (`List[IterableDataset]`): list of datasets to interleave
        probabilities (`List[float]`, optional, default None): If specified, the new iterable dataset samples
            examples from one source at a time according to these probabilities.
        seed (`int`, optional, default None): The random seed used to choose a source for each example.
        stopping_strategy (`str`, defaults to `first_exhausted`):
            Two strategies are proposed right now.
            By default, `first_exhausted` is an undersampling strategy, i.e the dataset construction is stopped as soon as one dataset has ran out of samples.
            If the strategy is `all_exhausted`,  we use an oversampling strategy, i.e the dataset construction is stopped as soon as every samples of every dataset has been added at least once.
            Note that if the strategy is `all_exhausted`, the interleaved dataset size can get enormous:
            - with no probabilities, the resulting dataset will have max_length_datasets*nb_dataset samples.
            - with given probabilities, the resulting dataset will have more samples if some datasets have really low probability of visiting.

    Output:
        `datasets.IterableDataset`
    """
    datasets = [d._resolve_features() for d in datasets]

    # Perform checks
    _check_if_features_can_be_aligned([dset.features for dset in datasets])

    # TODO: improve this to account for a mix of ClassLabel and Value for example
    # right now it would keep the type of the first dataset in the list
    features = Features(
        {k: v for features in _align_features([dset.features for dset in datasets]) for k, v in features.items()}
    )

    ex_iterables = [d._ex_iterable for d in datasets]

    # Use cycling or random cycling of sources
    if probabilities is None:
        ex_iterable = CyclingMultiSourcesExamplesIterable(ex_iterables, stopping_strategy=stopping_strategy)
    else:
        generator = np.random.default_rng(seed)
        ex_iterable = HomogenousRandomlyCyclingMultiSourcesExamplesIterable(
            ex_iterables, batch_size=batch_size, generator=generator, probabilities=probabilities, stopping_strategy=stopping_strategy
        )
    # Set new info - we update the features
    # setting the features also ensures to fill missing columns with None
    if info is None:
        info = DatasetInfo.from_merge([d.info for d in datasets])
    else:
        info = info.copy()
    info.features = features
    # Get all the auth tokens per repository - in case the datasets come from different private repositories
    token_per_repo_id = {
        repo_id: token for dataset in datasets for repo_id, token in dataset._token_per_repo_id.items()
    }
    # Return new daset
    return IterableDataset(ex_iterable=ex_iterable, info=info, split=split, token_per_repo_id=token_per_repo_id)


class HomogenousRandomlyCyclingMultiSourcesExamplesIterable(CyclingMultiSourcesExamplesIterable):
    def __init__(
        self,
        ex_iterables: List[_BaseExamplesIterable],
        batch_size: int,
        generator: np.random.Generator,
        probabilities: Optional[List[float]] = None,
        stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "first_exhausted",
    ):
        super(HomogenousRandomlyCyclingMultiSourcesExamplesIterable, self).__init__(ex_iterables=ex_iterables, stopping_strategy=stopping_strategy)
        self.generator = generator
        self.probabilities = probabilities

        self.batch_size = batch_size

    @staticmethod
    def _iter_random_indices(
        rng: np.random.Generator,
        num_sources: int,
        random_batch_size: int,
        p: Optional[List[float]] = None,
    ) -> Iterator[int]:
        """Get an infinite iterator that randomly samples the index of the source to pick examples from."""
        # # debug
        # import torch.distributed as dist
        # _dist_is_initialized = dist.is_initialized()

        # rank = dist.get_rank() if _dist_is_initialized else -1
        # cnt = 0

        while True:
            # if _dist_is_initialized:
            #     synchronize_rng_states(rng_types, rng)
            
            class_idx_chosen: int = rng.choice(num_sources, p=p)
            # print(f"[Homogenous Dataset {rank}] Use class_idx_chosen = {class_idx_chosen}")
            for _ in range(random_batch_size):
                # print(f"[Homogenous Dataset {rank}] Yield data item {cnt}")
                # cnt += 1
                yield class_idx_chosen

    def _get_indices_iterator(self):
        rng = deepcopy(self.generator)
        # this is an infinite iterator that randomly samples the index of the source to pick examples from
        return self._iter_random_indices(rng, num_sources=len(self.ex_iterables), random_batch_size=self.batch_size, p=self.probabilities)
    
    def shuffle_data_sources(self, generator: np.random.Generator) -> "HomogenousRandomlyCyclingMultiSourcesExamplesIterable":
        """Shuffle the data sources of each wrapped examples iterable."""
        ex_iterables = [ex_iterable.shuffle_data_sources(generator) for ex_iterable in self.ex_iterables]
        return HomogenousRandomlyCyclingMultiSourcesExamplesIterable(
            ex_iterables,
            batch_size=self.batch_size,
            generator=generator,
            probabilities=self.probabilities,
            stopping_strategy=self.stopping_strategy,
        )

    def shard_data_sources(self, worker_id: int, num_workers: int) -> "HomogenousRandomlyCyclingMultiSourcesExamplesIterable":
        """Either keep only the requested shard, or propagate the request to the underlying iterable."""
        return HomogenousRandomlyCyclingMultiSourcesExamplesIterable(
            [iterable.shard_data_sources(worker_id, num_workers) for iterable in self.ex_iterables],
            self.batch_size,
            self.generator,
            self.probabilities,
            self.stopping_strategy,
        )


if __name__ == '__main__':
    from datasets.distributed import split_dataset_by_node

    # Test
    d1 = Dataset.from_dict({"a": [0] * 100000}).to_iterable_dataset(1024)
    d2 = Dataset.from_dict({"a": [1] * 10000}).to_iterable_dataset(1024)
    d3 = Dataset.from_dict({"a": [2] * 10000}).to_iterable_dataset(1024)

    dataset = interleave_datasets_homologenous([d1, d2, d3], batch_size=8, probabilities=[0.7, 0.2, 0.1], seed=42, stopping_strategy="all_exhausted")
    cnt = {idx: 0 for idx in range(3)}

    # Distributed datasets (Has bug)
    dataset1 = split_dataset_by_node(dataset, 0, 2)
    dataset2 = split_dataset_by_node(dataset, 1, 2)

    for i, (item1, item2) in enumerate(zip(dataset1, dataset2)):
        if i > 200:
            break
        print(item1)
        assert item1['a'] == item2['a']
        cnt[item1['a']] += 1
    
    num_sum = sum(cnt.values())
    percent = {idx: cnt[idx] / num_sum for idx in cnt.keys()}
    
    print(f"cnt: {cnt}")
    print(f"percent: {percent}")
    