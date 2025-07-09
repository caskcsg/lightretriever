from random import Random

INSTS = {
    "e5": {
        # Retrieval
        "agnews": ["Given a news title, retrieve the news descriptions that match the title"],
        "AllNLI": ["Given a premise, retrieve a hypothesis that is entailed by the premise", "Retrieve semantically similar text."],
        "altlex": ["Given a sentence, retrieve a paraphrase Wikipedia sentence", "Given a passage, retrieve a Wikipedia passage that forms paraphrase pairs"],
        "amazon-qa": ["Given a question, retrieve the corresponding answers from Amazon", "Given a question, retrieve an Amazon answer that solves the question"],
        "amazon_review_2018": ["Given a title, retrieve the corresponding reviews from Amazon", "Given a title, retrieve a Amazon review"],
        "amazon_review_2018_1m": ["Given a title, retrieve the corresponding reviews from Amazon", "Given a title, retrieve a Amazon review"],
        "ccnews_title_text": ["Given a news title, retrieve articles that match the title"],
        "cMedQA2": ["Given a Chinese community medical question, retrieve replies that best answer the question"],
        "cnn_dailymail": ["Given highlight sentences, retrieve an relevant article that match the sentences"],
        "cnn_dailymail_splitted": ["Given a news article, retrieve its highlight sentences", "Given a passage, retrieve the corresponding highlight sentences"],
        "coco_captions": ["Given a caption, retrieve a similar caption from the same image", "Given a caption, retrieve a caption that describes the same image"],
        "codesearchnet": ["Given a comment of the function code, retrieve the corresponding code blocks"],
        "dureader": ["Given a Chinese search query, retrieve web passages that answer the question"],
        "eli5_question_answer": ["Provided a user question, retrieve the highest voted answers on Reddit ELI5 forum"],
        "fever": ["Given a claim, retrieve documents that support or refute the claim"],
        "fiqa": ["Given a financial question, judge whether the user replies best answer the question"],
        "flickr30k_captions": ["Given a caption, retrieve a similar caption from the same image", "Given a caption, retrieve a caption that describes the same image"],
        "gooaq_pairs": ["Given a web search query, retrieve the corresponding answers from Google"],
        "hotpotqa": ["Given a multi-hop question, retrieve documents that can help answer the question"],
        "medmcqa": ["Given a medical query, retrieve relevant passages that answer the query", "Given a medical question, retrieve passages that answer the question"],
        "miracl": ["Given a question, retrieve Wikipedia passages that answer the question", "Retrieve Wikipedia passages that answer the question"],
        "MLDR": ["Given a question, retrieve documents that answer the question", "Retrieve documents that answer the question"],
        
        "mmarco-ar": ["بناءً على استعلام بحث على الويب، استرجع المقاطع ذات الصلة التي تجيب على الاستعلام"],# ar
        "indicmarco-bn": ["একটি ওয়েব অনুসন্ধানের প্রশ্নের ভিত্তিতে, প্রাসঙ্গিক অনুচ্ছেদগুলি পুনরুদ্ধার করুন যা প্রশ্নের উত্তর দেয়"],
        "mmarco-de": ["Angesichts einer Websuchanfrage rufe relevante Passagen ab, die die Anfrage beantworten"],# de
        "mmarco-en": ["Given a web search query, retrieve relevant passages that answer the query"],# en
        "mmarco-es": ["Dada una consulta de búsqueda web, recupera los pasajes relevantes que respondan a la consulta"],# es
        "neumarco-fa": ["با توجه به یک پرس‌وجوی جستجوی وب، بخش‌های مرتبطی را که به پرس‌وجو پاسخ می‌دهند بازیابی کنید"],# fa
        "mmarco-fr": ["Étant donné une requête de recherche web, récupérez les passages pertinents qui répondent à la requête"],# fr
        "mmarco-hi": ["एक वेब खोज क्वेरी देने पर, उन प्रासंगिक अनुच्छेदों को पुनः प्राप्त करें जो क्वेरी का उत्तर देते हैं"],# hi
        "mmarco-id": ["Diberikan kueri pencarian web, ambil bagian teks yang relevan yang menjawab kueri tersebut"],# id
        "mmarco-it": ["Data una query di ricerca web, recupera i passaggi pertinenti che rispondono alla query"],# it
        "mmarco-ja": ["ウェブ検索クエリが与えられた場合、それに答える関連する文章を取得する"],# ja
        "marco-ko": ["웹 검색 쿼리가 주어지면, 해당 쿼리에 대한 관련된 문단을 검색하세요"], # ko
        "mmarco-nl": ["Gegeven een webzoekopdracht, haal relevante passages op die de zoekopdracht beantwoorden"], # nl
        "mmarco-pt": ["Dada uma consulta de pesquisa na web, recupere passagens relevantes que respondam à consulta"],# pt
        "indicmarco-te": ["ఒక వెబ్ శోధన ప్రశ్నను ఇచ్చినప్పుడు, ప్రశ్నకు సమాధానం ఇచ్చే సంబంధిత పేరాలను తిరిగి పొందండి"],
        "mmarco-ru": ["Данный веб-запрос, найдите соответствующие фрагменты, которые отвечают на запрос"],# ru
        "mmarco-vi": ["Với một truy vấn tìm kiếm trên web, truy xuất các đoạn văn bản có liên quan trả lời truy vấn"],# vi
        "mmarco-zh": ["给定一个网页搜索查询，检索能够回答该查询的相关段落"],# zh

        "mr_tydi_combined": ["Given a question, retrieve Wikipedia passages that answer the question", "Retrieve Wikipedia passages that answer the question"],
        "msmarco": ["Given a web search query, retrieve relevant passages that answer the query"],
        "nfcorpus": ["Given a question, judge whether the document best answers the question"],
        "npr": ["Given a news title, retrieve articles that match the title"],
        "nq": ["Given a question, retrieve Wikipedia passages that answer the question", "Retrieve Wikipedia passages that answer the question"],
        "others": ["Given a web search query, retrieve relevant passages that answer the query"],
        "PAQ_pairs": ["Given a question, retrieve Wikipedia passages that answer the question", "Retrieve Wikipedia passages that answer the question"],
        "PAQ_pairs_100k": ["Given a question, retrieve Wikipedia passages that answer the question", "Retrieve Wikipedia passages that answer the question"],
        "quora_duplicates_triplets": ["Given a question, retrieve questions that are semantically equivalent to the given question", "Find questions that have the same meaning as the input question"],
        "S2ORC_title_abstract": ["Given a title, retrieve the abstract from scientific papers", "Given a title, retrieve abstracts from scientific papers that match the title"],
        "S2ORC_title_abstract_100k": ["Given a title, retrieve the abstract from scientific papers", "Given a title, retrieve abstracts from scientific papers that match the title"],
        "scifact": ["Given a scientific claim, judge whether the document supports or refutes the claim"],
        "searchQA_top5_snippets": ["Given a question, retrieve text snippets that answer the question", "Retrieve text snippets that answer the question"],
        "sentence-compression": ["Given a sentence, retrieve a short sentence that is semantically equivalent to the given sentence"],
        "SimpleWiki": ["Given a Wikipedia sentence, retrieve sentences that are semantically equivalent to the given sentence", "Retrieve semantically similar text."],
        "specter_train_triples": ["Given a title, retrieve semantic related titles", "Retrieve semantic related titles from scientific publications"],
        "squad_pairs": ["Given a question, retrieve Wikipedia passages that answer the question", "Retrieve Wikipedia passages that answer the question"],
        "stackexchange_duplicate_questions_body_body": ["Retrieve duplicate passages from StackOverflow forum"],
        "stackexchange_duplicate_questions_title-body_title-body": ["Retrieve duplicate questions and passages from StackOverflow forum"],
        "stackexchange_duplicate_questions_title_title": ["Retrieve duplicate questions from StackOverflow forum"],
        "t2ranking": ["Given a Chinese search query, retrieve web passages that answer the question"],
        "trivia": ["Given a question, retrieve Wikipedia passages that answer the question", "Retrieve Wikipedia passages that answer the question"],
        "WikiAnswers": ["Retrieve duplicate questions from Wikipedia"],
        "WikiAnswers_100k": ["Retrieve duplicate questions from Wikipedia"],
        "wikihow": ["Given a summary, retrieve Wikipedia passages that match the summary"],
        "xsum": ["Given a news summary, retrieve articles that match the summary"],
        "yahoo_answers_question_answer": ["Given a question, retrieve Yahoo answers that solve the question"],
        "yahoo_answers_title_answer": ["Given a title, retrieve Yahoo answers that match the title"],
        "yahoo_answers_title_question": ["Given a title, retrieve the corresponding Yahoo questions"],
        
        # Clustering
        # "arxiv_clustering": ["Identify the main and secondary category of Arxiv papers based on the titles and abstracts"],
        # "biorxiv_clustering": ["Identify the main category of Biorxiv papers based on the titles and abstracts"],
        # "medrxiv_clustering": ["Identify the main category of Medrxiv papers based on the titles and abstracts"],
        "AllArxiv_clustering": [""],  # Merge above three to one file, and add prompt to the training set

        # Classification
        "All_classification": [""],   # Merge the web collections to one file, and add prompt to the training set

        # MKQA Test
        "MKQA": ["Given a question, retrieve Wikipedia passages that answer the question"],
    },
    "instructor": {
        # Retrieval
        "msmarco": ["Represent the query for retrieving supporting documents: ", "Represent the query: "],
        "nq": ["Represent the Wikipedia question for retrieving supporting documents: ", "Represent the Wikipedia question: "],
        "trivia": ["Represent the Wikipedia question for retrieving supporting documents: ", "Represent the Wikipedia question: "],
        "AllNLI": ["Represent the statement: ", "Represent the post: "],
        "eli5_question_answer": ["Represent the user question from Reddit ELI5 forum for retrieving the highest voted answers: ", "Represent the user question: ", "Represent the user question from Reddit ELI5 forum: "],
        "dureader": ["Represent the Chinese search query for retrieving supporting web passages: ", "Represent the Chinese search query: "],
        "t2ranking": ["Represent the Chinese search query for retrieving supporting web passages: ", "Represent the Chinese search query: "],
        "fever": ["Represent the fact for retrieving supporting evidence: ", "Represent the fact: ", "Represent the query: "],
        "hotpotqa": ["Represent the multi-hop Wikipedia question for retrieving supporting documents: ", "Represent the multi-hop Wikipedia question: ", "Represent the Wikipedia question: ", "Represent the question: "],
        "squad_pairs": ["Represent the Wikipedia question for retrieving supporting documents: ", "Represent the Wikipedia question: "],
        "quora_duplicates_triplets": ["Represent the Quora question for retrieving questions: ", "Represent the question: "],
        "mr_tydi_combined": ["Represent the Wikipedia question for retrieving supporting documents: ", "Represent the Wikipedia question: "],
        "miracl": ["Represent the Wikipedia question for retrieving supporting documents: ", "Represent the Wikipedia question: "],
        "gooaq_pairs": ["Represent the query for retrieving supporting documents: ", "Represent the query: "],
        "codesearchnet": ["Represent the comment of the function code for retrieving the corresponding code blocks: ", "Represent the comment of the function code: ", "Represent the comment: "],
        "stackexchange_duplicate_questions_title_title": ["Represent the title: ", "Represent the duplicate questions: "],
        "wikihow": ["Represent the summary for retrieving corresponding Wikipedia passages: ", "Represent the summary: "],
        "yahoo_answers_question_answer": ["Represent the question for retrieving answers: ", "Represent the question: "],
        "yahoo_answers_title_answer": ["Represent the title for retrieving answers: ", "Represent the title: "],
        "yahoo_answers_title_question": ["Represent the question for retrieving questions", "Represent the question: "],
        "agnews": ["Represent the news title for retrieving the corresponding descriptions that match the title", "Represent the news title: "],
        "medi": [""],

        # Clustering
        "AllArxiv_clustering": ["Represent the passage for clustering: ", "Represent the passage: "],  # Merge above three to one file
    },
    "e5_reranker": {
        # Relevance Judgement
        "agnews": ["Given a news title, judge whether the news description matches the title"],
        "AllNLI": ["Given a premise, judge whether the hypothesis is entailed by the premise", "Judge whether the text is semantically similar."],
        "altlex": ["Given a sentence, judge whether the Wikipedia sentence is a paraphrase", "Given a passage, judge whether the Wikipedia passage forms a paraphrase pair"],
        "amazon-qa": ["Given a question, judge whether the answer from Amazon solves the question"],
        "amazon_review_2018": ["Given a title, judge whether the review from Amazon matches the title"],
        "amazon_review_2018_1m": ["Given a title, judge whether the review from Amazon matches the title"],
        "ccnews_title_text": ["Given a news title, judge whether the article matches the title"],
        "cnn_dailymail": ["Given highlight sentences, judge whether the article matches the sentences"],
        "cnn_dailymail_splitted": ["Given a news article, judge whether the highlight sentences match the article", "Given a passage, judge whether the highlight sentences match the passage"],
        "coco_captions": ["Given a caption, judge whether another caption describes the same image"],
        "codesearchnet": ["Given a comment of the function code, judge whether the code block matches the comment"],
        "dureader": ["Given a Chinese search query, judge whether the web passage answers the question"],
        "eli5_question_answer": ["Given a user question, judge whether the answer from Reddit ELI5 forum is relevant"],
        "fever": ["Given a claim, judge whether the document supports or refutes the claim"],
        "flickr30k_captions": ["Given a caption, judge whether another caption describes the same image"],
        "gooaq_pairs": ["Given a web search query, judge whether the answer from Google is relevant"],
        "hotpotqa": ["Given a multi-hop question, judge whether the document helps answer the question"],
        "medmcqa": ["Given a medical query, judge whether the passage answers the query"],
        "miracl": ["Given a question, judge whether the Wikipedia passage answers the question"],
        "MLDR": ["Given a question, judge whether the document answers the question"],
        "mr_tydi_combined": ["Given a question, judge whether the Wikipedia passage answers the question"],
        "msmarco": ["Given a web search query, judge whether the passage answers the query"],
        "npr": ["Given a news title, judge whether the article matches the title"],
        "nq": ["Given a question, judge whether the Wikipedia passage answers the question"],
        "PAQ_pairs": ["Given a question, judge whether the Wikipedia passage answers the question"],
        "PAQ_pairs_100k": ["Given a question, judge whether the Wikipedia passage answers the question"],
        "quora_duplicates_triplets": ["Given a question, judge whether another question is semantically equivalent"],
        "S2ORC_title_abstract": ["Given a title, judge whether the abstract from a scientific paper matches the title"],
        "S2ORC_title_abstract_100k": ["Given a title, judge whether the abstract from a scientific paper matches the title"],
        "searchQA_top5_snippets": ["Given a question, judge whether the text snippet answers the question"],
        "sentence-compression": ["Given a sentence, judge whether a short sentence is semantically equivalent"],
        "SimpleWiki": ["Given a Wikipedia sentence, judge whether another sentence is semantically equivalent"],
        "specter_train_triples": ["Given a title, judge whether another title from scientific publications is semantically related"],
        "squad_pairs": ["Given a question, judge whether the Wikipedia passage answers the question"],
        "stackexchange_duplicate_questions_body_body": ["Judge whether the passages from StackOverflow forum are duplicates"],
        "stackexchange_duplicate_questions_title-body_title-body": ["Judge whether the questions and passages from StackOverflow forum are duplicates"],
        "stackexchange_duplicate_questions_title_title": ["Judge whether the questions from StackOverflow forum are duplicates"],
        "t2ranking": ["Given a Chinese search query, judge whether the web passage answers the question"],
        "trivia": ["Given a question, judge whether the Wikipedia passage answers the question"],
        "WikiAnswers": ["Judge whether the questions from Wikipedia are duplicates"],
        "WikiAnswers_100k": ["Judge whether the questions from Wikipedia are duplicates"],
        "wikihow": ["Given a summary, judge whether the Wikipedia passage matches the summary"],
        "xsum": ["Given a news summary, judge whether the article matches the summary"],
        "yahoo_answers_question_answer": ["Given a question, judge whether the Yahoo answer solves the question"],
        "yahoo_answers_title_answer": ["Given a title, judge whether the Yahoo answer matches the title"],
        "yahoo_answers_title_question": ["Given a title, judge whether the Yahoo question matches the title"],
        
        # Clustering
        "AllArxiv_clustering": ["Judge whether the passage belongs to the same category as the Arxiv paper"],

        # Classification
        "All_classification": ["Judge whether the document belongs to the appropriate category"],

        # MKQA Test
        "MKQA": ["Given a question, judge whether the Wikipedia passage answers the question"],
    },
    "instructor_reranker": {
        # Relevance Judgement
        "msmarco": ["Judge whether the query is relevant to the document"],
        "nq": ["Judge whether the Wikipedia question is relevant to the document"],
        "trivia": ["Judge whether the Wikipedia question is relevant to the document"],
        "AllNLI": ["Judge whether the statement is relevant", "Judge whether the post is relevant"],
        "eli5_question_answer": ["Judge whether the user question from Reddit ELI5 forum is relevant to the answer"],
        "dureader": ["Judge whether the Chinese search query is relevant to the web passage"],
        "t2ranking": ["Judge whether the Chinese search query is relevant to the web passage"],
        "fever": ["Judge whether the fact is relevant to the evidence"],
        "hotpotqa": ["Judge whether the multi-hop Wikipedia question is relevant to the document"],
        "squad_pairs": ["Judge whether the Wikipedia question is relevant to the document"],
        "quora_duplicates_triplets": ["Judge whether the Quora question is relevant to another question"],
        "mr_tydi_combined": ["Judge whether the Wikipedia question is relevant to the document"],
        "miracl": ["Judge whether the Wikipedia question is relevant to the document"],
        "gooaq_pairs": ["Judge whether the query is relevant to the document"],
        "codesearchnet": ["Judge whether the comment of the function code is relevant to the code block"],
        "stackexchange_duplicate_questions_title_title": ["Judge whether the title is relevant to the duplicate questions"],
        "wikihow": ["Judge whether the summary is relevant to the Wikipedia passage"],
        "yahoo_answers_question_answer": ["Judge whether the question is relevant to the answer"],
        "yahoo_answers_title_answer": ["Judge whether the title is relevant to the answer"],
        "yahoo_answers_title_question": ["Judge whether the question is relevant to another question"],
        "agnews": ["Judge whether the news title is relevant to the description"],
        "medi": ["Judge whether the media content is relevant"],

        # Clustering
        "AllArxiv_clustering": ["Judge whether the passage is relevant to the cluster"],
    }
}

def get_prompt(
    prompt_type: str, 
    task_name: str, 
    rng: Random
) -> str:
    """ Add prompt for query of QA/Retrieval/Rerank tasks. 
        [TODO: Remove this line] Also add prompt on both side for Symmetrical tasks (e.g. NLI) 
        Note: 
        1. For Retrieval tasks, NO passage prompt is used.
        2. For Reranking tasks, a prompt of `\nPassage: ` is always added in front of passage text.
    """
    if prompt_type in ['e5', 'e5_reranker']:
        instruct_list: list[str] = INSTS[prompt_type][task_name]
        instruct: str = instruct_list[0] if len(instruct_list) == 1 else rng.choice(instruct_list)
        prompt = 'Instruct: {}\nQuery: '.format(instruct) if instruct != '' else ''
    elif prompt_type == 'bge':
        if any(i in task_name for i in ["NLI", "altlex", "captions", "duplicate", "SimpleWiki", "specter_train_triples", "WikiAnswers"]):
            # No need to add prompt which is not retrieval tasks
            prompt = ""
        else:
            # Only add query prompt to **Retrieval Tasks**
            prompt = "Represent this sentence for searching relevant passages: "
    elif prompt_type == 'reranker':   # Default Reranker Prompts
        prompt = "Instruct: Given a Query and a Passage, determine if the Passage answers or is semantically similar to the Query.\nQuery: "
    elif prompt_type == 'reranker_noinst':
        prompt = "Query: "
    elif prompt_type == 'reranker_yes':
        prompt = "Instruct: Given a Query and a Passage, determine if the Passage answers or is semantically similar to the Query by returning yes or no.\nQuery: "
    else:
        instruct_list: list[str] = INSTS[prompt_type][task_name]
        prompt: str = instruct_list[0] if len(instruct_list) == 1 else rng.choice(instruct_list)

    return prompt


def get_prompt_list(
    prompt_type: str, 
    task_name: str, 
    num: int,
    seed: int = 42,
):
    """ Sample a prompt `num` times, returns List of prompts """
    rng = Random(seed)

    prompts: list[str] = []
    for _ in range(num):
        prompt = get_prompt(prompt_type, task_name, rng)
        prompts.append(prompt)

    return prompts

