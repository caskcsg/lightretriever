import logging
logger = logging.getLogger(__name__)

def get_task_def_by_task_name_and_type_e5(task_name: str, task_type: str) -> str:
    if task_type in ['STS']:
        return "Retrieve semantically similar text."

    if task_type in ['Summarization']:
        return "Given a news summary, retrieve other semantically similar summaries"

    if task_type in ['BitextMining']:
        return "Retrieve parallel sentences."

    if task_type in ['Classification']:
        task_name_to_instruct: dict[str, str] = {
            'AmazonCounterfactualClassification': 'Classify a given Amazon customer review text as either counterfactual or not-counterfactual',
            'AmazonPolarityClassification': 'Classify Amazon reviews into positive or negative sentiment',
            'AmazonReviewsClassification': 'Classify the given Amazon review into its appropriate rating category',
            'Banking77Classification': 'Given a online banking query, find the corresponding intents',
            'EmotionClassification': 'Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise',
            'ImdbClassification': 'Classify the sentiment expressed in the given movie review text from the IMDB dataset',
            'MassiveIntentClassification': 'Given a user utterance as query, find the user intents',
            'MassiveScenarioClassification': 'Given a user utterance as query, find the user scenarios',
            'MTOPDomainClassification': 'Classify the intent domain of the given utterance in task-oriented conversation',
            'MTOPIntentClassification': 'Classify the intent of the given utterance in task-oriented conversation',
            'ToxicConversationsClassification': 'Classify the given comments as either toxic or not toxic',
            'TweetSentimentExtractionClassification': 'Classify the sentiment of a given tweet as either positive, negative, or neutral',
            # C-MTEB eval instructions
            'TNews': 'Classify the fine-grained category of the given news title',
            'IFlyTek': 'Given an App description text, find the appropriate fine-grained category',
            'MultilingualSentiment': 'Classify sentiment of the customer review into positive, neutral, or negative',
            'JDReview': 'Classify the customer review for iPhone on e-commerce platform into positive or negative',
            'OnlineShopping': 'Classify the customer review for online shopping into positive or negative',
            'Waimai': 'Classify the customer review from a food takeaway platform into positive or negative',
        }
        return task_name_to_instruct[task_name]

    if task_type in ['Clustering']:
        task_name_to_instruct: dict[str, str] = {
            'ArxivClusteringP2P': 'Identify the main and secondary category of Arxiv papers based on the titles and abstracts',
            'ArxivClusteringS2S': 'Identify the main and secondary category of Arxiv papers based on the titles',
            'BiorxivClusteringP2P': 'Identify the main category of Biorxiv papers based on the titles and abstracts',
            'BiorxivClusteringS2S': 'Identify the main category of Biorxiv papers based on the titles',
            'MedrxivClusteringP2P': 'Identify the main category of Medrxiv papers based on the titles and abstracts',
            'MedrxivClusteringS2S': 'Identify the main category of Medrxiv papers based on the titles',
            'RedditClustering': 'Identify the topic or theme of Reddit posts based on the titles',
            'RedditClusteringP2P': 'Identify the topic or theme of Reddit posts based on the titles and posts',
            'StackExchangeClustering': 'Identify the topic or theme of StackExchange posts based on the titles',
            'StackExchangeClusteringP2P': 'Identify the topic or theme of StackExchange posts based on the given paragraphs',
            'TwentyNewsgroupsClustering': 'Identify the topic or theme of the given news articles',
            # C-MTEB eval instructions
            'CLSClusteringS2S': 'Identify the main category of scholar papers based on the titles',
            'CLSClusteringP2P': 'Identify the main category of scholar papers based on the titles and abstracts',
            'ThuNewsClusteringS2S': 'Identify the topic or theme of the given news articles based on the titles',
            'ThuNewsClusteringP2P': 'Identify the topic or theme of the given news articles based on the titles and contents',
        }
        return task_name_to_instruct[task_name]

    if task_type in ['Reranking', 'PairClassification']:
        task_name_to_instruct: dict[str, str] = {
            'AskUbuntuDupQuestions': 'Retrieve duplicate questions from AskUbuntu forum',
            'MindSmallReranking': 'Retrieve relevant news articles based on user browsing history',
            'SciDocsRR': 'Given a title of a scientific paper, retrieve the titles of other relevant papers',
            'StackOverflowDupQuestions': 'Retrieve duplicate questions from StackOverflow forum',
            'SprintDuplicateQuestions': 'Retrieve duplicate questions from Sprint forum',
            'TwitterSemEval2015': 'Retrieve tweets that are semantically similar to the given tweet',
            'TwitterURLCorpus': 'Retrieve tweets that are semantically similar to the given tweet',
            # C-MTEB eval instructions
            'T2Reranking': 'Given a Chinese search query, retrieve web passages that answer the question',
            'MMarcoReranking': 'Given a Chinese search query, retrieve web passages that answer the question',
            'CMedQAv1': 'Given a Chinese community medical question, retrieve replies that best answer the question',
            'CMedQAv2': 'Given a Chinese community medical question, retrieve replies that best answer the question',
            'Ocnli': 'Retrieve semantically similar text.',
            'Cmnli': 'Retrieve semantically similar text.',
        }
        return task_name_to_instruct[task_name]

    if task_type in ['Retrieval']:
        if task_name.lower().startswith('cqadupstack'):
            return 'Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question'

        task_name_to_instruct: dict[str, str] = {
            'ArguAna': 'Given a claim, find documents that refute the claim',
            'ClimateFEVER': 'Given a claim about climate change, retrieve documents that support or refute the claim',
            'DBPedia': 'Given a query, retrieve relevant entity descriptions from DBPedia',
            'FEVER': 'Given a claim, retrieve documents that support or refute the claim',
            'FiQA2018': 'Given a financial question, retrieve user replies that best answer the question',
            'HotpotQA': 'Given a multi-hop question, retrieve documents that can help answer the question',
            'MSMARCO': 'Given a web search query, retrieve relevant passages that answer the query',
            'NFCorpus': 'Given a question, retrieve relevant documents that best answer the question',
            'NQ': 'Given a question, retrieve Wikipedia passages that answer the question',
            'QuoraRetrieval': 'Given a question, retrieve questions that are semantically equivalent to the given question',
            'SCIDOCS': 'Given a scientific paper title, retrieve paper abstracts that are cited by the given paper',
            'SciFact': 'Given a scientific claim, retrieve documents that support or refute the claim',
            'Touche2020': 'Given a question, retrieve detailed and persuasive arguments that answer the question',
            'TRECCOVID': 'Given a query on COVID-19, retrieve documents that answer the query',
            # C-MTEB eval instructions
            'T2Retrieval': 'Given a Chinese search query, retrieve web passages that answer the question',
            'MMarcoRetrieval': 'Given a web search query, retrieve relevant passages that answer the query',
            'DuRetrieval': 'Given a Chinese search query, retrieve web passages that answer the question',
            'CovidRetrieval': 'Given a question on COVID-19, retrieve news articles that answer the question',
            'CmedqaRetrieval': 'Given a Chinese community medical question, retrieve replies that best answer the question',
            'EcomRetrieval': 'Given a user query from an e-commerce website, retrieve description sentences of relevant products',
            'MedicalRetrieval': 'Given a medical question, retrieve user replies that best answer the question',
            'VideoRetrieval': 'Given a video search query, retrieve the titles of relevant videos',

            # MIRACL
            'MIRACLRetrieval': 'Given a question, retrieve Wikipedia passages that answer the question',
            ## MIRACL Indivisual Lang
            "MIRACLRetrieval-ar": "بناءً على استعلام بحث على الويب، استرجع المقاطع ذات الصلة التي تجيب على الاستعلام",
            "MIRACLRetrieval-bn": "একটি ওয়েব অনুসন্ধানের প্রশ্নের ভিত্তিতে, প্রাসঙ্গিক অনুচ্ছেদগুলি পুনরুদ্ধার করুন যা প্রশ্নের উত্তর দেয়",
            "MIRACLRetrieval-en": "Given a web search query, retrieve relevant passages that answer the query",
            "MIRACLRetrieval-es": "Dada una consulta de búsqueda web, recupera los pasajes relevantes que respondan a la consulta",
            "MIRACLRetrieval-fa": "با توجه به یک پرس‌وجوی جستجوی وب، بخش‌های مرتبطی را که به پرس‌وجو پاسخ می‌دهند بازیابی کنید",
            "MIRACLRetrieval-fi": "Kun annetaan verkkohakukysely, hae asiaankuuluvat tekstikohdat, jotka vastaavat kyselyyn",
            "MIRACLRetrieval-fr": "Étant donné une requête de recherche web, récupérez les passages pertinents qui répondent à la requête",
            "MIRACLRetrieval-hi": "एक वेब खोज क्वेरी देने पर, उन प्रासंगिक अनुच्छेदों को पुनः प्राप्त करें जो क्वेरी का उत्तर देते हैं",
            "MIRACLRetrieval-id": "Diberikan kueri pencarian web, ambil bagian teks yang relevan yang menjawab kueri tersebut",
            "MIRACLRetrieval-ja": "ウェブ検索クエリが与えられた場合、それに答える関連する文章を取得する",
            "MIRACLRetrieval-ko": "웹 검색 쿼리가 주어지면, 해당 쿼리에 대한 관련된 문단을 검색하세요",
            "MIRACLRetrieval-ru": "Данный веб-запрос, найдите соответствующие фрагменты, которые отвечают на запрос",
            "MIRACLRetrieval-sw": "Ukipewa swali la utafutaji wa wavuti, pata vifungu vinavyohusiana vinavyojibu swali",
            "MIRACLRetrieval-te": "ఒక వెబ్ శోధన ప్రశ్నను ఇచ్చినప్పుడు, ప్రశ్నకు సమాధానం ఇచ్చే సంబంధిత పేరాలను తిరిగి పొందండి",
            "MIRACLRetrieval-th": "เมื่อได้รับคำค้นหาบนเว็บ ดึงข้อความที่เกี่ยวข้องซึ่งตอบคำถามนั้น",
            "MIRACLRetrieval-zh": "给定一个网页搜索查询，检索能够回答该查询的相关段落",
            "MIRACLRetrieval-de": "Angesichts einer Websuchanfrage rufe relevante Passagen ab, die die Anfrage beantworten",
            "MIRACLRetrieval-yo": "Ti a ba fun ni ibeere wiwa wẹẹbu, gba awọn gbolohun ti o yẹ ti yoo dahun ibeere naa",
            
            # MLDR
            'MultiLongDocRetrieval': 'Given a question, retrieve documents that answer the question',

            # MKQA Test
            "MKQA": "Given a question, retrieve Wikipedia passages that answer the question",
            ## MKQA Indivisual Lang
            "MKQA-ar": "بناءً على استعلام بحث على الويب، استرجع المقاطع ذات الصلة التي تجيب على الاستعلام",
            "MKQA-da": "Givet en websøgningsforespørgsel, hent relevante passager, der besvarer forespørgslen",
            "MKQA-de": "Angesichts einer Websuchanfrage rufe relevante Passagen ab, die die Anfrage beantworten",
            "MKQA-en": "Given a web search query, retrieve relevant passages that answer the query",
            "MKQA-es": "Dada una consulta de búsqueda web, recupera los pasajes relevantes que respondan a la consulta",
            "MKQA-fi": "Kun annetaan verkkohakukysely, hae asiaankuuluvat tekstikohdat, jotka vastaavat kyselyyn",
            "MKQA-fr": "Étant donné une requête de recherche web, récupérez les passages pertinents qui répondent à la requête",
            "MKQA-he": "בהינתן שאילתת חיפוש באינטרנט, שלוף קטעים רלוונטיים העונים לשאילתה",
            "MKQA-hu": "Egy webes keresési lekérdezés alapján keresd meg a releváns szövegrészeket, amelyek megválaszolják a lekérdezést",
            "MKQA-it": "Data una query di ricerca web, recupera i passaggi pertinenti che rispondono alla query",
            "MKQA-ja": "ウェブ検索クエリが与えられた場合、それに答える関連する文章を取得する",
            "MKQA-ko": "웹 검색 쿼리가 주어지면, 해당 쿼리에 대한 관련된 문단을 검색하세요",
            "MKQA-km": "ដោយផ្អែកលើសំណួរស្វែងរកតាមអ៊ីនធឺណិត ស្វែងរកកថាភាគដែលពាក់ព័ន្ធដែលអាចឆ្លើយសំណួរនេះ",
            "MKQA-ms": "Diberikan pertanyaan carian web, dapatkan petikan yang relevan yang menjawab pertanyaan tersebut",
            "MKQA-nl": "Gegeven een webzoekopdracht, haal relevante passages op die de zoekopdracht beantwoorden",
            "MKQA-no": "Gitt et nettsøk, hent relevante avsnitt som besvarer søket",
            "MKQA-pl": "Podano zapytanie wyszukiwania w sieci, pobierz odpowiednie fragmenty odpowiadające na zapytanie",
            "MKQA-pt": "Dada uma consulta de pesquisa na web, recupere passagens relevantes que respondam à consulta",
            "MKQA-ru": "Данный веб-запрос, найдите соответствующие фрагменты, которые отвечают на запрос",
            "MKQA-sv": "Givet en webbsökning, hämta relevanta stycken som besvarar frågan",
            "MKQA-th": "เมื่อได้รับคำค้นหาบนเว็บ ดึงข้อความที่เกี่ยวข้องซึ่งตอบคำถามนั้น",
            "MKQA-tr": "Verilen bir web arama sorgusuna göre, sorguyu yanıtlayan ilgili pasajları getir",
            "MKQA-vi": "Với một truy vấn tìm kiếm trên web, truy xuất các đoạn văn bản có liên quan trả lời truy vấn",
            "MKQA-zh_cn": "给定一个网页搜索查询，检索能够回答该查询的相关段落",
            "MKQA-zh_hk": "給定一個網頁搜尋查詢，檢索能夠回答該查詢的相關段落",
            "MKQA-zh_tw": "給定一個網頁搜尋查詢，檢索能夠回答該查詢的相關段落",
        }

        # add lower case keys to match some beir names
        task_name_to_instruct.update({k.lower(): v for k, v in task_name_to_instruct.items()})
        # other cases where lower case match still doesn't work
        task_name_to_instruct['trec-covid'] = task_name_to_instruct['TRECCOVID']
        task_name_to_instruct['climate-fever'] = task_name_to_instruct['ClimateFEVER']
        task_name_to_instruct['dbpedia-entity'] = task_name_to_instruct['DBPedia']
        task_name_to_instruct['webis-touche2020'] = task_name_to_instruct['Touche2020']
        task_name_to_instruct['fiqa'] = task_name_to_instruct['FiQA2018']
        task_name_to_instruct['quora'] = task_name_to_instruct['QuoraRetrieval']

        return task_name_to_instruct[task_name]

    raise ValueError(f"No instruction config for task {task_name} with type {task_type}")


def get_detailed_instruct(task_description: str) -> str:
    if not task_description:
        return ''

    return 'Instruct: {}\nQuery: '.format(task_description)


def get_mteb_prompt(task_name: str, task_type: str, prompt_type: str):
    if prompt_type == "e5_ori":     # Special case for e5_ori
        if task_type in ["Reranking", "Retrieval"]:
            query_prompt = "query: "
            corpus_prompt = "passage: "
        else:
            query_prompt = "query: "
            corpus_prompt = "query: "
    
    elif prompt_type == 'e5':
        instruct = get_task_def_by_task_name_and_type_e5(task_name, task_type)
        query_prompt = get_detailed_instruct(instruct)
        if task_type in ["Reranking", "Retrieval"]:
            corpus_prompt = ""
        else:
            corpus_prompt = query_prompt    # Actually Unused
    
    elif prompt_type == 'llm2vec-e5-Meta-Llama-3-8B-Instruct':
        # Ref: https://github.com/McGill-NLP/llm2vec/blob/main/llm2vec/dataset/E5Data.py#L78
        #      https://github.com/McGill-NLP/llm2vec/blob/main/llm2vec/llm2vec.py#L148
        #      https://github.com/McGill-NLP/llm2vec/blob/main/test_configs/mteb/task_to_instructions.json
        instruct = get_task_def_by_task_name_and_type_e5(task_name, task_type)
        query_prompt = "<|start_header_id|>user<|end_header_id|>\n\n" + instruct + ": "
        if task_type in ["Reranking", "Retrieval"]:
            corpus_prompt = "<|start_header_id|>user<|end_header_id|>\n\n"
        else:
            corpus_prompt = query_prompt    # Actually Unused
    
    elif prompt_type == 'bge-en':
        query_prompt, corpus_prompt = "", ""
        if task_type in ['Retrieval']:
            query_prompt = "Represent this sentence for searching relevant passages: "
    
    elif prompt_type == 'bge-zh':
        query_prompt, corpus_prompt = "", ""
        if task_type in ['Retrieval']:
            query_prompt = "为这个句子生成表示以用于检索相关文章："
    
    else:
        raise NotImplementedError()
    
    return query_prompt, corpus_prompt
