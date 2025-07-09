# Get NLTK stop word lists
import os
from pathlib import Path
import unicodedata

def get_nltk_stopwords():
    stopwords: dict[str, list[str]] = dict()

    cwd = os.path.split(os.path.realpath(__file__))[0]
    for filepath in (Path(cwd) / "nltk").iterdir():
        if filepath.is_file():
            with open(filepath, "r") as f:
                lines = [i.strip() for i in f.readlines() if i.strip()]
            stopwords[filepath.name] = lines
    
    return stopwords

def get_nltk_stopword_sets():
    stopwords = get_nltk_stopwords()
    stopword_sets = set()
    for stopword_list in stopwords.values():
        for i in stopword_list:
            stopword_sets.add(i)
    return stopword_sets

def get_nltk_stopword_list():
    return list(get_nltk_stopword_sets())



def get_lucene_stopwords():
    stopwords: dict[str, list[str]] = dict()

    cwd = os.path.split(os.path.realpath(__file__))[0]
    for filepath in (Path(cwd) / "lucene").iterdir():
        if filepath.is_file():
            with open(filepath, "r") as f:
                lines = [i.strip() for i in f.readlines() if i.strip() and (not i.startswith('#'))]
            stopwords[filepath.stem] = lines
    
    return stopwords

def get_lucene_stopword_sets():
    stopwords = get_lucene_stopwords()
    stopword_sets = set()
    for stopword_list in stopwords.values():
        for i in stopword_list:
            stopword_sets.add(i)
    return stopword_sets

def get_lucene_stopword_list():
    return list(get_lucene_stopword_sets())



def get_unicode_punctuation_list():
    punctuation_list = [
        chr(i) for i in range(0x110000)  # Unicode char range
        if unicodedata.category(chr(i)).startswith("P")  # Filter start with punction
    ]
    return punctuation_list


if __name__ == '__main__':
    nltk_stopwords = get_nltk_stopwords()
    lucene_stopwords = get_lucene_stopwords()

    nltk_stopword_sets = get_nltk_stopword_sets()
    lucene_stopword_sets = get_lucene_stopword_sets()
    
    print()
