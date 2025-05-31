from difflib import SequenceMatcher
import pandas as pd
from pathlib import Path
import json
from pyvi import ViTokenizer
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from config import DATA_DIR, CURRENT_DIR, JSON_FILE, STOPWORDS_FILE, TFIDF_MATRIX_FILE, VECTORIZER_FILE
from models.managers.mysql import fetch_data_from_mysql
from functools import lru_cache

@lru_cache(maxsize=1)
def get_data_path(filename):
    data_path = DATA_DIR / filename
    if data_path.exists():
        return data_path
    current_path = CURRENT_DIR / filename
    if current_path.exists():
        return current_path
    return Path(filename)

@lru_cache(maxsize=1)
def load_stopwords():
    try:
        stopwords_path = get_data_path(STOPWORDS_FILE)
        if not stopwords_path.exists():
            return []
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = [line.strip() for line in f if line.strip()]
        return stopwords
    except Exception:
        return []

def load_json_data(json_file):
    try:
        json_path = get_data_path(json_file)
        if not json_path.exists():
            return pd.DataFrame(columns=['question', 'answer'])
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df if all(col in df.columns for col in ['question', 'answer']) else pd.DataFrame(columns=['question', 'answer'])
    except Exception:
        return pd.DataFrame(columns=['question', 'answer'])

def tokenize_vietnamese(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(ViTokenizer.tokenize(text).split())

def prepare_data():
    json_df = load_json_data(JSON_FILE)
    if not json_df.empty:
        json_df['source'] = 'json'
    
    mysql_df = fetch_data_from_mysql()
    
    if not mysql_df.empty and not json_df.empty:
        df = pd.concat([json_df, mysql_df], ignore_index=True)
    elif not mysql_df.empty:
        df = mysql_df
    else:
        df = json_df
    
    if df.empty:
        df = pd.DataFrame(columns=['question', 'answer'])
    
    df['question'] = df['question'].astype(str).fillna('')
    df['answer'] = df['answer'].astype(str).fillna('')
    df = df.drop_duplicates(subset=['question'], keep='last').reset_index(drop=True)
    
    df['question_tokenized'] = df['question'].apply(tokenize_vietnamese)
    df['answer_tokenized'] = df['answer'].apply(tokenize_vietnamese)
    df['content'] = df['question_tokenized'] + ' ' + df['answer_tokenized']
    
    vietnamese_stopwords = load_stopwords()
    vectorizer, tfidf_matrix = create_tfidf_model(df, vietnamese_stopwords)
    
    return df, vectorizer, tfidf_matrix

def create_tfidf_model(df, stopwords):
    vectorizer = TfidfVectorizer(
        min_df=2,
        max_features=10000,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        ngram_range=(1, 2),
        stop_words=stopwords
    )
    tfidf_matrix = vectorizer.fit_transform(df['content'] if len(df) > 0 else ["fallback content"])
    
    try:
        tfidf_path = get_data_path(TFIDF_MATRIX_FILE)
        vectorizer_path = get_data_path(VECTORIZER_FILE)
        joblib.dump(tfidf_matrix, DATA_DIR / tfidf_path)
        joblib.dump(vectorizer, DATA_DIR / vectorizer_path)
    except Exception:
        pass
    return vectorizer, tfidf_matrix

qa_pairs = []
json_file = None

def find_best_match(question, threshold=0.55):
    if not qa_pairs:
        return None
        
    question_key = question.lower()
    keywords = set(question_key.split())
    
    potential_matches = []
    for qa in qa_pairs:
        qa_keywords = qa.get("keywords", set(qa["question"].split()))
        qa["keywords"] = qa_keywords
        
        if keywords.intersection(qa_keywords):
            potential_matches.append(qa)
    
    if not potential_matches:
        return None
        
    best_match = None
    best_score = 0
    
    for qa in potential_matches:
        seq_score = SequenceMatcher(None, question_key, qa["question"]).ratio()
        kw_score = len(keywords.intersection(qa["keywords"])) / len(keywords) if keywords else 0
        score = seq_score * 0.6 + kw_score * 0.4
        
        if score > best_score:
            best_score = score
            best_match = qa
    
    if best_match and best_score >= threshold:
        return {
            "answer": best_match["answer"],
            "source": best_match["source"],
            "line_number": best_match["line_number"]
        }
    return None