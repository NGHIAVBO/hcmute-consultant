import logging
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

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('views.log', encoding='utf-8'),
        logging.StreamHandler()  # Optional: keep console output for debugging
    ]
)
logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_data_path(filename):
    data_path = DATA_DIR / filename
    if data_path.exists():
        logger.info(f"Found file {filename} at {data_path}")
        return data_path
    current_path = CURRENT_DIR / filename
    if current_path.exists():
        logger.info(f"Found file {filename} at {current_path}")
        return current_path
    logger.warning(f"File {filename} not found in DATA_DIR or CURRENT_DIR, using {filename}")
    return Path(filename)

@lru_cache(maxsize=1)
def load_stopwords():
    try:
        stopwords_path = get_data_path(STOPWORDS_FILE)
        if not stopwords_path.exists():
            logger.warning(f"Stopwords file {stopwords_path} not found")
            return []
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(stopwords)} stopwords from {stopwords_path}")
        return stopwords
    except Exception as e:
        logger.error(f"Error loading stopwords from {stopwords_path}: {str(e)}")
        return []

def load_json_data(json_file):
    try:
        json_path = get_data_path(json_file)
        logger.debug(f"Attempting to load JSON file: {json_path}")
        if not json_path.exists():
            logger.error(f"JSON file {json_path} not found")
            return pd.DataFrame(columns=['question', 'answer', 'source'])
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not data:
            logger.warning(f"JSON file {json_path} is empty")
            return pd.DataFrame(columns=['question', 'answer', 'source'])
        df = pd.DataFrame(data)
        if not all(col in df.columns for col in ['question', 'answer']):
            logger.error(f"JSON file {json_path} missing required columns 'question' or 'answer'")
            return pd.DataFrame(columns=['question', 'answer', 'source'])
        df['source'] = 'json'
        logger.info(f"Successfully loaded {len(df)} records from {json_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading JSON data from {json_path}: {str(e)}")
        return pd.DataFrame(columns=['question', 'answer', 'source'])

def tokenize_vietnamese(text):
    if not isinstance(text, str) or not text.strip():
        logger.debug("Empty or non-string input for tokenization")
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    try:
        tokenized = " ".join(ViTokenizer.tokenize(text).split())
        logger.debug(f"Tokenized text: {text[:50]}... -> {tokenized[:50]}...")
        return tokenized
    except Exception as e:
        logger.error(f"Error tokenizing text: {str(e)}")
        return text

def prepare_data():
    try:
        logger.info("Starting data preparation")
        json_df = load_json_data(JSON_FILE)
        if not json_df.empty:
            logger.info(f"Loaded {len(json_df)} JSON records")
        
        mysql_df = fetch_data_from_mysql()
        if not mysql_df.empty:
            logger.info(f"Loaded {len(mysql_df)} MySQL records")
        
        if not mysql_df.empty and not json_df.empty:
            df = pd.concat([json_df, mysql_df], ignore_index=True)
            logger.info(f"Combined {len(json_df)} JSON and {len(mysql_df)} MySQL records into {len(df)} total records")
        elif not mysql_df.empty:
            df = mysql_df
            logger.info("Using MySQL data only")
        else:
            df = json_df
            logger.info("Using JSON data only")
        
        if df.empty:
            logger.warning("No data loaded from JSON or MySQL")
            df = pd.DataFrame(columns=['question', 'answer', 'source'])
        
        df['question'] = df['question'].astype(str).fillna('')
        df['answer'] = df['answer'].astype(str).fillna('')
        df = df.drop_duplicates(subset=['question'], keep='last').reset_index(drop=True)
        logger.info(f"After deduplication, {len(df)} records remain")
        
        df['question_tokenized'] = df['question'].apply(tokenize_vietnamese)
        df['answer_tokenized'] = df['answer'].apply(tokenize_vietnamese)
        df['content'] = df['question_tokenized'] + ' ' + df['answer_tokenized']
        logger.debug("Completed tokenization and content creation")
        
        vietnamese_stopwords = load_stopwords()
        vectorizer, tfidf_matrix = create_tfidf_model(df, vietnamese_stopwords)
        
        logger.info("Data preparation completed successfully")
        return df, vectorizer, tfidf_matrix
    except Exception as e:
        logger.error(f"Error in prepare_data: {str(e)}")
        return pd.DataFrame(columns=['question', 'answer', 'source']), None, None

def create_tfidf_model(df, stopwords):
    try:
        logger.debug("Creating TF-IDF model")
        vectorizer = TfidfVectorizer(
            min_df=2,
            max_features=10000,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1, 2),
            stop_words=stopwords
        )
        content = df['content'] if len(df) > 0 else ["fallback content"]
        tfidf_matrix = vectorizer.fit_transform(content)
        logger.info(f"Created TF-IDF matrix with shape {tfidf_matrix.shape}")
        
        try:
            tfidf_path = get_data_path(TFIDF_MATRIX_FILE)
            vectorizer_path = get_data_path(VECTORIZER_FILE)
            joblib.dump(tfidf_matrix, tfidf_path)
            joblib.dump(vectorizer, vectorizer_path)
            logger.info(f"Saved TF-IDF matrix to {tfidf_path} and vectorizer to {vectorizer_path}")
        except Exception as e:
            logger.warning(f"Error saving TF-IDF model: {str(e)}")
        
        return vectorizer, tfidf_matrix
    except Exception as e:
        logger.error(f"Error creating TF-IDF model: {str(e)}")
        return None, None

qa_pairs = []

def find_best_match(question, threshold=0.55):
    global qa_pairs
    if not qa_pairs:
        try:
            json_df = load_json_data(JSON_FILE)
            if not json_df.empty:
                qa_pairs = [
                    {
                        "question": row['question'].lower(),
                        "answer": row['answer'],
                        "source": row['source'],
                        "line_number": idx + 1,
                        "keywords": set(tokenize_vietnamese(row['question']).split())
                    }
                    for idx, row in json_df.iterrows()
                ]
                logger.info(f"Loaded {len(qa_pairs)} QA pairs for find_best_match")
            else:
                logger.warning("No QA pairs loaded due to empty JSON data")
                return None
        except Exception as e:
            logger.error(f"Error loading QA pairs: {str(e)}")
            return None
        
    logger.debug(f"Searching for best match for question: {question}")
    question_key = question.lower()
    keywords = set(tokenize_vietnamese(question_key).split())
    
    potential_matches = []
    for qa in qa_pairs:
        qa_keywords = qa.get("keywords", set(qa["question"].split()))
        qa["keywords"] = qa_keywords
        
        if keywords.intersection(qa_keywords):
            potential_matches.append(qa)
    
    logger.debug(f"Found {len(potential_matches)} potential matches")
    if not potential_matches:
        logger.info("No potential matches found")
        return None
        
    best_match = None
    best_score = 0
    
    for qa in potential_matches:
        seq_score = SequenceMatcher(None, question_key, qa["question"]).ratio()
        kw_score = len(keywords.intersection(qa["keywords"])) / len(keywords) if keywords else 0
        score = seq_score * 0.6 + kw_score * 0.4
        logger.debug(f"Evaluated QA pair: seq_score={seq_score:.2f}, kw_score={kw_score:.2f}, total_score={score:.2f}")
        
        if score > best_score:
            best_score = score
            best_match = qa
    
    if best_match and best_score >= threshold:
        logger.info(f"Best match found with score {best_score:.2f}: {best_match['question']}")
        return {
            "answer": best_match["answer"],
            "source": best_match["source"],
            "line_number": best_match["line_number"]
        }
    logger.info(f"No match found above threshold {threshold}")
    return None