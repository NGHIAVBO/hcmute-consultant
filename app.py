import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import pandas as pd
import google.generativeai as genai
from config import GOOGLE_API_KEY, LOCAL_URL, PRODUCTION_URL

from models.managers.json import prepare_data
from models.processors.similar_questions import recommend_similar_questions
from models.processors.llm_chain import generate_alternative_answers
from models.managers.pdf import process_directory_pdfs
from models.processors.text_splitter import get_text_chunks
from models.storages.vector_database import get_vector_database
from models.processors.query_processor import process_query

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('views.log', encoding='utf-8'),
        logging.StreamHandler()  # Optional: keep console output
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [LOCAL_URL, PRODUCTION_URL, "*"]}})

genai.configure(api_key=GOOGLE_API_KEY)

def initialize_app():
    result = True
    
    try:
        logger.info("Initializing app: preparing data")
        df, vectorizer, tfidf_matrix = prepare_data()
        app.config['df'] = df
        app.config['vectorizer'] = vectorizer
        app.config['tfidf_matrix'] = tfidf_matrix
        logger.info(f"Data prepared: {len(df)} records, vectorizer: {vectorizer is not None}, tfidf_matrix: {tfidf_matrix is not None}")
        if df.empty or vectorizer is None or tfidf_matrix is None:
            logger.warning("Data preparation returned empty or invalid data")
            result = False
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        app.config['df'] = pd.DataFrame(columns=['question', 'answer', 'source'])
        app.config['vectorizer'] = None
        app.config['tfidf_matrix'] = None
        result = False
    
    try:
        if not (os.path.exists("faiss_index") and os.path.exists("faiss_index/index.faiss")):
            logger.info("Processing PDFs")
            success = process_directory_pdfs(
                force_reprocess=False,
                get_text_chunks_fn=get_text_chunks,
                get_vector_database_fn=get_vector_database
            )
            if not success:
                logger.error("PDF processing failed")
                result = False
            else:
                logger.info("PDF processing completed successfully")
        else:
            logger.info("FAISS index already exists, skipping PDF processing")
    except Exception as e:
        logger.error(f"Error processing PDFs: {str(e)}")
        result = False
    
    logger.info(f"App initialization completed with result: {result}")
    return result

def ensure_recommend_data_loaded():
    """
    Ensure df, vectorizer, and tfidf_matrix are loaded in app.config.
    This is necessary for environments like Railway with multiple workers.
    """
    if (
        'df' not in app.config
        or app.config['df'] is None
        or app.config.get('vectorizer') is None
        or app.config.get('tfidf_matrix') is None
    ):
        try:
            logger.debug("Reloading recommendation data")
            df, vectorizer, tfidf_matrix = prepare_data()
            app.config['df'] = df
            app.config['vectorizer'] = vectorizer
            app.config['tfidf_matrix'] = tfidf_matrix
            logger.info(f"Reloaded data: {len(df)} records")
            if df.empty:
                logger.warning("Reloaded data is empty")
        except Exception as e:
            logger.error(f"Error reloading recommendation data: {str(e)}")
            app.config['df'] = pd.DataFrame(columns=['question', 'answer', 'source'])
            app.config['vectorizer'] = None
            app.config['tfidf_matrix'] = None

@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        logger.debug("Handling /recommend endpoint")
        ensure_recommend_data_loaded()
        query = request.args.get('text', '').strip()
        logger.info(f"Received query for /recommend: {query}")
        if not query:
            logger.warning("Empty query received for /recommend")
            return jsonify({
                'status': 'error',
                'message': 'Tham số truy vấn "text" là bắt buộc và không được rỗng'
            }), 400
            
        recommended_indices, similarity_scores = recommend_similar_questions(query, 5)
        logger.debug(f"Found {len(recommended_indices)} recommended indices")
        if not recommended_indices or not similarity_scores:
            logger.info(f"No recommendations found for query: {query}")
            return jsonify({
                'status': 'success',
                'message': f'Không tìm thấy gợi ý phù hợp cho truy vấn "{query}"',
                'data': []
            })
            
        df = app.config['df']
        recommendations = []
        
        for idx, score in zip(recommended_indices, similarity_scores):
            if idx < len(df) and score > 0.1:
                result = {
                    'question': df.iloc[idx]['question'],
                    'answer': df.iloc[idx]['answer'],
                    'similarity_score': float(score)
                }
                if 'source' in df.columns:
                    result['source'] = df.iloc[idx]['source']
                if 'question_id' in df.columns and not pd.isna(df.iloc[idx].get('question_id')):
                    result['question_id'] = int(df.iloc[idx]['question_id'])
                if 'answer_id' in df.columns and not pd.isna(df.iloc[idx].get('answer_id')):
                    result['answer_id'] = int(df.iloc[idx]['answer_id'])
                recommendations.append(result)
                logger.debug(f"Added recommendation: question={result['question'][:50]}..., score={score}")
                
        if not recommendations:
            logger.info(f"No recommendations above threshold for query: {query}")
            return jsonify({
                'status': 'success',
                'message': f'Không tìm thấy gợi ý phù hợp cho truy vấn "{query}"',
                'data': []
            })
            
        logger.info(f"Returning {len(recommendations)} recommendations for query: {query}")
        return jsonify({
            'status': 'success',
            'message': f'Đã gợi ý {len(recommendations)} mục cho truy vấn "{query}"',
            'data': recommendations
        })
        
    except Exception as e:
        logger.error(f"Error in /recommend endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Lỗi máy chủ nội bộ: {str(e)}'
        }), 500

@app.route('/recommend-answers', methods=['GET'])
def get_recommend_answers():
    try:
        logger.debug("Handling /recommend-answers endpoint")
        query = request.args.get('text', '').strip()
        logger.info(f"Received query for /recommend-answers: {query}")
        if not query:
            logger.warning("Empty query received for /recommend-answers")
            return jsonify({
                'status': 'error',
                'message': 'Tham số truy vấn "text" là bắt buộc và không được rỗng'
            }), 400
            
        answer = process_query(query)
        logger.debug(f"Generated primary answer: {answer[:100]}...")
        
        alternative_answers = generate_alternative_answers(query, answer)
        logger.info(f"Generated {len(alternative_answers)} alternative answers")
        if len(alternative_answers) > 5:
            alternative_answers = alternative_answers[:5]
            logger.debug("Limited to 5 alternative answers")
            
        result_answers = [{"answer": a} for a in alternative_answers]
        
        logger.info("Returning alternative answers")
        return jsonify({
            'status': 'success',
            'message': 'Đã tạo 5 câu trả lời thay thế',
            'data': result_answers
        })
        
    except Exception as e:
        logger.error(f"Error in /recommend-answers endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Lỗi máy chủ nội bộ: {str(e)}'
        }), 500

@app.route('/chat', methods=['GET'])
def chat():
    start_time = time.time()
    question = request.args.get("text", "").strip()
    logger.info(f"Received query for /chat: {question}")
    
    if not question:
        logger.warning("Empty query received for /chat")
        return jsonify({
            "status": "fail",
            "message": "Vui lòng nhập câu hỏi",
            "data": {"time": round(time.time() - start_time, 2)}
        }), 400

    try:
        answer = process_query(question)
        process_time = round(time.time() - start_time, 2)
        logger.debug(f"Processed query in {process_time:.2f}s: {answer[:100]}...")
        
        if "*(Kết quả từ cache" in answer:
            parts = answer.split("\n\n*(")
            main_answer = parts[0]
            logger.info("Returning cached answer")
            return jsonify({
                "status": "success",
                "message": "Lấy câu trả lời từ cache thành công",
                "data": {
                    "question": question,
                    "answer": main_answer,
                    "time": process_time
                }
            })

        logger.info("Returning generated answer")
        return jsonify({
            "status": "success",
            "message": "Tìm câu trả lời thành công",
            "data": {
                "question": question,
                "answer": answer,
                "time": process_time
            }
        })

    except Exception as e:
        logger.error(f"Error in /chat endpoint: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Lỗi khi xử lý câu hỏi: {str(e)}",
            "data": {"time": round(time.time() - start_time, 2)}
        }), 500

if __name__ == "__main__":
    logger.info("Starting Flask application")
    initialize_app()
    app.run(host="0.0.0.0", port=5000, debug=False)