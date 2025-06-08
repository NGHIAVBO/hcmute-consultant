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

import sys
import importlib.util
import subprocess

from config import JSON_FILE, CONVERT_FILE

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [LOCAL_URL, PRODUCTION_URL, "*"]}})

genai.configure(api_key=GOOGLE_API_KEY)

def check_and_create_output_json():
    if not os.path.exists(JSON_FILE):
        convert_file_path = os.path.join(os.path.dirname(__file__), CONVERT_FILE)
        if os.path.exists(convert_file_path):
            try:
                spec = importlib.util.spec_from_file_location("convert", convert_file_path)
                convert_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(convert_module)
                
                if hasattr(convert_module, "export_mysql_data_to_json"):
                    count = convert_module.export_mysql_data_to_json()
                    return count > 0
                else:
                    result = subprocess.run(["python", convert_file_path], capture_output=True, text=True)
                    if result.returncode == 0:
                        if os.path.exists(JSON_FILE):
                            return True
                    else:
                        print(f"Lỗi khi chạy convert.py: {result.stderr}")
            except Exception as e:
                print(f"Lỗi khi import hoặc thực thi convert.py: {str(e)}")
        
        return False
    return True

def initialize_app():
    result = True
    
    json_created = check_and_create_output_json()
    if not json_created:
        print("Không thể tạo output.json từ MySQL.")
    
    try:
        df, vectorizer, tfidf_matrix = prepare_data()
        app.config['df'] = df
        app.config['vectorizer'] = vectorizer
        app.config['tfidf_matrix'] = tfidf_matrix
    except Exception as e:
        print(f"Lỗi khi chuẩn bị dữ liệu: {str(e)}")
        app.config['df'] = pd.DataFrame(columns=['question', 'answer'])
        app.config['vectorizer'] = None
        app.config['tfidf_matrix'] = None
        result = False
    
    try:
        if not (os.path.exists("faiss_index") and os.path.exists("faiss_index/index.faiss")):
            success = process_directory_pdfs(
                force_reprocess=False,
                get_text_chunks_fn=get_text_chunks,
                get_vector_database_fn=get_vector_database
            )
            if not success:
                result = False
    except Exception as e:
        print(f"Lỗi khi xử lý PDF: {str(e)}")
        result = False
    
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
            df, vectorizer, tfidf_matrix = prepare_data()
            app.config['df'] = df
            app.config['vectorizer'] = vectorizer
            app.config['tfidf_matrix'] = tfidf_matrix
        except Exception as e:
            app.config['df'] = pd.DataFrame(columns=['question', 'answer'])
            app.config['vectorizer'] = None
            app.config['tfidf_matrix'] = None

@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        ensure_recommend_data_loaded()
        query = request.args.get('text', '').strip()
        if not query:
            return jsonify({
                'status': 'error',
                'message': 'Tham số truy vấn "text" là bắt buộc và không được rỗng'
            }), 400
            
        recommended_indices, similarity_scores = recommend_similar_questions(query, 5)
        if not recommended_indices or not similarity_scores:
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
                
        if not recommendations:
            return jsonify({
                'status': 'success',
                'message': f'Không tìm thấy gợi ý phù hợp cho truy vấn "{query}"',
                'data': []
            })
            
        return jsonify({
            'status': 'success',
            'message': f'Đã gợi ý {len(recommendations)} mục cho truy vấn "{query}"',
            'data': recommendations
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Lỗi máy chủ nội bộ: {str(e)}'
        }), 500

@app.route('/recommend-answers', methods=['GET'])
def get_recommend_answers():
    try:
        query = request.args.get('text', '').strip()
        if not query:
            return jsonify({
                'status': 'error',
                'message': 'Tham số truy vấn "text" là bắt buộc và không được rỗng'
            }), 400
            
        answer = process_query(query)
        
        alternative_answers = generate_alternative_answers(query, answer)
        if len(alternative_answers) > 5:
            alternative_answers = alternative_answers[:5]
            
        result_answers = [{"answer": a} for a in alternative_answers]
        
        return jsonify({
            'status': 'success',
            'message': 'Đã tạo 5 câu trả lời thay thế',
            'data': result_answers
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Lỗi máy chủ nội bộ: {str(e)}'
        }), 500

@app.route('/chat', methods=['GET'])
def chat():
    start_time = time.time()
    question = request.args.get("text", "").strip()
    
    if not question:
        return jsonify({
            "status": "fail",
            "message": "Vui lòng nhập câu hỏi",
            "data": {"time": round(time.time() - start_time, 2)}
        }), 400

    try:
        answer = process_query(question)
        process_time = round(time.time() - start_time, 2)
        
        if "*(Kết quả từ cache" in answer:
            parts = answer.split("\n\n*(")
            main_answer = parts[0]
            return jsonify({
                "status": "success",
                "message": "Lấy câu trả lời từ cache thành công",
                "data": {
                    "question": question,
                    "answer": main_answer,
                    "time": process_time
                }
            })

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
        return jsonify({
            "status": "error",
            "message": f"Lỗi khi xử lý câu hỏi: {str(e)}",
            "data": {"time": round(time.time() - start_time, 2)}
        }), 500

if __name__ == "__main__":
    initialize_app()
    app.run(host="0.0.0.0", port=8080, debug=False)