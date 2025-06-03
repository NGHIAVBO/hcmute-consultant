import logging
import os
from models.processors.llm_chain import get_gemini_response
from models.processors.small_talk import is_small_talk
from models.storages.vector_database import load_vector_database
from models.managers.cache import get_cache, set_cache
from models.managers.json import find_best_match
from config import PDF_FILE, JSON_FILE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

vector_database = None

def load_vector_db_once():
    global vector_database
    if vector_database is None:
        try:
            logger.info("Loading vector database from faiss_index.")
            vector_database = load_vector_database()[0]
            if vector_database is None:
                logger.error("Vector database is None after loading.")
            else:
                logger.info("Vector database loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load vector database: {str(e)}")
            vector_database = None
    return vector_database

def print_reference_sources(response, json_match=None):
    if response and "source_documents" in response:
        source_grouping = {}
        for doc in response["source_documents"]:
            source_key = f"{doc.metadata.get('source', 'Unknown')}"
            if source_key not in source_grouping:
                source_grouping[source_key] = []
            source_grouping[source_key].append(doc)
        
        logger.info("\n" + "="*50)
        logger.info("Nguồn tham khảo:")
        for source, docs in source_grouping.items():
            logger.info(f"- {source}: {len(docs)} đoạn văn")
        logger.info("="*50 + "\n")
    
    if json_match:
        logger.info("\n" + "="*50)
        logger.info("Nguồn tham khảo JSON:")
        logger.info(f"- File: {json_match['source']}")
        logger.info(f"- Dòng số: {json_match['line_number']*4}")
        logger.info("="*50 + "\n")

def process_query(prompt):
    logger.info(f"Processing query: {prompt}")
    
    # Check cache first
    cached_result, cache_hit, time_saved = get_cache(prompt)
    if cache_hit:
        logger.info(f"Cache hit for query: {prompt}, returning cached result.")
        return f"{cached_result}\n\n*(Kết quả từ cache, tiết kiệm {time_saved:.2f}s)*"
    
    # Handle small talk
    small_talk_response = is_small_talk(prompt)
    if small_talk_response:
        logger.info(f"Query identified as small talk: {prompt}")
        set_cache(prompt, small_talk_response, 0)
        return small_talk_response
    
    try:
        # Load vector database
        vector_database = load_vector_db_once()
        result = None
        
        # Try PDF processing first
        if vector_database and os.path.exists(PDF_FILE):
            logger.info(f"Processing query with PDF: {PDF_FILE}")
            context_prompt = f"Dựa trên thông tin trong {PDF_FILE}, {prompt}"
            logger.info(f"Constructed context prompt: {context_prompt}")

            # Get response from Gemini
            response = get_gemini_response(vector_database, context_prompt, filter_pdf=PDF_FILE)
            if response and "output_text" in response:
                answer = response["output_text"]
                logger.info(f"Received response from Gemini: {answer[:100]}...")
                
                if answer and not any(phrase in answer.lower() for phrase in ["không tìm thấy thông tin", "không có thông tin"]):
                    result = answer
                    set_cache(prompt, result, 0)
                    print_reference_sources(response, None)
            else:
                logger.error("No response or output_text from get_gemini_response.")
        
        else:
            if not vector_database:
                logger.error("Vector database not loaded.")
            if not os.path.exists(PDF_FILE):
                logger.error(f"PDF file not found at: {PDF_FILE}")

        # Fall back to JSON if PDF processing failed or returned no information
        if result is None and os.path.exists(JSON_FILE):
            logger.info(f"Falling back to JSON match in {JSON_FILE}")
            json_match = find_best_match(prompt)
            if json_match:
                logger.info(f"Found JSON match for query: {prompt}")
                result = json_match["answer"]
                set_cache(prompt, result, 0)
                print_reference_sources(None, json_match)
        
        # Return result or error
        if result:
            return result
        return "Không tìm thấy thông tin bạn đưa ra. Vui lòng đặt câu hỏi khác."
        
    except Exception as e:
        logger.error(f"Error processing query '{prompt}': {str(e)}")
        # Try JSON fallback in case of exception
        if os.path.exists(JSON_FILE):
            logger.info(f"Falling back to JSON match in {JSON_FILE} after error")
            json_match = find_best_match(prompt)
            if json_match:
                logger.info(f"Found JSON match for query: {prompt}")
                set_cache(prompt, json_match["answer"], 0)
                print_reference_sources(None, json_match)
                return json_match["answer"]
        return "Xin lỗi, tôi không thể xử lý yêu cầu của bạn. Vui lòng thử lại sau."