import logging
from models.processors.llm_chain import get_gemini_response
from models.processors.small_talk import is_small_talk
from models.storages.vector_database import load_vector_database
from models.managers.cache import get_cache, set_cache
from models.managers.json import find_best_match
from config import PDF_FILE

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

vector_database = None

def load_vector_db_once():
    global vector_database
    if vector_database is None:
        try:
            logger.debug("Loading vector database")
            vector_database = load_vector_database()[0]
            logger.info("Vector database loaded successfully")
        except Exception as e:
            logger.error(f"Error loading vector database: {str(e)}")
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
    logger.debug(f"Processing query: {prompt}")
    cached_result, cache_hit, time_saved = get_cache(prompt)
    if cache_hit:
        logger.info(f"Cache hit for query: {prompt}, time saved: {time_saved:.2f}s")
        return f"{cached_result}\n\n*(Kết quả từ cache, tiết kiệm {time_saved:.2f}s)*"
    
    small_talk_response = is_small_talk(prompt)
    if small_talk_response:
        logger.info(f"Query identified as small talk: {prompt}")
        return small_talk_response
    
    try:
        vector_database = load_vector_db_once()
        if not vector_database:
            logger.error("Vector database not loaded")
            return "Xin lỗi, tôi không thể xử lý yêu cầu của bạn. Vui lòng thử lại sau."
            
        context_prompt = f"Dựa trên thông tin trong {PDF_FILE}, {prompt}"
        logger.debug(f"Context prompt: {context_prompt}")

        response = get_gemini_response(vector_database, context_prompt, filter_pdf=PDF_FILE)
        if not response:
            logger.error("No response from get_gemini_response")
            return "Xin lỗi, tôi không thể xử lý yêu cầu của bạn. Vui lòng thử lại sau."
        
        answer = response["output_text"]
        if not answer:
            logger.error("Empty response from get_gemini_response")
            return "Xin lỗi, không nhận được câu trả lời. Vui lòng thử lại sau."
            
        if any(phrase in answer.lower() for phrase in ["không tìm thấy thông tin", "không có thông tin"]):
            result = "Không tìm thấy thông tin bạn đưa ra. Vui lòng đặt câu hỏi khác."
            logger.info(f"No relevant information found for query: {prompt}")
        else:
            result = answer
            logger.info(f"Generated answer: {answer[:100]}...")
        
        set_cache(prompt, result, 0)
        logger.debug(f"Cached result for query: {prompt}")
        
        json_match = find_best_match(prompt)
        print_reference_sources(response, json_match)
        
        if json_match:
            logger.info(f"JSON match found: {json_match['answer'][:50]}...")
            result = json_match["answer"]
            set_cache(prompt, result, 0)
            logger.debug(f"Cached JSON match result for query: {prompt}")
            return result
        
        return result
        
    except Exception as e:
        logger.error(f"Error in process_query: {str(e)}")
        return "Xin lỗi, tôi không thể xử lý yêu cầu của bạn. Vui lòng thử lại sau."