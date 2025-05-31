import hashlib
import time

cache = []

def get_cache(query):
    key = hashlib.md5(query.lower().strip().encode()).hexdigest()
    for entry in cache:
        if entry["key"] == key:
            return entry["result"], True, entry.get("processing_time", 0)
    return None, False, 0

def set_cache(query, result, processing_time):
    key = hashlib.md5(query.lower().strip().encode()).hexdigest()
    cache.insert(0, {
        "key": key,
        "query": query,
        "result": result,
        "timestamp": time.time(),
        "processing_time": processing_time
    })