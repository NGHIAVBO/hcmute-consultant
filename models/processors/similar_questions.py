from flask import current_app
from sklearn.metrics.pairwise import cosine_similarity
from models.managers.json import tokenize_vietnamese

def recommend_similar_questions(query, top_n=5):
    try:
        vectorizer = current_app.config['vectorizer']
        tfidf_matrix = current_app.config['tfidf_matrix']
        query_tokenized = tokenize_vietnamese(query)
        query_tfidf = vectorizer.transform([query_tokenized])
        sim_scores = cosine_similarity(query_tfidf, tfidf_matrix)[0]
        sim_scores_with_indices = [(idx, score) for idx, score in enumerate(sim_scores) if score > 0.1]
        sim_scores_with_indices = sorted(sim_scores_with_indices, key=lambda x: x[1], reverse=True)
        top_results = sim_scores_with_indices[:top_n]
        question_indices = [i[0] for i in top_results]
        question_scores = [i[1] for i in top_results]
        return question_indices, question_scores
    except Exception as e:
        return [], []