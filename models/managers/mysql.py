import pandas as pd
import mysql.connector
from mysql.connector import Error, pooling
from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
from functools import lru_cache
import contextlib

try:
    connection_pool = pooling.MySQLConnectionPool(
        pool_name="mypool",
        pool_size=5,
        host=MYSQL_HOST,
        port=int(MYSQL_PORT),
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )
except:
    connection_pool = None

@contextlib.contextmanager
def get_connection():
    conn = None
    try:
        if connection_pool:
            conn = connection_pool.get_connection()
        else:
            conn = mysql.connector.connect(
                host=MYSQL_HOST,
                port=int(MYSQL_PORT),
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DATABASE
            )
        yield conn
    except Error:
        yield None
    finally:
        if conn and (hasattr(conn, 'is_connected') and conn.is_connected()):
            conn.close()

def get_query_questions():
    return """
    SELECT q.id, q.content, q.created_at, q.title, q.status_approval, q.role_ask_id, q.user_id
    FROM question q
    WHERE q.status_delete = 0
    """

def get_query_answers():
    return """
    SELECT a.id, a.content, a.created_at, a.question_id, a.status_answer, a.status_approval,
           a.title, a.role_consultant_id, a.user_id
    FROM answer a
    """

def fetch_data_from_mysql():
    with get_connection() as connection:
        if not connection:
            return pd.DataFrame()

        try:
            questions_df = pd.read_sql(get_query_questions(), connection)
            answers_df = pd.read_sql(get_query_answers(), connection)
            
            if questions_df.empty or answers_df.empty:
                return pd.DataFrame()
                
            merged_df = pd.merge(
                questions_df,
                answers_df,
                left_on='id',
                right_on='question_id',
                how='inner',
                suffixes=('_question', '_answer')
            )
            
            if merged_df.empty:
                merged_df = pd.merge(
                    questions_df,
                    answers_df,
                    left_on='id',
                    right_on='question_id',
                    how='left',
                    suffixes=('_question', '_answer')
                )
                merged_df = merged_df[merged_df['question_id'].notna()]
            
            if merged_df.empty:
                return pd.DataFrame()
                
            return pd.DataFrame({
                'question': merged_df['content_question'],
                'answer': merged_df['content_answer'],
                'question_id': merged_df['id_question'],
                'answer_id': merged_df['id_answer'],
                'source': 'mysql'
            })
        
        except Error:
            return pd.DataFrame()