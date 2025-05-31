import pyodbc
import mysql.connector
import re
import emoji
import json
import os
from config import JSON_FILE, DATA_DIR, SQL_FILE

sqlserver_config = {
    "user": "sa",
    "password": "12345",
    "server": "localhost",
    "database": "TuVanSinhVien",
    "instance_name": "SQLEXPRESS"
}

mysql_config = {
    "user": "root",
    "password": "12345",
    "host": "localhost",
    "database": "kltn"
}

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = emoji.replace_emoji(text, '')
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*([,.!?;:])\s*', r'\1 ', text)
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    text = text.strip()
    text = re.sub(r'\s{2,}', ' ', text)
    return text

def html_to_text(html_content, options=None):
    if html_content is None:
        return ""
    if not isinstance(html_content, str):
        html_content = str(html_content)
    html_content = re.sub(r'<!--.*?-->', '', html_content, flags=re.DOTALL)
    html_content = re.sub(r'<script.*?>.*?</script>', '', html_content, flags=re.DOTALL)
    html_content = re.sub(r'<style.*?>.*?</style>', '', html_content, flags=re.DOTALL)
    html_content = re.sub(r'<br[^>]*>', '\n', html_content)
    html_content = re.sub(r'</p>\s*<p[^>]*>', '\n\n', html_content)
    html_content = re.sub(r'<div[^>]*>', '\n', html_content)
    html_content = re.sub(r'</div>', '\n', html_content)
    text = re.sub(r'<[^>]+>', '', html_content)
    text = re.sub(r'& nbsp ;', ' ', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'\n', ' ', text)
    return text

def connect_to_sqlserver():
    conn_str = (
        f"DRIVER={{SQL Server}};"
        f"SERVER={sqlserver_config['server']}\\{sqlserver_config['instance_name']};"
        f"DATABASE={sqlserver_config['database']};"
        f"UID={sqlserver_config['user']};"
        f"PWD={sqlserver_config['password']};"
    )
    return pyodbc.connect(conn_str)

def connect_to_mysql():
    return mysql.connector.connect(**mysql_config)

def fetch_data_from_sqlserver():
    conn = connect_to_sqlserver()
    cursor = conn.cursor()
    
    question_query = "SELECT [ID], [Title], [Contents], [Date], [Author_ID], [Field_ID], [Dep_ID], [Status], [Total_view], [Filename] FROM [TuVanSinhVien].[dbo].[Questions]"
    cursor.execute(question_query)
    questions = cursor.fetchall()
    
    answer_query = "SELECT [ID], [Author_ID], [Contents], [Date], [Status_Public], [Status], [Status_Approval], [Value], [Ques_ID], [Filename] FROM [TuVanSinhVien].[dbo].[Answers]"
    cursor.execute(answer_query)
    answers = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return questions, answers

def truncate_content(content, max_length=3000):
    if content and len(content) > max_length:
        return content[:max_length]
    return content

def insert_into_mysql(questions, answers):
    conn = connect_to_mysql()
    cursor = conn.cursor()
    
    alter_tables_query = """
    ALTER TABLE question MODIFY COLUMN content LONGTEXT;
    ALTER TABLE answer MODIFY COLUMN content LONGTEXT;
    """
    
    try:
        for statement in alter_tables_query.split(';'):
            if statement.strip():
                cursor.execute(statement)
        conn.commit()
        print("Table columns altered to LONGTEXT successfully")
    except Exception as e:
        print(f"Error altering tables (continuing anyway): {str(e)}")
    
    cursor.execute("DELETE FROM answer")
    cursor.execute("DELETE FROM question")
    cursor.execute("ALTER TABLE question AUTO_INCREMENT = 1")
    cursor.execute("ALTER TABLE answer AUTO_INCREMENT = 1")
    
    unique_content_to_question = {}
    content_to_id = {}
    duplicate_count = 0
    
    for q in questions:
        old_id = q[0]
        plain_content = html_to_text(q[2])
        content = truncate_content(clean_text(plain_content))
        
        if not content.strip():
            continue
        
        if content not in unique_content_to_question:
            unique_content_to_question[content] = {
                "old_id": old_id,
                "title": truncate_content(clean_text(q[1]), 255) if q[1] else "Untitled",
                "content": content,
                "views": q[8] if q[8] else 0
            }
            content_to_id[content] = old_id
        else:
            duplicate_count += 1
    
    print(f"Found {len(unique_content_to_question)} unique questions and {duplicate_count} duplicates")
    
    question_to_answer = {}
    duplicate_answer_count = 0
    
    for a in answers:
        plain_content = html_to_text(a[2])
        content = truncate_content(clean_text(plain_content))
        
        if not content.strip():
            continue
            
        old_question_id = a[8]
        
        question_content = None
        for q_content, q_id in content_to_id.items():
            if q_id == old_question_id:
                question_content = q_content
                break
                
        if not question_content:
            continue
            
        if old_question_id not in question_to_answer:
            question_to_answer[old_question_id] = {
                "content": content,
                "original_id": a[0]
            }
        else:
            duplicate_answer_count += 1
    
    print(f"Found {len(question_to_answer)} unique question-answer pairs and {duplicate_answer_count} duplicate answers")
    
    valid_questions = []
    
    for content, q_data in unique_content_to_question.items():
        old_id = q_data["old_id"]
        
        if old_id in question_to_answer:
            valid_questions.append(q_data)
    
    question_insert_query = """
    INSERT INTO question (
        title, content, created_at, file_name, status_approval, 
        status_delete, status_public, views, department_id, 
        field_id, parent_question_id, role_ask_id, user_id
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    default_date = "2025-05-01"
    
    old_id_to_new_id = {}
    new_id = 1
    
    valid_questions.sort(key=lambda q: q["old_id"])
    
    for q_data in valid_questions:
        old_id = q_data["old_id"]
        
        question_data = (
            q_data["title"],
            q_data["content"],
            default_date,
            None,
            1,
            0,
            1,
            q_data["views"],
            20,
            160,
            None,
            1,
            61
        )
        
        try:
            cursor.execute(question_insert_query, question_data)
            old_id_to_new_id[old_id] = new_id
            new_id += 1
        except Exception as e:
            print(f"Error inserting question ID {old_id}: {str(e)}")
            continue
    
    print(f"Inserted {len(old_id_to_new_id)} unique questions")
    
    answer_insert_query = """
    INSERT INTO answer (
        content, created_at, file, status_answer, status_approval, 
        title, question_id, role_consultant_id, user_id
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    answer_count = 0
    
    for old_id, new_id in old_id_to_new_id.items():
        if old_id in question_to_answer:
            answer_data = (
                question_to_answer[old_id]["content"],
                default_date,
                None,
                1,
                0,
                "Phản hồi từ thầy Nguyễn Hữu Trung",
                new_id,
                2,
                63
            )
            
            try:
                cursor.execute(answer_insert_query, answer_data)
                answer_count += 1
            except Exception as e:
                print(f"Error inserting answer for question {old_id} (new ID {new_id}): {str(e)}")
                continue
    
    print(f"Inserted {answer_count} answers")
    conn.commit()
    cursor.close()
    conn.close()

def export_mysql_data_to_json():
    try:
        print("Starting export from MySQL to output.json...")
        
        conn = connect_to_mysql()
        cursor = conn.cursor(dictionary=True)
        
        query = """
        SELECT 
            q.id as question_id, 
            q.content as question, 
            a.content as answer
        FROM 
            question q
        JOIN 
            answer a ON q.id = a.question_id
        ORDER BY 
            q.id ASC
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        output_data = []
        for row in results:
            output_data.append({
                "question": row["question"],
                "answer": row["answer"]
            })
        
        with open(os.path.join(DATA_DIR, JSON_FILE), "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully exported {len(output_data)} QA pairs to {JSON_FILE}")
        
        cursor.close()
        conn.close()
        
        return len(output_data)
    
    except Exception as e:
        print(f"Error exporting data to JSON: {str(e)}")
        return 0

def export_data_to_sql(output_file=os.path.join(DATA_DIR, SQL_FILE)):
    try:
        print(f"Starting export to SQL script {output_file}...")
        
        conn = connect_to_mysql()
        cursor = conn.cursor(dictionary=True)
        
        query = """
        SELECT 
            q.id, q.title, q.content, q.created_at, q.file_name, 
            q.status_approval, q.status_delete, q.status_public, 
            q.views, q.department_id, q.field_id, q.parent_question_id, 
            q.role_ask_id, q.user_id,
            a.id as answer_id, a.content as answer_content, a.created_at as answer_created_at,
            a.file, a.status_answer, a.status_approval as answer_status_approval,
            a.title as answer_title, a.role_consultant_id, a.user_id as answer_user_id
        FROM 
            question q
        JOIN 
            answer a ON q.id = a.question_id
        ORDER BY 
            q.id ASC
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("-- Insert questions\n")
            for row in results:
                title = row["title"].replace("'", "''") if row["title"] else "Untitled"
                content = row["content"].replace("'", "''") if row["content"] else ""
                
                if row["file_name"]:
                    file_name = "'" + row["file_name"].replace("'", "''") + "'"
                else:
                    file_name = "NULL"
                
                if row["created_at"]:
                    created_at = "'" + row["created_at"].strftime('%Y-%m-%d') + "'"
                else:
                    created_at = "'2025-05-01'"
                
                status_approval = row["status_approval"] if row["status_approval"] is not None else 1
                status_delete = row["status_delete"] if row["status_delete"] is not None else 0
                status_public = row["status_public"] if row["status_public"] is not None else 1
                
                views = row["views"] if row["views"] is not None else 0
                department_id = row["department_id"] if row["department_id"] is not None else "NULL"
                field_id = row["field_id"] if row["field_id"] is not None else "NULL"
                parent_question_id = row["parent_question_id"] if row["parent_question_id"] is not None else "NULL"
                role_ask_id = row["role_ask_id"] if row["role_ask_id"] is not None else "NULL"
                user_id = row["user_id"] if row["user_id"] is not None else "NULL"
                
                f.write(f"INSERT INTO question (id, content, created_at, file_name, status_approval, status_delete, status_public, title, views, department_id, field_id, parent_question_id, role_ask_id, user_id) VALUES ")
                f.write(f"({row['id']}, '{content}', {created_at}, {file_name}, b'{status_approval}', b'{status_delete}', b'{status_public}', '{title}', {views}, {department_id}, {field_id}, {parent_question_id}, {role_ask_id}, {user_id});\n")
            
            f.write("\n-- Insert answers\n")
            for row in results:
                answer_title = row["answer_title"].replace("'", "''") if row["answer_title"] else ""
                answer_content = row["answer_content"].replace("'", "''") if row["answer_content"] else ""
                
                if row["file"]:
                    answer_file = "'" + row["file"].replace("'", "''") + "'"
                else:
                    answer_file = "NULL"
                
                if row["answer_created_at"]:
                    answer_created_at = "'" + row["answer_created_at"].strftime('%Y-%m-%d') + "'"
                else:
                    answer_created_at = "'2025-05-01'"
                
                status_answer = row["status_answer"] if row["status_answer"] is not None else 1
                answer_status_approval = row["answer_status_approval"] if row["answer_status_approval"] is not None else 0
                
                role_consultant_id = row["role_consultant_id"] if row["role_consultant_id"] is not None else "NULL"
                answer_user_id = row["answer_user_id"] if row["answer_user_id"] is not None else "NULL"
                
                f.write(f"INSERT INTO answer (id, content, created_at, file, status_answer, status_approval, title, question_id, role_consultant_id, user_id) VALUES ")
                f.write(f"({row['answer_id']}, '{answer_content}', {answer_created_at}, {answer_file}, b'{status_answer}', b'{answer_status_approval}', '{answer_title}', {row['id']}, {role_consultant_id}, {answer_user_id});\n")
        
        cursor.close()
        conn.close()
        
        print(f"Successfully exported {len(results)} QA pairs to {output_file}")
        return len(results)
    
    except Exception as e:
        print(f"Error exporting data to SQL: {str(e)}")
        return 0
    
def main():
    try:
        print("Starting data migration from SQL Server to MySQL...")
        questions, answers = fetch_data_from_sqlserver()
        print(f"Retrieved {len(questions)} questions and {len(answers)} answers from SQL Server")
        
        insert_into_mysql(questions, answers)
        print("Data migration completed successfully!")
     
    except Exception as e:
        print(f"Error during migration: {str(e)}")

if __name__ == "__main__":
    main()