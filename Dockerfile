FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY data/ data/
COPY vietnamese-stopwords.txt vietnamese-stopwords.txt
COPY output.json output.json
COPY output.sql output.sql
COPY tfidf_matrix.pkl tfidf_matrix.pkl
COPY tfidf_vectorizer.pkl tfidf_vectorizer.pkl
COPY SoTaySinhVien2024.pdf SoTaySinhVien2024.pdf
COPY convert.py convert.py

ENV PORT=5000

EXPOSE 5000

CMD gunicorn --bind 0.0.0.0:$PORT app:app