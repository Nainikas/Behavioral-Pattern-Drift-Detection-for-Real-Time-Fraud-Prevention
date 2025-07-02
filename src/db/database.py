# src/db/database.py

import os
import psycopg2

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        dbname=os.getenv("DB_NAME", "frauddb"),
        user=os.getenv("DB_USER", "frauduser"),
        password=os.getenv("DB_PASSWORD", "fraudpass"),
        port=5432
    )
