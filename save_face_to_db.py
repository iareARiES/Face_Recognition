import psycopg2
import os
from dotenv import load_dotenv

# Load DB config
load_dotenv("face.env")

# Connect using no password
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
)

def save_face_to_db(name, embedding):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO face_embeddings (name, embedding) VALUES (%s, %s)",
            (name, embedding.tolist())
        )
        conn.commit()
        print(f"[✅ DB] Face for '{name}' saved to local PostgreSQL.")
