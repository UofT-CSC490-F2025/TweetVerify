from flask import Flask, request, jsonify, session, render_template, redirect, url_for
import psycopg2
from psycopg2.extras import RealDictCursor
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import glob
import re
import uuid
from datetime import datetime
from psycopg2 import pool

app = Flask(__name__, template_folder="src/web/templates")
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = "model_save"
ALLOWED_EXTENSIONS = {"pt", "pth", "pkl", "model"}
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db_pool = pool.SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    host=os.getenv("DB_HOST"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS"),
)


def get_db_connection():
    return db_pool.getconn()


def put_db_connection(conn):
    db_pool.putconn(conn)


from contextlib import contextmanager


@contextmanager
def db_cursor(cursor_factory=None):
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=cursor_factory)
        yield cur, conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        put_db_connection(conn)


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400

    hashed_password = generate_password_hash(password)

    try:
        with db_cursor() as (cur, conn):
            cur.execute("SELECT id FROM users WHERE username = %s", (username,))
            if cur.fetchone():
                return jsonify({"error": "Username already exists"}), 400
            cur.execute(
                "INSERT INTO users (username, password) VALUES (%s, %s)",
                (username, hashed_password),
            )
        return jsonify({"message": "User registered successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400

    try:
        with db_cursor(cursor_factory=RealDictCursor) as (cur, conn):
            cur.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cur.fetchone()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return jsonify(
                {"message": "Login successful", "redirect": url_for("dashboard")}
            )
        else:
            return jsonify({"error": "Invalid username or password"}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def init_db():
    try:
        with db_cursor() as (cur, conn):
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL
                );
            """
            )
        print("✅ users table initialized.")
    except Exception as e:
        print(f"❌ Failed to initialize DB: {e}")


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5001)
