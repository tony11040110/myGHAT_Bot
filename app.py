import os
import json
import sqlite3
import threading
import webbrowser
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from dotenv import load_dotenv
from openai import OpenAI
from werkzeug.utils import secure_filename
from pypdf import PdfReader
from docx import Document

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chat.db"
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"txt", "md", "py", "json", "csv", "pdf", "docx"}
PORT = 5000
DEFAULT_MODEL = "openai/gpt-oss-20b"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("找不到 GROQ_API_KEY，請檢查 .env")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)


def log_event(level, message):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] [{level}] {message}")


def open_browser():
    webbrowser.open(f"http://127.0.0.1:{PORT}")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL DEFAULT 'New Chat',
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (chat_id) REFERENCES chats (id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            original_name TEXT NOT NULL,
            saved_name TEXT NOT NULL,
            content_type TEXT,
            extracted_text TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (chat_id) REFERENCES chats (id)
        )
    """)

    conn.commit()
    conn.close()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_file(file_path: Path) -> str:
    suffix = file_path.suffix.lower()

    try:
        if suffix in {".txt", ".md", ".py", ".json", ".csv"}:
            return file_path.read_text(encoding="utf-8", errors="ignore")[:20000]

        if suffix == ".pdf":
            reader = PdfReader(str(file_path))
            parts = []
            for page in reader.pages[:20]:
                text = page.extract_text() or ""
                if text.strip():
                    parts.append(text)
            joined = "\n\n".join(parts).strip()
            if not joined:
                return "[這個 PDF 可能是掃描影像型 PDF，抽不出文字內容]"
            return joined[:20000]

        if suffix == ".docx":
            doc = Document(str(file_path))
            parts = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    parts.append(text)
            joined = "\n".join(parts).strip()
            if not joined:
                return "[DOCX 內容為空或沒有可讀文字]"
            return joined[:20000]

    except Exception as e:
        return f"[讀取檔案失敗: {e}]"

    return "[目前不支援此檔案格式]"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/chats", methods=["GET"])
def list_chats():
    conn = get_db()
    chats = conn.execute("""
        SELECT id, title, created_at
        FROM chats
        ORDER BY id DESC
    """).fetchall()
    conn.close()

    return jsonify([
        {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"]
        }
        for row in chats
    ])


@app.route("/api/chats", methods=["POST"])
def create_chat():
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO chats (title, created_at) VALUES (?, ?)",
        ("New Chat", now)
    )
    chat_id = cur.lastrowid
    conn.commit()
    conn.close()

    log_event("INFO", f"建立新聊天 chat_id={chat_id}")
    return jsonify({
        "id": chat_id,
        "title": "New Chat",
        "created_at": now
    })


@app.route("/api/chats/<int:chat_id>/messages", methods=["GET"])
def get_messages(chat_id):
    conn = get_db()
    rows = conn.execute("""
        SELECT id, role, content, created_at
        FROM messages
        WHERE chat_id = ?
        ORDER BY id ASC
    """, (chat_id,)).fetchall()
    conn.close()

    return jsonify([
        {
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"]
        }
        for row in rows
    ])


@app.route("/api/chats/<int:chat_id>/files", methods=["GET"])
def list_files(chat_id):
    conn = get_db()
    rows = conn.execute("""
        SELECT id, original_name, created_at
        FROM files
        WHERE chat_id = ?
        ORDER BY id DESC
    """, (chat_id,)).fetchall()
    conn.close()

    return jsonify([
        {
            "id": row["id"],
            "original_name": row["original_name"],
            "created_at": row["created_at"]
        }
        for row in rows
    ])


@app.route("/api/chats/<int:chat_id>/upload", methods=["POST"])
def upload_file(chat_id):
    if "file" not in request.files:
        return jsonify({"error": "沒有收到檔案"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "檔名是空的"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "目前只支援 txt / md / py / json / csv / pdf / docx"}), 400

    original_name = file.filename
    safe_name = secure_filename(original_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_name = f"{timestamp}_{safe_name}"
    save_path = UPLOAD_FOLDER / saved_name
    file.save(save_path)

    extracted_text = extract_text_from_file(save_path)
    now = datetime.now().isoformat(timespec="seconds")

    conn = get_db()

    chat_row = conn.execute("SELECT id FROM chats WHERE id = ?", (chat_id,)).fetchone()
    if not chat_row:
        conn.close()
        return jsonify({"error": "chat not found"}), 404

    conn.execute("""
        INSERT INTO files (chat_id, original_name, saved_name, content_type, extracted_text, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        chat_id,
        original_name,
        saved_name,
        file.content_type,
        extracted_text,
        now
    ))

    snippet = extracted_text[:3000]
    file_note = (
        f"[使用者上傳檔案: {original_name}]\n\n"
        f"以下是可供對話參考的檔案內容摘要：\n{snippet}"
    )

    conn.execute("""
        INSERT INTO messages (chat_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
    """, (chat_id, "assistant", file_note, now))

    conn.commit()
    conn.close()

    log_event("FILE", f"chat_id={chat_id} 上傳檔案: {original_name}")
    return jsonify({
        "success": True,
        "filename": original_name,
        "preview": snippet
    })


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}

    chat_id = data.get("chat_id")
    user_message = (data.get("message") or "").strip()
    system_prompt = (data.get("system_prompt") or "You are a helpful assistant.").strip()
    model = (data.get("model") or DEFAULT_MODEL).strip()
    temperature = data.get("temperature", 0.7)

    if not chat_id:
        return jsonify({"error": "chat_id is required"}), 400

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    try:
        temperature = float(temperature)
    except Exception:
        temperature = 0.7

    temperature = max(0.0, min(2.0, temperature))

    conn = get_db()

    chat_row = conn.execute(
        "SELECT id, title FROM chats WHERE id = ?",
        (chat_id,)
    ).fetchone()

    if not chat_row:
        conn.close()
        return jsonify({"error": "chat not found"}), 404

    history_rows = conn.execute("""
        SELECT role, content
        FROM messages
        WHERE chat_id = ?
        ORDER BY id ASC
        LIMIT 20
    """, (chat_id,)).fetchall()

    messages = [{"role": "system", "content": system_prompt}]
    for row in history_rows:
        if row["role"] in {"user", "assistant"}:
            messages.append({"role": row["role"], "content": row["content"]})

    messages.append({"role": "user", "content": user_message})

    now = datetime.now().isoformat(timespec="seconds")
    conn.execute("""
        INSERT INTO messages (chat_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
    """, (chat_id, "user", user_message, now))
    conn.commit()

    if chat_row["title"] == "New Chat":
        new_title = user_message[:30]
        conn.execute("UPDATE chats SET title = ? WHERE id = ?", (new_title, chat_id))
        conn.commit()

    log_event("USER", f"chat_id={chat_id} | {user_message[:80]}")
    log_event("INFO", f"模型: {model} | temperature: {temperature}")

    def generate():
        full_reply = ""

        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True
            )

            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta
                    content = delta.content or ""
                except Exception:
                    content = ""

                if content:
                    full_reply += content
                    yield f"data: {json.dumps({'type': 'token', 'content': content}, ensure_ascii=False)}\n\n"

            conn2 = get_db()
            conn2.execute("""
                INSERT INTO messages (chat_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
            """, (
                chat_id,
                "assistant",
                full_reply,
                datetime.now().isoformat(timespec="seconds")
            ))
            conn2.commit()
            conn2.close()

            log_event("AI", f"chat_id={chat_id} | {full_reply[:120]}")
            yield f"data: {json.dumps({'type': 'done', 'full_content': full_reply}, ensure_ascii=False)}\n\n"

        except Exception as e:
            log_event("ERROR", f"API 呼叫失敗: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

        finally:
            conn.close()

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    init_db()
    log_event("INFO", "SQLite 初始化完成")
    log_event("INFO", "Flask server 準備啟動")
    threading.Timer(1.0, open_browser).start()
    app.run(debug=False, port=PORT)