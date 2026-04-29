import os
import re
import ast
import json
import math
import base64
import sqlite3
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, request, jsonify, render_template_string, Response, stream_with_context
from dotenv import load_dotenv
from openai import OpenAI
from werkzeug.utils import secure_filename
from pypdf import PdfReader
from docx import Document

"""
HW2 Upgrade Version
Features added based on HW1:
1. Long-term memory
2. Multimodal upload support: text / code / csv / pdf / docx / image metadata
3. Auto routing between models
4. Tool use with MCP-like tool registry
5. Safer calculator tool
6. Error fallback
7. Reflection / answer quality check option
8. Better context construction from chat history, uploaded files, and memory
"""

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chat.db"
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

TEXT_EXTENSIONS = {"txt", "md", "py", "json", "csv"}
DOCUMENT_EXTENSIONS = {"pdf", "docx"}
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
ALLOWED_EXTENSIONS = TEXT_EXTENSIONS | DOCUMENT_EXTENSIONS | IMAGE_EXTENSIONS

PORT = 5000

# Multiple-model registry for HW2 auto-routing.
# You can override any model in .env, but by default these are different models,
# so the router really demonstrates "auto routing between models".
MODEL_REGISTRY = {
    "general": os.getenv("GENERAL_MODEL", "openai/gpt-oss-20b"),
    "fast": os.getenv("FAST_MODEL", "llama-3.1-8b-instant"),
    "coding": os.getenv("CODING_MODEL", "llama-3.3-70b-versatile"),
    "reasoning": os.getenv("REASONING_MODEL", "openai/gpt-oss-120b"),
    "tool": os.getenv("TOOL_MODEL", "llama-3.1-8b-instant"),
}

DEFAULT_MODEL = MODEL_REGISTRY["general"]
FAST_MODEL = MODEL_REGISTRY["fast"]
REASONING_MODEL = MODEL_REGISTRY["reasoning"]
CODING_MODEL = MODEL_REGISTRY["coding"]
TOOL_MODEL = MODEL_REGISTRY["tool"]

MAX_HISTORY_MESSAGES = 20
MAX_FILE_CONTEXT_CHARS = 12000
MAX_MEMORY_ITEMS = 6
MAX_MEMORY_CHARS = 6000

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12MB

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("找不到 GROQ_API_KEY，請檢查 .env")

client = OpenAI(
    api_key=api_key,
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
)


def log_event(level: str, message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] [{level}] {message}")


def open_browser() -> None:
    webbrowser.open(f"http://127.0.0.1:{PORT}")


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
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
            file_kind TEXT DEFAULT 'text',
            created_at TEXT NOT NULL,
            FOREIGN KEY (chat_id) REFERENCES chats (id)
        )
    """)

    # Older HW1 databases may not have file_kind.
    try:
        cur.execute("ALTER TABLE files ADD COLUMN file_kind TEXT DEFAULT 'text'")
    except sqlite3.OperationalError:
        pass

    cur.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            content TEXT NOT NULL,
            importance INTEGER NOT NULL DEFAULT 1,
            tags TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (chat_id) REFERENCES chats (id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS tool_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            tool_name TEXT NOT NULL,
            tool_input TEXT,
            tool_output TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (chat_id) REFERENCES chats (id)
        )
    """)

    conn.commit()
    conn.close()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_kind(file_path: Path) -> str:
    suffix = file_path.suffix.lower().replace(".", "")
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in DOCUMENT_EXTENSIONS:
        return "document"
    return "text"


def extract_text_from_file(file_path: Path) -> str:
    suffix = file_path.suffix.lower()

    try:
        if suffix in {".txt", ".md", ".py", ".json", ".csv"}:
            return file_path.read_text(encoding="utf-8", errors="ignore")[:20000]

        if suffix == ".pdf":
            reader = PdfReader(str(file_path))
            parts = []
            for page_index, page in enumerate(reader.pages[:20], start=1):
                text = page.extract_text() or ""
                if text.strip():
                    parts.append(f"[PDF page {page_index}]\n{text}")
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

        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            size_kb = file_path.stat().st_size / 1024
            return (
                f"[Image uploaded: {file_path.name}]\n"
                f"File size: {size_kb:.1f} KB\n"
                "This HW2 system can store image files as multimodal context. "
                "If a vision-capable model is configured, this file can be sent to the model."
            )

    except Exception as e:
        return f"[讀取檔案失敗: {e}]"

    return "[目前不支援此檔案格式]"


# -------------------------
# Long-term memory
# -------------------------

def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]+", text.lower())


def score_memory(query: str, content: str, importance: int) -> float:
    q_tokens = set(tokenize(query))
    c_tokens = set(tokenize(content))
    if not q_tokens or not c_tokens:
        return float(importance)
    overlap = len(q_tokens & c_tokens)
    return overlap * 2.0 + importance * 0.5


def retrieve_memory(chat_id: int, query: str, limit: int = MAX_MEMORY_ITEMS) -> str:
    conn = get_db()
    rows = conn.execute("""
        SELECT content, importance, tags, created_at
        FROM memory
        WHERE chat_id IS NULL OR chat_id = ?
        ORDER BY id DESC
        LIMIT 80
    """, (chat_id,)).fetchall()
    conn.close()

    scored = []
    for row in rows:
        s = score_memory(query, row["content"], row["importance"])
        scored.append((s, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [row for score, row in scored[:limit] if score > 0]

    if not selected:
        return ""

    parts = []
    total = 0
    for row in selected:
        item = f"- {row['content']}"
        if row["tags"]:
            item += f"  (tags: {row['tags']})"
        if total + len(item) > MAX_MEMORY_CHARS:
            break
        parts.append(item)
        total += len(item)
    return "\n".join(parts)


def should_save_memory(user_message: str) -> bool:
    triggers = [
        "remember", "記住", "以後", "from now on", "preference", "偏好",
        "我喜歡", "我不喜歡", "my name", "我是", "我正在做", "我的專題"
    ]
    if any(t.lower() in user_message.lower() for t in triggers):
        return True
    return len(user_message) >= 80 and any(word in user_message for word in ["專題", "作業", "project", "homework", "HW"])


def save_memory(chat_id: Optional[int], content: str, importance: int = 1, tags: str = "auto") -> None:
    content = content.strip()
    if not content:
        return
    conn = get_db()
    conn.execute("""
        INSERT INTO memory (chat_id, content, importance, tags, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        chat_id,
        content[:1200],
        max(1, min(5, int(importance))),
        tags[:200],
        datetime.now().isoformat(timespec="seconds")
    ))
    conn.commit()
    conn.close()


# -------------------------
# Safe tools / MCP-like registry
# -------------------------

class SafeCalculator(ast.NodeVisitor):
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.USub, ast.UAdd, ast.Load, ast.Call, ast.Name
    )
    allowed_names = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "round": round,
    }

    def visit(self, node):
        if not isinstance(node, self.allowed_nodes):
            raise ValueError(f"Unsupported expression: {type(node).__name__}")
        return super().visit(node)

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name) or node.func.id not in self.allowed_names:
            raise ValueError("Function not allowed")
        for arg in node.args:
            self.visit(arg)

    def visit_Name(self, node):
        if node.id not in self.allowed_names:
            raise ValueError(f"Name not allowed: {node.id}")


def calculator_tool(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    SafeCalculator().visit(tree)
    result = eval(compile(tree, "<calculator>", "eval"), {"__builtins__": {}}, SafeCalculator.allowed_names)
    return str(result)


def current_time_tool(_: str = "") -> str:
    return datetime.now().isoformat(timespec="seconds")


def search_memory_tool(chat_id: int, query: str) -> str:
    result = retrieve_memory(chat_id, query)
    return result or "No relevant memory found."


def save_memory_tool(chat_id: int, text: str) -> str:
    save_memory(chat_id, text, importance=3, tags="manual-tool")
    return "Memory saved."


TOOL_REGISTRY = {
    "calculator": {
        "description": "Evaluate a safe math expression. Example input: 'sqrt(16) + 2'.",
        "handler": calculator_tool,
    },
    "current_time": {
        "description": "Return current server time.",
        "handler": current_time_tool,
    },
    "search_memory": {
        "description": "Search long-term memory for relevant information.",
        "handler": None,
    },
    "save_memory": {
        "description": "Save important information to long-term memory.",
        "handler": None,
    },
}


def tool_instruction() -> str:
    tool_lines = []
    for name, spec in TOOL_REGISTRY.items():
        tool_lines.append(f"- {name}: {spec['description']}")
    return (
        "\n\nTool use protocol:\n"
        "You may request a tool only when useful. If a tool is needed, output exactly one JSON object:\n"
        "{\"tool\": \"tool_name\", \"input\": \"tool input\"}\n"
        "Available tools:\n" + "\n".join(tool_lines) +
        "\nAfter receiving a tool result, answer the user normally."
    )


def parse_tool_call(text: str) -> Optional[Dict[str, str]]:
    stripped = text.strip()
    match = re.search(r"\{\s*\"tool\"\s*:\s*\".*?\"\s*,\s*\"input\"\s*:\s*\".*?\"\s*\}", stripped, re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict) and "tool" in obj and "input" in obj:
            return {"tool": str(obj["tool"]), "input": str(obj["input"])}
    except Exception:
        return None
    return None



def detect_direct_tool_request(user_message: str) -> Optional[Dict[str, str]]:
    text = user_message.strip()
    parsed = parse_tool_call(text)
    if parsed:
        return parsed
    m = re.match(r"^\s*(calculator|calc|計算機)\s*[:：]\s*(.+)$", text, re.I)
    if m:
        return {"tool": "calculator", "input": m.group(2).strip().replace("^", "**")}
    if re.search(r"(current\s*time|現在時間|幾點|what\s*time)", text, re.I):
        return {"tool": "current_time", "input": ""}
    m = re.search(r"(?:用工具計算|幫我算|請計算|計算)\s*([0-9\s\.\+\-\*\/\%\(\)\^]+)", text, re.I)
    if m:
        expr = m.group(1).strip().replace("^", "**")
        if expr:
            return {"tool": "calculator", "input": expr}
    m = re.match(r"^\s*(save_memory|記憶|記住)\s*[:：]\s*(.+)$", text, re.I)
    if m:
        return {"tool": "save_memory", "input": m.group(2).strip()}
    m = re.match(r"^\s*(search_memory|查記憶|搜尋記憶)\s*[:：]\s*(.+)$", text, re.I)
    if m:
        return {"tool": "search_memory", "input": m.group(2).strip()}
    return None


def format_direct_tool_answer(tool_name: str, tool_input: str, tool_output: str) -> str:
    if tool_name == "calculator":
        return f"工具 calculator 執行結果：\n\n{tool_input} = {tool_output}"
    return f"工具 {tool_name} 執行結果：\n\n{tool_output}"

def execute_tool(chat_id: int, tool_name: str, tool_input: str) -> str:
    if tool_name == "search_memory":
        output = search_memory_tool(chat_id, tool_input)
    elif tool_name == "save_memory":
        output = save_memory_tool(chat_id, tool_input)
    elif tool_name in TOOL_REGISTRY and TOOL_REGISTRY[tool_name]["handler"]:
        output = TOOL_REGISTRY[tool_name]["handler"](tool_input)
    else:
        output = f"Unknown tool: {tool_name}"

    conn = get_db()
    conn.execute("""
        INSERT INTO tool_logs (chat_id, tool_name, tool_input, tool_output, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        chat_id,
        tool_name,
        tool_input,
        output[:4000],
        datetime.now().isoformat(timespec="seconds")
    ))
    conn.commit()
    conn.close()
    return output


# -------------------------
# Auto model routing
# -------------------------

def route_model(user_message: str, requested_model: Optional[str] = None) -> Tuple[str, str]:
    if requested_model and requested_model.strip() and requested_model.strip() != "auto":
        return requested_model.strip(), "manual"

    msg = user_message.lower()

    code_keywords = [
        "code", "python", "flask", "debug", "bug", "function", "class",
        "程式", "程式碼", "錯誤", "除錯", "verilog", "c++", "javascript", "html", "css"
    ]
    reasoning_keywords = [
        "proof", "derive", "why", "explain", "complexity",
        "數學", "證明", "公式", "推導", "為什麼", "複雜度", "解釋"
    ]
    tool_keywords = [
        "calculator", "用工具", "tool", "現在時間", "current time",
        "幫我算", "請計算", "計算", "+", "-", "*", "/"
    ]

    # Tool tasks are checked before reasoning tasks because words like "計算"
    # should route to the tool-capable path.
    if any(k in msg for k in tool_keywords):
        return TOOL_MODEL, "tool"
    if any(k in msg for k in code_keywords):
        return CODING_MODEL, "coding"
    if any(k in msg for k in reasoning_keywords):
        return REASONING_MODEL, "reasoning"
    if len(user_message) < 80:
        return FAST_MODEL, "fast"
    return DEFAULT_MODEL, "general"


def get_recent_file_context(chat_id: int) -> str:
    conn = get_db()
    rows = conn.execute("""
        SELECT original_name, file_kind, extracted_text, created_at
        FROM files
        WHERE chat_id = ?
        ORDER BY id DESC
        LIMIT 5
    """, (chat_id,)).fetchall()
    conn.close()

    if not rows:
        return ""

    parts = []
    total = 0
    for row in rows:
        text = row["extracted_text"] or ""
        block = (
            f"[Uploaded file: {row['original_name']} | kind={row['file_kind']} | uploaded_at={row['created_at']}]\n"
            f"{text[:4000]}"
        )
        if total + len(block) > MAX_FILE_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def build_messages(chat_id: int, user_message: str, system_prompt: str) -> List[Dict[str, str]]:
    memory_text = retrieve_memory(chat_id, user_message)
    file_context = get_recent_file_context(chat_id)

    upgraded_system = (
        system_prompt.strip()
        + "\n\nYou are an HW2 upgraded AI assistant with long-term memory, multimodal file context, auto-routing, and tool use."
        + "\nAnswer in the user's language unless asked otherwise."
        + tool_instruction()
    )

    messages = [{"role": "system", "content": upgraded_system}]

    if memory_text:
        messages.append({"role": "system", "content": f"Relevant long-term memory:\n{memory_text}"})

    if file_context:
        messages.append({"role": "system", "content": f"Recent uploaded file context:\n{file_context}"})

    conn = get_db()
    history_rows = conn.execute("""
        SELECT role, content
        FROM messages
        WHERE chat_id = ?
        ORDER BY id ASC
        LIMIT ?
    """, (chat_id, MAX_HISTORY_MESSAGES)).fetchall()
    conn.close()

    for row in history_rows:
        if row["role"] in {"user", "assistant"}:
            messages.append({"role": row["role"], "content": row["content"]})

    messages.append({"role": "user", "content": user_message})
    return messages


def call_llm_once(model: str, messages: List[Dict[str, str]], temperature: float, stream: bool = False):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=stream
    )


def non_stream_completion(model: str, messages: List[Dict[str, str]], temperature: float) -> str:
    resp = call_llm_once(model, messages, temperature, stream=False)
    return resp.choices[0].message.content or ""


def reflect_answer_if_needed(model: str, user_message: str, answer: str) -> str:
    # Keep this lightweight. It only adds a short note when an answer may be incomplete.
    if len(answer) < 80 or len(user_message) < 40:
        return answer
    try:
        critique_messages = [
            {"role": "system", "content": "Check the answer briefly. If it is acceptable, say OK. If something important is missing, give one short improvement note in Chinese."},
            {"role": "user", "content": f"Question:\n{user_message}\n\nAnswer:\n{answer}"}
        ]
        critique = non_stream_completion(model, critique_messages, temperature=0.0).strip()
        if critique and critique.upper() != "OK" and "OK" not in critique[:10]:
            return answer + "\n\n---\n自我檢查補充：" + critique[:300]
    except Exception:
        return answer
    return answer


@app.route("/")
def home():
    return render_template_string("""
<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HW2 AI Assistant</title>
<style>
body{margin:0;font-family:Arial,"Microsoft JhengHei",sans-serif;background:#eef3fb;color:#172033;height:100vh;overflow:hidden}
.app{display:grid;grid-template-columns:280px 1fr;gap:16px;padding:16px;height:100vh}
.side,.main{background:#fff;border:1px solid #d9e2ef;border-radius:22px;box-shadow:0 12px 32px #0001;overflow:hidden}
.side{display:flex;flex-direction:column}.brand{padding:20px;border-bottom:1px solid #e5eaf2}.brand h1{margin:0 0 6px;font-size:21px}.brand p{margin:0;color:#667085;font-size:13px;line-height:1.45}
.new{margin:14px;padding:12px;border:0;border-radius:14px;background:#2563eb;color:white;font-weight:700;cursor:pointer}.label{padding:5px 18px;color:#667085;font-size:12px;font-weight:700}.list{padding:0 12px 12px;overflow:auto}
.chat{padding:12px;border-radius:14px;margin:6px 0;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.chat:hover,.chat.active{background:#edf4ff;color:#1d4ed8;font-weight:700}
.main{display:grid;grid-template-rows:auto 1fr auto}.top{display:flex;justify-content:space-between;gap:12px;align-items:center;padding:16px 20px;border-bottom:1px solid #e5eaf2}.top h2{margin:0;font-size:19px}.status{font-size:13px;color:#667085;margin-top:4px}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap}select{border:1px solid #d9e2ef;border-radius:12px;padding:9px;background:white}.upload{border:1px dashed #9db8e8;background:#f5f8ff;color:#1d4ed8;border-radius:12px;padding:9px 12px;font-weight:700;cursor:pointer}.upload input{display:none}
.msgs{padding:22px;overflow:auto}.empty{max-width:760px;margin:8vh auto;padding:32px;border:1px solid #d9e2ef;border-radius:22px;background:#fff;box-shadow:0 12px 32px #0001;text-align:center}.empty h3{font-size:26px;margin:0 0 10px}.empty p{color:#667085}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;text-align:left}.card{border:1px solid #d9e2ef;border-radius:16px;padding:14px;cursor:pointer;background:#fff}.card:hover{border-color:#2563eb;transform:translateY(-1px)}.card b{display:block;margin-bottom:6px}.card span{font-size:13px;color:#667085}
.row{display:flex;margin:16px 0}.row.user{justify-content:flex-end}.bubble{max-width:min(850px,84%);border:1px solid #d9e2ef;border-radius:18px;background:#fff;box-shadow:0 8px 20px #0001;overflow:hidden}.user .bubble{background:#e8f0ff;border-color:#bcd0ff}
.head{display:flex;justify-content:space-between;padding:10px 14px;background:#f8fafc;border-bottom:1px solid #e5eaf2;color:#667085;font-size:12px;font-weight:800}.body{padding:15px 16px;line-height:1.75;font-size:15px;word-break:break-word}
.body p{margin:0 0 12px}.body pre{background:#0f172a;color:#e5e7eb;border-radius:14px;padding:14px;overflow:auto}.body code{font-family:Consolas,monospace}.body :not(pre)>code{background:#eef2ff;color:#1e3a8a;padding:2px 6px;border-radius:8px}
.composer{padding:16px 20px 20px;border-top:1px solid #e5eaf2}.box{display:grid;grid-template-columns:1fr 92px;gap:12px}textarea{min-height:58px;max-height:150px;resize:vertical;border:1px solid #d9e2ef;border-radius:18px;padding:14px;font:inherit}.send{border:0;border-radius:18px;background:#2563eb;color:white;font-weight:800;cursor:pointer}.hint{font-size:12px;color:#667085;margin-top:8px}
.toast{position:fixed;right:22px;bottom:22px;background:#111827;color:white;padding:12px 14px;border-radius:14px;opacity:0;transition:.2s}.toast.show{opacity:1}
@media(max-width:850px){body{overflow:auto}.app{grid-template-columns:1fr;height:auto}.grid{grid-template-columns:1fr}.bubble{max-width:95%}}
</style>
</head>
<body>
<div class="app">
<aside class="side">
  <div class="brand"><h1>HW2 AI Assistant</h1><p>Multi-model routing · Tool use/MCP · Memory · Multimodal files</p></div>
  <button class="new" onclick="createChat()">＋ New Chat</button>
  <div class="label">CHATS</div><div id="chatList" class="list"></div>
</aside>
<main class="main">
<header class="top">
  <div><h2 id="chatTitle">New Chat</h2><div id="status" class="status">Ready</div></div>
  <div class="controls"><select id="model"><option value="">Auto routing</option></select><label class="upload">📎 Upload<input id="file" type="file" onchange="uploadFile()"></label></div>
</header>
<section id="msgs" class="msgs">
<div class="empty"><h3>開始測試 HW2 功能</h3><p>點下面卡片可快速 demo，回答會用方框區隔，code 也會自動變成深色區塊。</p>
<div class="grid">
<div class="card" onclick="demo('請幫我寫一個 Python factorial function')"><b>Auto routing: coding</b><span>觸發 coding model</span></div>
<div class="card" onclick="demo('請證明 binary search 是 O(log n)')"><b>Auto routing: reasoning</b><span>觸發 reasoning model</span></div>
<div class="card" onclick="demo('calculator: 12345 * 6789')"><b>Tool use / MCP</b><span>呼叫 calculator tool</span></div>
<div class="card" onclick="demo('記住: 我正在做 HW2 AI assistant project')"><b>Long-term memory</b><span>存進 SQLite memory</span></div>
</div></div>
</section>
<footer class="composer">
  <div class="box"><textarea id="input" placeholder="輸入訊息，例如：calculator: 2048 * 32" onkeydown="key(event)"></textarea><button id="send" class="send" onclick="sendMsg()">Send</button></div>
  <div class="hint">Enter 送出，Shift+Enter 換行。Demo 時把 terminal 放旁邊看 ROUTER / TOOL / MEMORY log。</div>
</footer>
</main></div><div id="toast" class="toast"></div>
<script>
let cid=null,busy=false;const $=id=>document.getElementById(id),msgs=$('msgs'),input=$('input'),send=$('send'),chatList=$('chatList'),status=$('status'),title=$('chatTitle'),model=$('model');
function esc(s){return(s||'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'","&#039;")}
function md(t){let p=esc(t).split(/```/g),h='';for(let i=0;i<p.length;i++){if(i%2){let c=p[i],n=c.indexOf('\\n');if(n!=-1&&c.slice(0,n).length<25)c=c.slice(n+1);h+=`<pre><code>${c.trim()}</code></pre>`}else h+=p[i].replace(/`([^`]+)`/g,'<code>$1</code>').split(/\\n\\s*\\n/g).map(x=>x.trim()?`<p>${x.trim().replaceAll('\\n','<br>')}</p>`:'').join('')}return h}
function clean(){let e=msgs.querySelector('.empty');if(e)e.remove()}function add(role,text){clean();let r=document.createElement('div');r.className='row '+role;let lab=role=='user'?'YOU':'ASSISTANT',tm=new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});r.innerHTML=`<div class="bubble"><div class="head"><span>${lab}</span><span>${tm}</span></div><div class="body">${md(text)}</div></div>`;msgs.appendChild(r);msgs.scrollTop=msgs.scrollHeight;return r.querySelector('.body')}
function note(s){status.textContent=s}function toast(s){let t=$('toast');t.textContent=s;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),1800)}function demo(s){input.value=s;input.focus()}function key(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMsg()}}
async function loadModels(){try{let r=await fetch('/api/models');if(!r.ok)return;let m=await r.json();for(let [k,v] of Object.entries(m)){let o=document.createElement('option');o.value=v;o.textContent=k+': '+v;model.appendChild(o)}}catch(e){}}
async function chats(){let r=await fetch('/api/chats'),a=await r.json();chatList.innerHTML='';a.forEach(c=>{let d=document.createElement('div');d.className='chat'+(c.id===cid?' active':'');d.textContent=c.title||'New Chat';d.onclick=()=>select(c.id,c.title);chatList.appendChild(d)});if(!cid&&a.length)await select(a[0].id,a[0].title);else if(!cid)await createChat()}
async function createChat(){let r=await fetch('/api/chats',{method:'POST'}),c=await r.json();cid=c.id;title.textContent=c.title;msgs.innerHTML='<div class="empty"><h3>這是新的對話</h3><p>輸入訊息或上傳檔案開始 demo。</p></div>';note('New chat created');await chatsOnly()}
async function chatsOnly(){let r=await fetch('/api/chats'),a=await r.json();chatList.innerHTML='';a.forEach(c=>{let d=document.createElement('div');d.className='chat'+(c.id===cid?' active':'');d.textContent=c.title||'New Chat';d.onclick=()=>select(c.id,c.title);chatList.appendChild(d)})}
async function select(id,t){cid=id;title.textContent=t||'Chat '+id;await chatsOnly();let r=await fetch(`/api/chats/${id}/messages`),a=await r.json();msgs.innerHTML='';if(!a.length)msgs.innerHTML='<div class="empty"><h3>這是新的對話</h3><p>輸入訊息或上傳檔案開始 demo。</p></div>';else a.forEach(x=>add(x.role,x.content));note('Chat #'+id)}
async function uploadFile(){if(!cid)await createChat();let f=$('file').files[0];if(!f)return;let fd=new FormData();fd.append('file',f);note('Uploading...');try{let r=await fetch(`/api/chats/${cid}/upload`,{method:'POST',body:fd}),d=await r.json();if(!r.ok)throw Error(d.error||'Upload failed');toast('Uploaded: '+d.filename);await select(cid,title.textContent)}catch(e){toast(e.message)}$('file').value=''}
async function sendMsg(){if(busy)return;let message=input.value.trim();if(!message)return;if(!cid)await createChat();busy=true;send.disabled=true;input.value='';add('user',message);let box=add('assistant',''),full='';note('Thinking...');try{let r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({chat_id:cid,message,model:model.value,temperature:.7,system_prompt:'You are a helpful assistant. Use clean structure with code blocks when needed.'})});let reader=r.body.getReader(),dec=new TextDecoder(),buf='';while(true){let {value,done}=await reader.read();if(done)break;buf+=dec.decode(value,{stream:true});let evs=buf.split('\\n\\n');buf=evs.pop();for(let ev of evs){if(!ev.startsWith('data:'))continue;let p=JSON.parse(ev.slice(5).trim());if(p.type==='token'){full+=p.content;box.innerHTML=md(full);msgs.scrollTop=msgs.scrollHeight}else if(p.type==='done'){if(p.full_content){full=p.full_content;box.innerHTML=md(full)}note('Done')}else if(p.type==='error'){box.innerHTML=md('Error: '+p.content);note('Error')}}}await chatsOnly()}catch(e){box.innerHTML=md('Error: '+e.message);note('Error')}busy=false;send.disabled=false;input.focus()}
loadModels();chats();
</script>
</body></html>
    """)

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
        {"id": row["id"], "title": row["title"], "created_at": row["created_at"]}
        for row in chats
    ])


@app.route("/api/chats", methods=["POST"])
def create_chat():
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO chats (title, created_at) VALUES (?, ?)", ("New Chat", now))
    chat_id = cur.lastrowid
    conn.commit()
    conn.close()

    log_event("INFO", f"建立新聊天 chat_id={chat_id}")
    return jsonify({"id": chat_id, "title": "New Chat", "created_at": now})


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
        {"id": row["id"], "role": row["role"], "content": row["content"], "created_at": row["created_at"]}
        for row in rows
    ])


@app.route("/api/chats/<int:chat_id>/files", methods=["GET"])
def list_files(chat_id):
    conn = get_db()
    rows = conn.execute("""
        SELECT id, original_name, file_kind, created_at
        FROM files
        WHERE chat_id = ?
        ORDER BY id DESC
    """, (chat_id,)).fetchall()
    conn.close()

    return jsonify([
        {
            "id": row["id"],
            "original_name": row["original_name"],
            "file_kind": row["file_kind"],
            "created_at": row["created_at"]
        }
        for row in rows
    ])


@app.route("/api/chats/<int:chat_id>/memory", methods=["GET"])
def list_memory(chat_id):
    conn = get_db()
    rows = conn.execute("""
        SELECT id, content, importance, tags, created_at
        FROM memory
        WHERE chat_id IS NULL OR chat_id = ?
        ORDER BY id DESC
        LIMIT 50
    """, (chat_id,)).fetchall()
    conn.close()
    return jsonify([
        {
            "id": row["id"],
            "content": row["content"],
            "importance": row["importance"],
            "tags": row["tags"],
            "created_at": row["created_at"]
        }
        for row in rows
    ])


@app.route("/api/chats/<int:chat_id>/memory", methods=["POST"])
def add_memory(chat_id):
    data = request.get_json(silent=True) or {}
    content = (data.get("content") or "").strip()
    importance = data.get("importance", 3)
    tags = (data.get("tags") or "manual").strip()
    if not content:
        return jsonify({"error": "content is required"}), 400
    save_memory(chat_id, content, importance=importance, tags=tags)
    return jsonify({"success": True})


@app.route("/api/chats/<int:chat_id>/upload", methods=["POST"])
def upload_file(chat_id):
    if "file" not in request.files:
        return jsonify({"error": "沒有收到檔案"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "檔名是空的"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "目前支援 txt / md / py / json / csv / pdf / docx / png / jpg / jpeg / webp"}), 400

    original_name = file.filename
    safe_name = secure_filename(original_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_name = f"{timestamp}_{safe_name}"
    save_path = UPLOAD_FOLDER / saved_name
    file.save(save_path)

    extracted_text = extract_text_from_file(save_path)
    file_kind = get_file_kind(save_path)
    now = datetime.now().isoformat(timespec="seconds")

    conn = get_db()
    chat_row = conn.execute("SELECT id FROM chats WHERE id = ?", (chat_id,)).fetchone()
    if not chat_row:
        conn.close()
        return jsonify({"error": "chat not found"}), 404

    conn.execute("""
        INSERT INTO files (chat_id, original_name, saved_name, content_type, extracted_text, file_kind, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        chat_id,
        original_name,
        saved_name,
        file.content_type,
        extracted_text,
        file_kind,
        now
    ))

    snippet = extracted_text[:3000]
    file_note = (
        f"[使用者上傳檔案: {original_name}]\n"
        f"file_kind: {file_kind}\n\n"
        f"以下是可供對話參考的檔案內容摘要：\n{snippet}"
    )

    conn.execute("""
        INSERT INTO messages (chat_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
    """, (chat_id, "assistant", file_note, now))

    # Store a compact memory for future retrieval.
    conn.execute("""
        INSERT INTO memory (chat_id, content, importance, tags, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        chat_id,
        f"User uploaded {file_kind} file named {original_name}. Summary: {snippet[:700]}",
        2,
        f"file,{file_kind}",
        now
    ))

    conn.commit()
    conn.close()

    log_event("FILE", f"chat_id={chat_id} 上傳檔案: {original_name} kind={file_kind}")
    return jsonify({"success": True, "filename": original_name, "file_kind": file_kind, "preview": snippet})


@app.route("/api/tools", methods=["GET"])
def list_tools():
    return jsonify({name: spec["description"] for name, spec in TOOL_REGISTRY.items()})


@app.route("/api/models", methods=["GET"])
def list_models():
    return jsonify(MODEL_REGISTRY)


@app.route("/api/route", methods=["POST"])
def route_preview():
    data = request.get_json(silent=True) or {}
    message = data.get("message") or ""
    model, reason = route_model(message, data.get("model"))
    return jsonify({"model": model, "reason": reason})



@app.route("/api/tools/execute", methods=["POST"])
def api_execute_tool():
    data = request.get_json(silent=True) or {}
    chat_id = data.get("chat_id")
    tool_name = (data.get("tool") or "").strip()
    tool_input = str(data.get("input") or "")
    if not chat_id:
        return jsonify({"error": "chat_id is required"}), 400
    if not tool_name:
        return jsonify({"error": "tool is required"}), 400
    try:
        output = execute_tool(int(chat_id), tool_name, tool_input)
        log_event("TOOL", f"api tool={tool_name} input={tool_input[:80]} output={output[:80]}")
        return jsonify({"success": True, "tool": tool_name, "input": tool_input, "output": output})
    except Exception as e:
        log_event("TOOL_ERROR", f"api tool={tool_name} failed: {e}")
        return jsonify({"success": False, "tool": tool_name, "error": str(e)}), 400

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}

    chat_id = data.get("chat_id")
    user_message = (data.get("message") or "").strip()
    system_prompt = (data.get("system_prompt") or "You are a helpful assistant.").strip()
    requested_model = (data.get("model") or "auto").strip()
    temperature = data.get("temperature", 0.7)
    enable_reflection = bool(data.get("reflection", False))

    if not chat_id:
        return jsonify({"error": "chat_id is required"}), 400
    if not user_message:
        return jsonify({"error": "message is required"}), 400

    try:
        chat_id = int(chat_id)
    except Exception:
        return jsonify({"error": "invalid chat_id"}), 400

    try:
        temperature = float(temperature)
    except Exception:
        temperature = 0.7
    temperature = max(0.0, min(2.0, temperature))

    conn = get_db()
    chat_row = conn.execute("SELECT id, title FROM chats WHERE id = ?", (chat_id,)).fetchone()
    if not chat_row:
        conn.close()
        return jsonify({"error": "chat not found"}), 404

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
    conn.close()

    if should_save_memory(user_message):
        save_memory(chat_id, f"User said: {user_message}", importance=3, tags="auto-user")

    model, route_reason = route_model(user_message, requested_model)
    messages = build_messages(chat_id, user_message, system_prompt)

    log_event("USER", f"chat_id={chat_id} | {user_message[:80]}")
    log_event("ROUTE", f"model={model} | reason={route_reason} | temperature={temperature}")

    direct_tool_call = detect_direct_tool_request(user_message)
    if direct_tool_call:
        tool_name = direct_tool_call["tool"]
        tool_input = direct_tool_call["input"]
        try:
            tool_output = execute_tool(chat_id, tool_name, tool_input)
            final_reply = format_direct_tool_answer(tool_name, tool_input, tool_output)
            conn_tool = get_db()
            conn_tool.execute("""
                INSERT INTO messages (chat_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
            """, (chat_id, "assistant", final_reply, datetime.now().isoformat(timespec="seconds")))
            conn_tool.commit()
            conn_tool.close()
            log_event("TOOL", f"direct tool={tool_name} input={tool_input[:80]} output={tool_output[:80]}")

            def direct_generate():
                yield f"data: {json.dumps({'type': 'token', 'content': final_reply}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'full_content': final_reply, 'model': model, 'route_reason': 'direct_tool'}, ensure_ascii=False)}\n\n"

            return Response(
                stream_with_context(direct_generate()),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
            )
        except Exception as e:
            log_event("TOOL_ERROR", f"direct tool={tool_name} failed: {e}")

    def generate():
        full_reply = ""
        final_reply = ""

        try:
            # First pass: stream model output.
            try:
                stream = call_llm_once(model, messages, temperature, stream=True)
            except Exception as e:
                log_event("WARN", f"主要模型失敗，fallback 到 DEFAULT_MODEL: {e}")
                model_fallback = DEFAULT_MODEL
                stream = call_llm_once(model_fallback, messages, temperature, stream=True)

            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta
                    content = delta.content or ""
                except Exception:
                    content = ""

                if content:
                    full_reply += content
                    yield f"data: {json.dumps({'type': 'token', 'content': content}, ensure_ascii=False)}\n\n"

            final_reply = full_reply

            # MCP-like tool call handling. If first response is a tool request,
            # execute the tool and ask model to produce final answer.
            tool_call = parse_tool_call(full_reply)
            if tool_call:
                tool_name = tool_call["tool"]
                tool_input = tool_call["input"]
                tool_output = execute_tool(chat_id, tool_name, tool_input)

                notice = f"\n\n[Tool executed: {tool_name}]\nResult: {tool_output}\n\n"
                yield f"data: {json.dumps({'type': 'token', 'content': notice}, ensure_ascii=False)}\n\n"

                followup_messages = messages + [
                    {"role": "assistant", "content": full_reply},
                    {"role": "user", "content": f"Tool result from {tool_name}: {tool_output}\nPlease answer the original user question now."}
                ]
                final_reply = ""
                stream2 = call_llm_once(model, followup_messages, temperature, stream=True)
                for chunk in stream2:
                    try:
                        delta = chunk.choices[0].delta
                        content = delta.content or ""
                    except Exception:
                        content = ""
                    if content:
                        final_reply += content
                        yield f"data: {json.dumps({'type': 'token', 'content': content}, ensure_ascii=False)}\n\n"

            if enable_reflection:
                reflected = reflect_answer_if_needed(model, user_message, final_reply)
                if reflected != final_reply:
                    extra = reflected[len(final_reply):]
                    final_reply = reflected
                    yield f"data: {json.dumps({'type': 'token', 'content': extra}, ensure_ascii=False)}\n\n"

            conn2 = get_db()
            conn2.execute("""
                INSERT INTO messages (chat_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
            """, (
                chat_id,
                "assistant",
                final_reply,
                datetime.now().isoformat(timespec="seconds")
            ))
            conn2.commit()
            conn2.close()

            log_event("AI", f"chat_id={chat_id} | {final_reply[:120]}")
            yield f"data: {json.dumps({'type': 'done', 'full_content': final_reply, 'model': model, 'route_reason': route_reason}, ensure_ascii=False)}\n\n"

        except Exception as e:
            log_event("ERROR", f"API 呼叫失敗: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


if __name__ == "__main__":
    init_db()
    log_event("INFO", "SQLite 初始化完成")
    log_event("INFO", "HW2 Flask server 準備啟動")
    threading.Timer(1.0, open_browser).start()
    app.run(debug=False, port=PORT)
