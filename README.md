# MyGHAT Bot

A ChatGPT-like chatbot web app built with Flask + Groq + SQLite.

## Features
- ChatGPT-style chat UI
- Chat history stored in SQLite
- Left sidebar chat records
- File upload support
- Streaming response
- Custom system prompt
- Temperature control

## Tech Stack
- Python Flask
- Groq API
- SQLite
- HTML / CSS / JavaScript

## Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2.Create .env:
   ```bash
   GROQ_API_KEY=your_key_here
   ```
3.Run:
   ```bash
   python app.py
   ```
4.Open:
   ```bash
   http://127.0.0.1:5000
   ```
