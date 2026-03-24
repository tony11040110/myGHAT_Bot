const messagesEl = document.getElementById("messages");
const chatArea = document.getElementById("chatArea");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChatBtn");
const welcomeScreen = document.getElementById("welcomeScreen");
const tempSlider = document.getElementById("temperature");
const tempValue = document.getElementById("tempValue");
const suggestionCards = document.querySelectorAll(".suggestion-card");
const systemPromptEl = document.getElementById("systemPrompt");
const modelEl = document.getElementById("model");
const chatHistoryEl = document.getElementById("chatHistory");
const fileInput = document.getElementById("fileInput");
const fileListEl = document.getElementById("fileList");

let chats = [];
let currentChatId = null;

tempSlider.addEventListener("input", () => {
  tempValue.textContent = tempSlider.value;
});

function autoResizeTextarea() {
  userInput.style.height = "auto";
  userInput.style.height = Math.min(userInput.scrollHeight, 180) + "px";
}
userInput.addEventListener("input", autoResizeTextarea);

function hideWelcome() {
  welcomeScreen.style.display = "none";
}

function showWelcomeIfEmpty() {
  if (!messagesEl.children.length) {
    welcomeScreen.style.display = "block";
  } else {
    welcomeScreen.style.display = "none";
  }
}

function createMessageRow(role, text) {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const inner = document.createElement("div");
  inner.className = "message-inner";

  const avatar = document.createElement("div");
  avatar.className = `message-avatar ${role}`;
  avatar.textContent = role === "assistant" ? "AI" : "U";

  const body = document.createElement("div");
  body.className = "message-content";

  const roleLabel = document.createElement("div");
  roleLabel.className = "message-role";
  roleLabel.textContent = role === "assistant" ? "Assistant" : "You";

  const content = document.createElement("div");
  content.textContent = text;

  body.appendChild(roleLabel);
  body.appendChild(content);
  inner.appendChild(avatar);
  inner.appendChild(body);
  row.appendChild(inner);

  messagesEl.appendChild(row);
  chatArea.scrollTop = chatArea.scrollHeight;
  return content;
}

async function loadChats() {
  const res = await fetch("/api/chats");
  chats = await res.json();
  renderChatHistory();

  if (!currentChatId) {
    if (chats.length > 0) {
      currentChatId = chats[0].id;
    } else {
      const newChat = await createChat();
      currentChatId = newChat.id;
    }
  }

  await loadMessages(currentChatId);
}

async function createChat() {
  const res = await fetch("/api/chats", { method: "POST" });
  const chat = await res.json();
  chats.unshift(chat);
  renderChatHistory();
  return chat;
}

async function loadMessages(chatId) {
  currentChatId = chatId;
  messagesEl.innerHTML = "";

  const res = await fetch(`/api/chats/${chatId}/messages`);
  const messages = await res.json();

  for (const msg of messages) {
    createMessageRow(msg.role, msg.content);
  }

  renderChatHistory();
  await loadFiles(chatId);
  showWelcomeIfEmpty();
}

async function loadFiles(chatId) {
  if (!fileListEl) return;
  fileListEl.innerHTML = "";

  const res = await fetch(`/api/chats/${chatId}/files`);
  const files = await res.json();

  if (!files.length) {
    fileListEl.innerHTML = `<div class="file-item">No files yet</div>`;
    return;
  }

  files.forEach(file => {
    const div = document.createElement("div");
    div.className = "file-item";
    div.textContent = file.original_name;
    fileListEl.appendChild(div);
  });
}

function renderChatHistory() {
  chatHistoryEl.innerHTML = "";

  chats.forEach(chat => {
    const item = document.createElement("button");
    item.className = "chat-history-item";
    if (chat.id === currentChatId) item.classList.add("active");

    const title = document.createElement("div");
    title.className = "chat-history-title";
    title.textContent = chat.title || "New Chat";

    item.appendChild(title);
    item.addEventListener("click", async () => {
      await loadMessages(chat.id);
    });

    chatHistoryEl.appendChild(item);
  });
}

async function handleSend(prefilledText = null) {
  const text = (prefilledText ?? userInput.value).trim();
  if (!text || !currentChatId) return;

  hideWelcome();
  createMessageRow("user", text);
  userInput.value = "";
  autoResizeTextarea();

  const assistantEl = createMessageRow("assistant", "");

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        chat_id: currentChatId,
        message: text,
        system_prompt: systemPromptEl.value,
        model: modelEl.value,
        temperature: parseFloat(tempSlider.value)
      })
    });

    if (!res.ok) {
      const errorData = await res.json();
      assistantEl.textContent = `Error: ${errorData.error || "Unknown error"}`;
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");

    let buffer = "";
    let fullReply = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();

      for (const part of parts) {
        if (!part.startsWith("data: ")) continue;

        const jsonStr = part.slice(6);

        try {
          const payload = JSON.parse(jsonStr);

          if (payload.type === "token") {
            fullReply += payload.content;
            assistantEl.textContent = fullReply;
            chatArea.scrollTop = chatArea.scrollHeight;
          } else if (payload.type === "done") {
            assistantEl.textContent = payload.full_content || fullReply;
            await loadChats();
            await loadMessages(currentChatId);
          } else if (payload.type === "error") {
            assistantEl.textContent = `Error: ${payload.content}`;
          }
        } catch (err) {
          console.error("JSON parse error:", err);
        }
      }
    }
  } catch (error) {
    assistantEl.textContent = `Request failed: ${error.message}`;
  }
}

async function handleFileUpload(file) {
  if (!file || !currentChatId) return;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`/api/chats/${currentChatId}/upload`, {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    if (!res.ok) {
      alert(data.error || "檔案上傳失敗");
      return;
    }

    await loadMessages(currentChatId);
    await loadChats();
    await loadFiles(currentChatId);
  } catch (error) {
    alert(`檔案上傳失敗: ${error.message}`);
  }
}

sendBtn.addEventListener("click", () => handleSend());

userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
});

newChatBtn.addEventListener("click", async () => {
  const newChat = await createChat();
  currentChatId = newChat.id;
  await loadMessages(currentChatId);
});

fileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  await handleFileUpload(file);
  fileInput.value = "";
});

suggestionCards.forEach(card => {
  card.addEventListener("click", () => handleSend(card.textContent));
});

loadChats();
autoResizeTextarea();