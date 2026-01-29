function sendMessage() {
  const input = document.getElementById("user-input");
  const message = input.value.trim();
  if (!message) return;

  const chatBox = document.getElementById("chat-box");

  // User message
  chatBox.innerHTML += `
    <div class="message user">${message}</div>
  `;
  input.value = "";
  chatBox.scrollTop = chatBox.scrollHeight;

  fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message })
  })
  .then(res => res.json())
  .then(data => {
    chatBox.innerHTML += `
      <div class="message bot"><b>Sartun:</b> ${data.response}</div>
    `;
    chatBox.scrollTop = chatBox.scrollHeight;
  });
}
