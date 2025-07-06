$(document).ready(function () {
  const input = $("#message-text");
  const sendBtn = $("#send");
  const dataList = $("#symptoms-list");
  const suggestionBox = $(".symptoms-list-container");
  const chat = $("#conversation");
  const spinner = $("#spinner");
  const micBtn = $("#mic-btn");

  const nameInput = $("#user-name");
  const ageInput = $("#user-age");
  const genderInput = $("#user-gender");
  const submitDetailsBtn = $("#submit-details");
  const chatInputBox = $("#chat-input-box");
  const pdfSection = $("#pdf-section").hide();
  const downloadBtn = $("#download-pdf");

  const askAIInput = $("#ask-ai-input");
  const askAIBtn = $("#ask-ai-btn");
  const aiReplyDiv = $("#ai-reply");
  const askAISection = $("#ask-ai-section").hide();
  const askGeneralBtn = $("#ask-general-btn");

  let typingDiv = null;
  const symptoms = window.symptoms || [];

  // 🌙 Toggle dark mode
  $("#toggle-dark").on("click", () => {
    document.body.classList.toggle("dark-mode");
  });

  // 🧑 Submit user profile
  submitDetailsBtn.on("click", function () {
    const name = nameInput.val().trim();
    const age = parseInt(ageInput.val());
    const gender = genderInput.val();

    if (!name || !age || isNaN(age) || age <= 0 || !gender) {
      alert("Please enter valid name, age, and gender.");
      return;
    }

    $.ajax({
      url: "/start",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify({ name, age, gender }),
      success: function () {
        chat.show();
        chatInputBox.show();
        appendBotMessage(`Hi <b>${name}</b>! I'm <b>Meddy</b>, your AI health assistant.<br>
        Please tell me your symptoms like <i>fever, cough, headache</i>.<br>
        When finished, type <b>done</b>.`);
      }
    });
  });

  // 🔍 Autocomplete symptom list
  input.on("input", function () {
    const value = $(this).val().toLowerCase().trim();
    dataList.empty();

    if (value.length > 1) {
      const filtered = symptoms.filter(s => s.toLowerCase().includes(value)).slice(0, 6);
      if (filtered.length) {
        filtered.forEach(s => dataList.append(`<li>${s}</li>`));
        suggestionBox.slideDown();
      } else {
        suggestionBox.slideUp();
      }
    } else {
      suggestionBox.slideUp();
    }
  });

  dataList.on("click", "li", function () {
    input.val($(this).text());
    suggestionBox.slideUp();
  });

  input.on("blur", () => setTimeout(() => suggestionBox.slideUp(), 150));

  // 📤 Send Message
  sendBtn.on("click", sendMessage);
  input.on("keypress", e => {
    if (e.which === 13) sendMessage();
  });

  function sendMessage() {
    const text = input.val().trim();
    if (!text) return;

    appendUserMessage(text);
    input.val("");
    showTyping();
    spinner.show();

    if (isGeneralChat(text)) {
      askGeneralQuestion(text);
    } else {
      const splitSymptoms = text.split(/[,]|and|then/gi).map(s => s.trim()).filter(Boolean);
      processSymptomInputs(splitSymptoms);
    }
  }

  function isGeneralChat(text) {
    return /what|how|why|can|should|could|will|do you think|meddy|symptom|advice/i.test(text);
  }

  function askGeneralQuestion(text) {
    $.ajax({
      url: "/ask_ai",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify({ message: text }),
      success: function (res) {
        hideTyping();
        spinner.hide();
        appendBotMessage(res.reply || "🤖 Meddy didn’t respond.");
      },
      error: function () {
        hideTyping();
        spinner.hide();
        appendBotMessage("⚠️ Meddy couldn't process your request.");
      }
    });
  }

  function processSymptomInputs(symptomList) {
    if (!symptomList.length) return;

    let index = 0;

    function next() {
      if (index >= symptomList.length) {
        hideTyping();
        spinner.hide();
        return;
      }

      const sentence = symptomList[index++];
      $.ajax({
        url: "/symptom",
        method: "POST",
        contentType: "application/json",
        data: JSON.stringify({ sentence }),
        success: function (res) {
          if (typeof res === "string") {
            appendBotMessage(res);
          } else if (Array.isArray(res)) {
            res.forEach(displaySymptomDetails);
            pdfSection.show();
            askAISection.show();
          }
          next();
        },
        error: function () {
          appendBotMessage("⚠️ Something went wrong.");
          next();
        }
      });
    }

    next();
  }

  function displaySymptomDetails(data) {
    const html = `
      <div class="bot-message">
        <b>🧬 Disease:</b> ${data.disease}<br/>
        <b>🩺 Symptom:</b> ${data.symptom}<br/>
        <b>🔬 Confidence:</b> ${(data.confidence * 100).toFixed(1)}%<br/>
        <b>⚠️ Severity:</b> ${data.severity}<br/>
        <b>📄 Info:</b> <i>${data.description}</i><br/>
        <b>💊 Precautions:</b> ${data.precautions.join(", ")}
      </div>
    `;
    chat.append(html);
    scrollChat();
  }

  function appendBotMessage(msg) {
    chat.append(`<div class="bot-message">${msg}</div>`);
    scrollChat();
  }

  function appendUserMessage(msg) {
    chat.append(`<div class="user-message">${msg}</div>`);
    scrollChat();
  }

  function scrollChat() {
    chat[0].scrollTop = chat[0].scrollHeight;
  }

  function showTyping() {
    typingDiv = $('<div class="bot-message typing">Meddy is typing...</div>');
    chat.append(typingDiv);
    scrollChat();
  }

  function hideTyping() {
    if (typingDiv) typingDiv.remove();
    typingDiv = null;
  }

  // 📄 PDF Download
  downloadBtn.on("click", () => {
    window.location.href = "/download_pdf";
  });

  // 🧠 Ask Meddy (AI)
  askAIBtn.on("click", () => {
    const question = askAIInput.val().trim();
    if (!question) return;

    aiReplyDiv.html("🤖 Thinking...");
    $.ajax({
      type: "POST",
      url: "/ask_ai",
      contentType: "application/json",
      data: JSON.stringify({ message: question }),
      success: res => {
        aiReplyDiv.html(`<b>Meddy:</b> ${res.reply}`);
        askAIInput.val("");
      },
      error: () => {
        aiReplyDiv.html("⚠️ Meddy didn’t respond.");
      }
    });
  });

  // ⬇️ Scroll to Ask AI section
  askGeneralBtn?.on("click", function () {
    askAISection.show();
    $("html, body").animate({
      scrollTop: askAISection.offset().top - 50
    }, 600);
  });

  // 🎤 Voice Input
  if ('webkitSpeechRecognition' in window) {
    const recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.lang = "en-US";
    recognition.interimResults = false;

    micBtn.on("click", function () {
      recognition.start();
      micBtn.prop("disabled", true).html('<i class="fas fa-spinner fa-spin"></i>');
    });

    recognition.onresult = function (event) {
      input.val(event.results[0][0].transcript);
      micBtn.prop("disabled", false).html('<i class="fas fa-microphone"></i>');
    };

    recognition.onerror = function (event) {
      alert("Speech recognition error: " + event.error);
      micBtn.prop("disabled", false).html('<i class="fas fa-microphone"></i>');
    };
  } else {
    micBtn.prop("disabled", true).attr("title", "Speech recognition not supported");
  }
});
