// DiagnoseAI - Modern UI JavaScript (2025)

// Debounce utility
function debounce(fn, delay) {
  let timeoutId;
  return (...args) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  };
}

// DOM Ready
$(document).ready(function () {
  // Elements
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
  const userDetailsCard = $("#user-details");
  const chatContainer = $(".chat-container");
  const chatInputBox = $("#chat-input-box");
  const pdfSection = $("#pdf-section");
  const downloadBtn = $("#download-pdf");

  const askAIInput = $("#ask-ai-input");
  const askAIBtn = $("#ask-ai-btn");
  const aiReplyDiv = $("#ai-reply");
  const askAISection = $("#ask-ai-section");
  const chatToggle = $("#chat-toggle");

  let typingDiv = null;
  const symptoms = window.symptoms || [];
  let isProcessing = false;

  // Initialize
  init();

  function init() {
    setupEventListeners();
    setupVoiceRecognition();
    setupDarkMode();
    autoResizeTextarea();
  }

  function setupEventListeners() {
    // User details submission
    submitDetailsBtn.on("click", handleUserDetailsSubmit);
    
    // Chat functionality
    sendBtn.on("click", sendMessage);
    input.on("keypress", handleInputKeypress);
    input.on("input", debounce(handleInputChange, 300));
    
    // Autocomplete
    dataList.on("click", "li", handleSymptomSelect);
    input.on("blur", () => setTimeout(() => suggestionBox.slideUp(200), 150));
    
    // AI chat
    askAIBtn.on("click", handleAIQuestion);
    askAIInput.on("keypress", (e) => {
      if (e.which === 13) handleAIQuestion();
    });
    
    // PDF download
    downloadBtn.on("click", handlePDFDownload);
    
    // Chat toggle
    chatToggle.on("click", toggleChatWindow);
  }

  function setupDarkMode() {
    const darkModeBtn = $("#toggle-dark");
    const isDarkMode = localStorage.getItem('darkMode') === 'true';
    
    if (isDarkMode) {
      document.body.classList.add('dark-mode');
    }
    
    darkModeBtn.on("click", () => {
      document.body.classList.toggle("dark-mode");
      const isNowDark = document.body.classList.contains('dark-mode');
      localStorage.setItem('darkMode', isNowDark);
    });
  }

  function autoResizeTextarea() {
    input.on('input', function() {
      this.style.height = 'auto';
      this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });
  }

  // User Details Submission
  function handleUserDetailsSubmit() {
    const name = nameInput.val().trim();
    const age = parseInt(ageInput.val());
    const gender = genderInput.val();

    if (!validateUserDetails(name, age, gender)) return;

    setButtonLoading(submitDetailsBtn, true);

    $.ajax({
      url: "/start",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify({ name, age, gender }),
      success: function (response) {
        userDetailsCard.slideUp(400, () => {
          chatContainer.slideDown(400);
          appendBotMessage(`Hi <strong>${name}</strong>! I'm <strong>Meddy</strong>, your AI health assistant.<br>
            Please describe your symptoms (e.g., "fever, cough, headache").<br>
            When you're finished, type <strong>"done"</strong>.`);
          askAISection.slideDown(400);
        });
      },
      error: function () {
        showError("Failed to start session. Please try again.");
      },
      complete: function () {
        setButtonLoading(submitDetailsBtn, false);
      }
    });
  }

  function validateUserDetails(name, age, gender) {
    if (!name) {
      showError("Please enter your name.");
      nameInput.focus();
      return false;
    }
    
    if (!age || isNaN(age) || age <= 0 || age > 120) {
      showError("Please enter a valid age between 1 and 120.");
      ageInput.focus();
      return false;
    }
    
    if (!gender) {
      showError("Please select your gender.");
      genderInput.focus();
      return false;
    }
    
    return true;
  }

  // Chat Input Handling
  function handleInputKeypress(e) {
    if (e.which === 13 && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  function handleInputChange() {
    const value = input.val().toLowerCase().trim();
    dataList.empty();

    if (value.length > 1) {
      const filtered = symptoms
        .filter(s => s.toLowerCase().includes(value))
        .slice(0, 8);
        
      if (filtered.length) {
        filtered.forEach(symptom => {
          dataList.append(`<li data-symptom="${symptom}">${symptom}</li>`);
        });
        suggestionBox.slideDown(200);
      } else {
        suggestionBox.slideUp(200);
      }
    } else {
      suggestionBox.slideUp(200);
    }
  }

  function handleSymptomSelect() {
    const symptom = $(this).data('symptom') || $(this).text();
    input.val(symptom);
    suggestionBox.slideUp(200);
    input.focus();
  }

  // Message Sending
  function sendMessage() {
    const text = input.val().trim();
    if (!text || isProcessing) return;

    appendUserMessage(text);
    input.val("").trigger('input'); // Reset height
    showTyping();
    setProcessing(true);

    if (isGeneralHealthQuery(text)) {
      handleGeneralQuery(text);
    } else {
      handleSymptomInput(text);
    }
  }

  function isGeneralHealthQuery(text) {
    const generalKeywords = /\b(what|how|why|can|should|could|will|do you think|meddy|advice|recommend|suggest|tell me about)\b/i;
    const symptomKeywords = /\b(pain|hurt|ache|fever|cough|nausea|tired|dizzy|done)\b/i;
    
    return generalKeywords.test(text) && !symptomKeywords.test(text);
  }

  function handleGeneralQuery(text) {
    $.ajax({
      url: "/ask_ai",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify({ message: text }),
      success: function (response) {
        hideTyping();
        appendBotMessage(response.reply || "🤖 I couldn't process that question. Please try rephrasing.");
      },
      error: function () {
        hideTyping();
        appendBotMessage("⚠️ I'm having trouble connecting right now. Please try again in a moment.");
      },
      complete: function () {
        setProcessing(false);
      }
    });
  }

  function handleSymptomInput(text) {
    const symptoms = text.split(/[,]|and|then/gi)
      .map(s => s.trim())
      .filter(Boolean);

    processSymptoms(symptoms, 0);
  }

  function processSymptoms(symptomList, index) {
    if (index >= symptomList.length) {
      hideTyping();
      setProcessing(false);
      return;
    }

    const symptom = symptomList[index];
    
    $.ajax({
      url: "/symptom",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify({ sentence: symptom }),
      success: function (response) {
        if (typeof response === "string") {
          appendBotMessage(response);
          processSymptoms(symptomList, index + 1);
        } else if (Array.isArray(response)) {
          hideTyping();
          displayDiagnosisResults(response);
          pdfSection.slideDown(400);
          setProcessing(false);
        }
      },
      error: function () {
        appendBotMessage("⚠️ Something went wrong processing that symptom.");
        processSymptoms(symptomList, index + 1);
      }
    });
  }

  // AI Question Handling
  function handleAIQuestion() {
    const question = askAIInput.val().trim();
    if (!question) return;

    setButtonLoading(askAIBtn, true);
    aiReplyDiv.html(`
      <div class="typing-indicator">
        <span>Meddy is thinking</span>
        <div class="typing-dots">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
    `).show();

    $.ajax({
      url: "/ask_ai",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify({ message: question }),
      success: function (response) {
        aiReplyDiv.html(`<strong>Meddy:</strong> ${response.reply || "I couldn't process that question."}`);
        askAIInput.val("");
      },
      error: function () {
        aiReplyDiv.html("⚠️ I'm having trouble connecting right now. Please try again.");
      },
      complete: function () {
        setButtonLoading(askAIBtn, false);
      }
    });
  }

  // Display Functions
  function displayDiagnosisResults(results) {
    if (!results.length) return;

    const result = results[0]; // Take first result
    const diagnosisHtml = `
      <div class="diagnosis-card">
        <div class="diagnosis-header">
          <div class="diagnosis-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
            </svg>
          </div>
          <div>
            <h3 class="diagnosis-title">${result.disease}</h3>
            <p class="diagnosis-subtitle">AI Diagnosis Result</p>
          </div>
        </div>

        <div class="symptoms-section">
          <h4 style="margin-bottom: 1rem; font-weight: 600;">Analyzed Symptoms</h4>
          ${results.map(r => `
            <div class="symptom-item">
              <span class="symptom-name">${r.symptom}</span>
              <div class="symptom-meta">
                <span class="confidence-badge ${getConfidenceClass(r.confidence)}">
                  ${(r.confidence * 100).toFixed(1)}% confidence
                </span>
                <span class="severity-badge ${getSeverityClass(r.severity)}">
                  ${getSeverityIcon(r.severity)} Severity ${r.severity}
                </span>
              </div>
            </div>
          `).join('')}
        </div>

        <div class="precautions-section">
          <h4 style="margin-bottom: 1rem; font-weight: 600; display: flex; align-items: center; gap: 0.5rem;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
            Recommended Precautions
          </h4>
          <div class="precautions-list">
            ${result.precautions.map(precaution => `
              <div class="precaution-item">
                <svg class="precaution-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <polyline points="9,11 12,14 22,4"/>
                  <path d="M21,12v7a2,2 0 0,1 -2,2H5a2,2 0 0,1 -2,-2V5a2,2 0 0,1 2,-2h11"/>
                </svg>
                <span>${precaution}</span>
              </div>
            `).join('')}
          </div>
        </div>

        <div style="margin-top: 1.5rem; padding: 1rem; background: #f8fafc; border-radius: 8px; border-left: 4px solid #06b6d4;">
          <p style="margin: 0; color: #0f172a; line-height: 1.6;">
            <strong>About this condition:</strong> ${result.description}
          </p>
        </div>
      </div>
    `;

    $("#diagnosis-results").html(diagnosisHtml).hide().slideDown(600);
    
    // Scroll to results
    $("html, body").animate({
      scrollTop: $("#diagnosis-results").offset().top - 100
    }, 800);
  }

  function getConfidenceClass(confidence) {
    if (confidence >= 0.8) return 'confidence-high';
    if (confidence >= 0.6) return 'confidence-medium';
    return 'confidence-low';
  }

  function getSeverityClass(severity) {
    if (severity >= 6) return 'severity-high';
    if (severity >= 4) return 'severity-medium';
    return 'severity-low';
  }

  function getSeverityIcon(severity) {
    if (severity >= 6) return '🔴';
    if (severity >= 4) return '🟡';
    return '🟢';
  }

  // Message Display Functions
  function appendBotMessage(msg) {
    const messageHtml = `<div class="message bot-message">${msg}</div>`;
    chat.append(messageHtml);
    scrollToBottom();
  }

  function appendUserMessage(msg) {
    const messageHtml = `<div class="message user-message">${msg}</div>`;
    chat.append(messageHtml);
    scrollToBottom();
  }

  function showTyping() {
    typingDiv = $(`
      <div class="message bot-message typing-indicator">
        <span>Meddy is analyzing</span>
        <div class="typing-dots">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
    `);
    chat.append(typingDiv);
    scrollToBottom();
  }

  function hideTyping() {
    if (typingDiv) {
      typingDiv.fadeOut(200, function() {
        $(this).remove();
      });
      typingDiv = null;
    }
  }

  function scrollToBottom() {
    chat.animate({ scrollTop: chat[0].scrollHeight }, 300);
  }

  // Utility Functions
  function setButtonLoading(button, loading) {
    if (loading) {
      button.addClass('loading').prop('disabled', true);
      if (!button.data('original-text')) {
        button.data('original-text', button.html());
      }
      button.html('Processing...');
    } else {
      button.removeClass('loading').prop('disabled', false);
      if (button.data('original-text')) {
        button.html(button.data('original-text'));
      }
    }
  }

  function setProcessing(processing) {
    isProcessing = processing;
    sendBtn.prop('disabled', processing);
    
    if (processing) {
      spinner.show();
    } else {
      spinner.hide();
    }
  }

  function showError(message) {
    // Create temporary error message
    const errorDiv = $(`
      <div class="alert alert-error" style="
        background: #fee2e2; 
        color: #991b1b; 
        padding: 0.75rem 1rem; 
        border-radius: 8px; 
        margin: 1rem 0;
        border: 1px solid #fecaca;
        animation: fadeInUp 0.3s ease;
      ">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 0.5rem;">
          <circle cx="12" cy="12" r="10"/>
          <line x1="15" y1="9" x2="9" y2="15"/>
          <line x1="9" y1="9" x2="15" y2="15"/>
        </svg>
        ${message}
      </div>
    `);
    
    // Insert after user details card
    userDetailsCard.after(errorDiv);
    
    // Remove after 5 seconds
    setTimeout(() => {
      errorDiv.fadeOut(300, function() {
        $(this).remove();
      });
    }, 5000);
  }

  // PDF Download
  function handlePDFDownload() {
    setButtonLoading(downloadBtn, true);
    
    // Simulate download delay for UX
    setTimeout(() => {
      setButtonLoading(downloadBtn, false);
    }, 1000);
  }

  // Chat Window Toggle
  function toggleChatWindow() {
    const chatWindow = $(".chat-window");
    if (chatWindow.length === 0) {
      createChatWindow();
    } else {
      chatWindow.toggle();
    }
  }

  function createChatWindow() {
    const chatWindowHtml = `
      <div class="chat-window">
        <div class="chat-header">
          <h4 style="margin: 0; font-weight: 600;">Chat with Meddy</h4>
          <button class="chat-close" aria-label="Close chat">×</button>
        </div>
        <div class="chat-body" style="flex: 1; padding: 1rem;">
          <div class="message bot-message">Hi! How can I help you today?</div>
        </div>
        <div class="chat-input" style="padding: 1rem; border-top: 1px solid #e2e8f0;">
          <input type="text" placeholder="Type your message..." style="flex: 1; padding: 0.5rem; border: 1px solid #e2e8f0; border-radius: 6px;">
          <button class="btn btn-primary" style="padding: 0.5rem;">Send</button>
        </div>
      </div>
    `;
    
    $("body").append(chatWindowHtml);
    
    // Setup chat window events
    $(".chat-close").on("click", () => $(".chat-window").hide());
  }

  // Voice Recognition
  function setupVoiceRecognition() {
    if (!('webkitSpeechRecognition' in window)) {
      micBtn.prop("disabled", true).attr("title", "Speech recognition not supported in this browser");
      return;
    }

    const recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.lang = "en-US";
    recognition.interimResults = false;

    micBtn.on("click", function () {
      if (isProcessing) return;
      
      recognition.start();
      setButtonLoading(micBtn, true);
      micBtn.html(`
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="3"/>
          <path d="M12 1v6m0 6v6m11-7h-6m-6 0H1"/>
        </svg>
      `);
    });

    recognition.onresult = function (event) {
      const transcript = event.results[0][0].transcript;
      input.val(transcript).trigger('input');
      resetMicButton();
    };

    recognition.onerror = function (event) {
      console.error("Speech recognition error:", event.error);
      showError("Voice recognition failed. Please try typing instead.");
      resetMicButton();
    };

    recognition.onend = function () {
      resetMicButton();
    };

    function resetMicButton() {
      setButtonLoading(micBtn, false);
      micBtn.html(`
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
          <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
          <line x1="12" y1="19" x2="12" y2="23"/>
          <line x1="8" y1="23" x2="16" y2="23"/>
        </svg>
      `);
    }
  }

  // Smooth scrolling for anchor links
  $('a[href^="#"]').on('click', function(e) {
    e.preventDefault();
    const target = $($(this).attr('href'));
    if (target.length) {
      $('html, body').animate({
        scrollTop: target.offset().top - 100
      }, 800);
    }
  });

  // Show chat toggle after user starts assessment
  submitDetailsBtn.on('click', function() {
    setTimeout(() => {
      chatToggle.fadeIn(400);
    }, 1000);
  });
});