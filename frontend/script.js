/**
 * TPQ AI Assistant - Frontend Script
 *
 * Handles chat interactions, message rendering, loading states,
 * error handling, and auto-scroll behavior.
 */

// ============================================================
// DOM Elements
// ============================================================
const chatMessages = document.getElementById('chat-messages');
const chatForm = document.getElementById('chat-form');
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');
const charCount = document.getElementById('char-count');
const welcomeContainer = document.getElementById('welcome-container');

// ============================================================
// State
// ============================================================
let isLoading = false;

// ============================================================
// Input Handling
// ============================================================

/**
 * Auto-resize textarea based on content.
 */
messageInput.addEventListener('input', function () {
    // Reset height to auto to properly calculate scroll height
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';

    // Update character count
    const count = this.value.length;
    charCount.textContent = `${count} / 1000`;

    // Enable/disable send button
    sendButton.disabled = this.value.trim().length === 0 || isLoading;
});

/**
 * Handle Enter key to send message (Shift+Enter for new line).
 */
messageInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!sendButton.disabled) {
            chatForm.dispatchEvent(new Event('submit'));
        }
    }
});

// ============================================================
// Message Rendering
// ============================================================

/**
 * Get current time formatted as HH:MM.
 */
function getCurrentTime() {
    const now = new Date();
    return now.toLocaleTimeString('id-ID', {
        hour: '2-digit',
        minute: '2-digit',
    });
}

/**
 * Add a message bubble to the chat.
 *
 * @param {string} content - The message text.
 * @param {'user'|'assistant'|'error'} role - The message role.
 */
function addMessage(content, role) {
    // Hide welcome container on first message
    if (welcomeContainer) {
        welcomeContainer.style.display = 'none';
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const avatarText = role === 'user' ? '👤' : role === 'error' ? '⚠️' : '🤖';

    messageDiv.innerHTML = `
        <div class="message-avatar">${avatarText}</div>
        <div>
            <div class="message-content">${escapeHtml(content)}</div>
            <div class="message-time">${getCurrentTime()}</div>
        </div>
    `;

    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

/**
 * Escape HTML characters to prevent XSS.
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Show typing indicator.
 *
 * @returns {HTMLElement} The typing indicator element (for removal).
 */
function showTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    indicator.id = 'typing-indicator';

    indicator.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="typing-dots">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;

    chatMessages.appendChild(indicator);
    scrollToBottom();
    return indicator;
}

/**
 * Remove typing indicator.
 */
function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) {
        indicator.remove();
    }
}

/**
 * Scroll chat to the bottom.
 */
function scrollToBottom() {
    requestAnimationFrame(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    });
}

// ============================================================
// API Communication
// ============================================================

/**
 * Send a message to the API and display the response.
 *
 * @param {string} message - The user's message.
 */
async function sendMessage(message) {
    if (isLoading) return;
    isLoading = true;

    // Add user message
    addMessage(message, 'user');

    // Show typing indicator
    showTypingIndicator();

    // Disable input
    sendButton.disabled = true;
    messageInput.disabled = true;

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message }),
        });

        removeTypingIndicator();

        if (!response.ok) {
            let errorMessage = 'Terjadi kesalahan pada server.';

            try {
                const errorData = await response.json();
                errorMessage = errorData.detail || errorMessage;
            } catch {
                // Response is not JSON
            }

            if (response.status === 503) {
                errorMessage = 'Model belum tersedia. Pastikan model sudah di-load.';
            } else if (response.status === 422) {
                errorMessage = 'Pesan tidak valid. Pastikan pesan tidak kosong dan tidak melebihi 1000 karakter.';
            }

            addMessage(errorMessage, 'error');
        } else {
            const data = await response.json();
            addMessage(data.response, 'assistant');
        }
    } catch (error) {
        removeTypingIndicator();

        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            addMessage(
                'Tidak dapat terhubung ke server. Pastikan server API sedang berjalan.',
                'error'
            );
        } else {
            addMessage(
                `Terjadi kesalahan: ${error.message}`,
                'error'
            );
        }
    } finally {
        isLoading = false;
        messageInput.disabled = false;
        messageInput.focus();
        sendButton.disabled = messageInput.value.trim().length === 0;
    }
}

// ============================================================
// Event Handlers
// ============================================================

/**
 * Handle form submission.
 */
function handleSubmit(event) {
    event.preventDefault();

    const message = messageInput.value.trim();
    if (!message || isLoading) return;

    // Clear input
    messageInput.value = '';
    messageInput.style.height = 'auto';
    charCount.textContent = '0 / 1000';
    sendButton.disabled = true;

    sendMessage(message);
}

/**
 * Handle suggestion chip click.
 */
function sendSuggestion(message) {
    if (isLoading) return;
    messageInput.value = '';
    sendMessage(message);
}

// ============================================================
// Initialization
// ============================================================

// Focus input on load
window.addEventListener('load', () => {
    messageInput.focus();
});
