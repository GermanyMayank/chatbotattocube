document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatContainer = document.getElementById('chat-container');
    const submitBtn = document.getElementById('submit-btn');
    const clearChatBtn = document.getElementById('clear-chat');
    const typingIndicator = document.getElementById('typing-indicator');

    // --- State Management ---
    let conversationHistory = [];

    // --- Core Functions ---

    /**
     * Renders a message to the chat container using an HTML template.
     * @param {string} sender - Who sent the message ('user' or 'bot').
     * @param {string} text - The message content.
     */
    const renderMessage = (sender, text) => {
        const isUser = sender === 'user';
        const sourceSplit = '\n\n📌 **Sources:**\n';
        const [mainResponse, sources] = text.split(sourceSplit);

        const messageTemplate = `
            <div class="flex gap-3 ${isUser ? 'flex-row-reverse' : ''} message-wrapper">
                <div class="w-8 h-8 rounded-full ${isUser ? 'bg-gray-500' : 'bg-[var(--attocube-orange)]'} flex items-center justify-center text-white font-bold text-sm shrink-0">
                    ${isUser ? 'U' : 'A'}
                </div>
                <div class="chat-bubble ${isUser ? 'chat-bubble-user' : 'chat-bubble-bot'}">
                    <div class="bot-response">${marked.parse(mainResponse.trim())}</div>
                    ${sources && sources.trim() ? `
                        <details class="sources-details">
                            <summary>Show Sources ▼</summary>
                            <div class="sources-content">${sources.trim()}</div>
                        </details>
                    ` : ''}
                </div>
            </div>
        `;

        chatContainer.insertAdjacentHTML('beforeend', messageTemplate);
        chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: 'smooth' });
    };

    /**
     * Toggles the UI state between loading and idle.
     * @param {boolean} isLoading - Whether the app is in a loading state.
     */
    const setLoadingState = (isLoading) => {
        submitBtn.disabled = isLoading;
        typingIndicator.classList.toggle('hidden', !isLoading);
        if (!isLoading) {
            userInput.focus();
        }
    };

    /**
     * Sends the user's message to the backend API.
     * @param {string} message - The user's input message.
     * @returns {Promise<string>} - A promise that resolves to the bot's response text.
     */
    const fetchBotResponse = async (message) => {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_input: message })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        return data.response;
    };

    /**
     * Handles the chat form submission.
     * @param {Event} e - The form submission event.
     */
    const handleFormSubmit = async (e) => {
        e.preventDefault();
        const message = userInput.value.trim();
        if (!message) return;

        // Update UI and history
        renderMessage('user', message);
        conversationHistory.push({ role: 'user', content: message });
        userInput.value = '';
        setLoadingState(true);

        try {
            const botResponse = await fetchBotResponse(message);
            renderMessage('bot', botResponse);
            conversationHistory.push({ role: 'bot', content: botResponse });
        } catch (err) {
            console.error(err);
            renderMessage('bot', 'Sorry, something went wrong. Please check the console for details.');
        } finally {
            setLoadingState(false);
        }
    };

    /**
     * Initializes the chat, loading history or showing a welcome message.
     */
    const initializeChat = () => {
        // Feature disabled: No longer loads history from localStorage
        conversationHistory = [];
        const welcomeMsg = 'Hello! How can I help you with Attocube products today?';
        renderMessage('bot', welcomeMsg);
        conversationHistory.push({ role: 'bot', content: welcomeMsg });
    };
    
    /**
     * Clears the chat UI and resets the conversation state.
     */
    const clearChat = () => {
        chatContainer.innerHTML = '';
        initializeChat();
    };

    // --- Event Listeners ---
    chatForm.addEventListener('submit', handleFormSubmit);
    clearChatBtn.addEventListener('click', clearChat);

    // --- Initial Load ---
    initializeChat();
});