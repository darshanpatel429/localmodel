document.addEventListener('DOMContentLoaded', () => {
    const userInput = document.getElementById('userInput');
    const messagesDiv = document.getElementById('messages');

    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        // Create and display user message
        const userMessageDiv = document.createElement('div');
        userMessageDiv.className = 'message user-message';
        userMessageDiv.textContent = message;
        messagesDiv.appendChild(userMessageDiv);

        // Clear input
        userInput.value = '';

        try {
            // Show loading state
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'message loading-message';
            loadingDiv.textContent = 'Thinking...';
            messagesDiv.appendChild(loadingDiv);

            // Send to backend
            const response = await fetch('http://localhost:3000/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: message })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            // Remove loading message
            messagesDiv.removeChild(loadingDiv);

            // Display bot response with formatting
            const botMessageDiv = document.createElement('div');
            botMessageDiv.className = 'message bot-message';
            botMessageDiv.innerHTML = window.chatUtils.formatMessage(data.response);
            messagesDiv.appendChild(botMessageDiv);

            // Scroll to bottom
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

        } catch (error) {
            window.chatUtils.handleError(error, messagesDiv);
        }
    }

    // Add click event listener to send button
    document.querySelector('button').addEventListener('click', sendMessage);

    // Add enter key event listener to input field
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            sendMessage();
        }
    });
}); 