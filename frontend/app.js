document.getElementById('userInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Updated linkify function to handle parentheses and formatting
function linkify(text) {
    // Handle line breaks, lists, and URLs
    text = text.replace(/\n/g, '<br>'); // Line breaks
    text = text.replace(/(\d+\.)\s/g, '<br>$1 '); // Numbered lists
    text = text.replace(/•\s/g, '<br>• ').replace(/\*\s/g, '<br>* '); // Bullet points
    
    // URL detection with improved regex
    const urlRegex = /(https?:\/\/[^\s)]+)/g;
    text = text.replace(urlRegex, (url) => {
        const cleanUrl = url.replace(/[.,;:)]$/, '');
        return `<a href="${cleanUrl}" target="_blank" rel="noopener noreferrer">${cleanUrl}</a>`;
    });

    // Remove initial <br> if it exists
    return text.replace(/^<br>/, '');
}

async function sendMessage() {
    const userInput = document.getElementById('userInput').value.trim();
    if (!userInput) return;

    const messagesDiv = document.getElementById('messages');
    const button = document.querySelector('button');
    const input = document.getElementById('userInput');

    // Display user message
    const userMessageDiv = document.createElement('div');
    userMessageDiv.classList.add('message', 'user-message');
    userMessageDiv.innerText = userInput;
    messagesDiv.appendChild(userMessageDiv);

    // Clear input and disable controls
    input.value = '';
    input.disabled = true;
    button.disabled = true;
    button.innerText = 'Thinking...';

    // Add loading message
    const loadingDiv = document.createElement('div');
    loadingDiv.classList.add('message', 'loading-message');
    loadingDiv.innerHTML = 'Assistant is typing<span class="typing-indicator"></span>';
    messagesDiv.appendChild(loadingDiv);

    try {
        const response = await fetch('http://localhost:3000/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: userInput })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Remove loading message
        loadingDiv.remove();

        if (data.error) {
            throw new Error(data.error);
        }

        // Display bot response with clickable links
        const botMessageDiv = document.createElement('div');
        botMessageDiv.classList.add('message', 'bot-message');
        botMessageDiv.innerHTML = linkify(data.response);
        messagesDiv.appendChild(botMessageDiv);

    } catch (error) {
        // Use handleError function for error messages
        window.chatUtils.handleError(error, messagesDiv);
    } finally {
        // Re-enable controls
        input.disabled = false;
        button.disabled = false;
        button.innerText = 'Send';
        input.focus();
        
        // Auto-scroll to bottom
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }
}

// Expose linkify and handleError as utilities for consistency
window.chatUtils = {
    formatMessage: linkify,
    handleError: (error, messagesDiv) => {
        console.error('Error:', error);
        const errorDiv = document.createElement('div');
        errorDiv.classList.add('message', 'error-message');
        errorDiv.innerText = error.message || 'Sorry, something went wrong. Please try again.';
        messagesDiv.appendChild(errorDiv);
    }
};
