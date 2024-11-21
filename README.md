
# Frontend_Test Project

This project is a chatbot interface that utilizes a local models. The project consists of a frontend (HTML, CSS, and JavaScript) and a backend server built with Node.js and Express.

## Prerequisites

1. **Node.js** (v14 or later)
2. **npm** (Node Package Manager)
3. **OpenAI API Key** – Set this up in a `.env` file (see below).
4. **Assistant ID** – If needed for specific OpenAI configurations.

## Installation

1. **Clone the repository:**


2. **Install dependencies:**
   Navigate to the project root directory and install the necessary packages.
   ```bash
   npm install
   ```
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Models**
   Navigate to the project root directory and install the necessary packages.
   ```bash
   python preprocessing.py
   python training.py
   python inference.py
   ```

## Project Structure

- `index.html`: Main HTML file for the chatbot interface.
- `styles.css`: Styles for the frontend.
- `app.js`: Frontend JavaScript for handling user input and displaying messages.
- `server.js`: Backend server handling requests and communicating with the OpenAI API.
- `backend/preprocessing.py`: Script for preparing data and generating embeddings.
- `backend/training.py`: Script for training the model and saving embeddings.
- `backend/inference.py`: Script for loading embeddings and generating responses based on user queries.

## Running the Project

1. **Start the server:**
   Run the following command in the root directory:
   ```bash
   node server.js
   ```

2. **Access the app:**
   Open a browser and go to [http://localhost:3000](http://localhost:3000) to view the chatbot interface.

## Dependencies

The backend requires the following Node.js packages:

- `express`: Web framework for Node.js.
- `cors`: Middleware to allow cross-origin requests.

Install these packages by running:
```bash
npm install express cors 
```

## Notes

- Ensure the `.env` file is added to `.gitignore` to prevent accidental sharing of sensitive information.
- You might need to adjust API configurations in `server.js` to suit your API settings.

## Acknowledgments

This project uses the OpenAI API for generating chatbot responses.
