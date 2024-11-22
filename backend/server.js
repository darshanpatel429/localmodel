const { spawn } = require("child_process");
const path = require("path");
const express = require("express");
const cors = require("cors");
const sqlite3 = require("sqlite3");
const session = require("express-session");

const app = express();
app.use(cors());
app.use(express.json());

// Session management for logging
app.use(
    session({
        secret: "123",
        resave: false,
        saveUninitialized: true,
    })
);

// Serve static files from the frontend directory
app.use(express.static(path.join(__dirname, "../frontend")));

// Configure SQLite database
const db = new sqlite3.Database("chat.logs");
db.serialize(() => {
    db.run(`CREATE TABLE IF NOT EXISTS Logs (
        SessionID TEXT, 
        dt DATETIME DEFAULT CURRENT_TIMESTAMP, 
        UserQuery TEXT, 
        Response TEXT
    )`);
});

// Generate response using Python (local_rag.py)
async function generateResponse(userInput) {
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn("python3", [path.join(__dirname, "local_rag.py")]);

        // Send user input to the Python script
        pythonProcess.stdin.write(JSON.stringify({ query: userInput }));
        pythonProcess.stdin.end();

        let result = "";
        let error = "";

        // Capture Python script's stdout
        pythonProcess.stdout.on("data", (data) => {
            result += data.toString();
        });

        // Capture Python script's stderr
        pythonProcess.stderr.on("data", (data) => {
            error += data.toString();
        });

        // Handle process close
        pythonProcess.on("close", (code) => {
            if (code !== 0) {
                console.error(`Python script exited with code ${code}.`);
                console.error("Python Error Output:", error);
                reject(new Error("Python script failed. Check logs for details."));
                return;
            }

            try {
                const jsonResponse = JSON.parse(result);
                if (jsonResponse.error) {
                    reject(new Error(jsonResponse.error));
                } else {
                    resolve(jsonResponse.response);
                }
            } catch (parseError) {
                console.error("Failed to parse Python response:", parseError);
                reject(new Error("Invalid response from Python script."));
            }
        });
    });
}

// Chat endpoint
app.post("/api/chat", async (req, res) => {
    try {
        const userInput = req.body.message;
        if (!userInput || typeof userInput !== "string") {
            return res.status(400).json({ error: "Invalid input message" });
        }

        const assistantResponse = await generateResponse(userInput);

        // Log the chat to SQLite
        const stmt = db.prepare(
            "INSERT INTO Logs (SessionID, UserQuery, Response) VALUES (?, ?, ?)"
        );
        stmt.run(req.sessionID, userInput, assistantResponse, (err) => {
            if (err) {
                console.error("Failed to log to database:", err);
            }
        });

        res.json({ response: assistantResponse });
    } catch (error) {
        console.error("Error generating response:", error.message);
        res.status(500).json({ error: "Internal Server Error" });
    }
});

// Logs endpoint
app.get("/api/logs", (req, res) => {
    db.all("SELECT * FROM Logs ORDER BY dt DESC", (err, rows) => {
        if (err) {
            console.error("Error fetching logs:", err);
            return res.status(500).json({ error: "Failed to fetch logs" });
        }
        res.json({ logs: rows });
    });
});

app.get("/api/logs/:sessionId", (req, res) => {
    const { sessionId } = req.params;
    db.all(
        "SELECT * FROM Logs WHERE SessionID = ? ORDER BY dt DESC",
        [sessionId],
        (err, rows) => {
            if (err) {
                console.error("Error fetching logs:", err);
                return res.status(500).json({ error: "Failed to fetch logs" });
            }
            res.json({ logs: rows });
        }
    );
});

app.delete("/api/deleteAllLogs", (req, res) => {
    db.run("DELETE FROM Logs", function (err) {
        if (err) {
            console.error("Error deleting logs:", err);
            return res.status(500).json({ error: "Failed to delete logs" });
        }
        res.status(200).json({ message: "All logs deleted" });
    });
});

// Start the server
const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});