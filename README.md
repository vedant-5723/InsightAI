# InsightAI — AI Data Analyst & Dashboard Generator

InsightAI is an AI-powered data analysis application that helps users explore datasets, generate visual dashboards, and ask questions about their data using natural language. It combines a React-based interface, a Python backend, data analysis tools, and configurable AI providers.

The project is designed to make data exploration more accessible while keeping the application runnable locally.

## Features

- **Dataset upload:** Upload CSV and Excel files for analysis.
- **Automated data profiling:** Inspect dataset structure, column types, and missing values.
- **Statistical analysis:** Generate summaries and perform aggregations, grouping, filtering, and top-value analysis.
- **Interactive dashboards:** View key performance indicators and charts generated from uploaded data.
- **AI data analyst:** Ask natural-language questions about datasets.
- **Multiple AI providers:** Supports Ollama for local inference and Gemini as an alternative provider.
- **Error handling:** Handle common analysis and data-processing errors.
- **Automated tests:** Includes backend tests for analysis functionality.

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | React, Vite, JavaScript |
| Backend | Python, FastAPI |
| Data analysis | Python data-processing libraries |
| Local AI | Ollama |
| Alternative AI provider | Gemini API |
| Testing | pytest |
| Version control | Git and GitHub |

## Architecture

```text
User
  |
  v
React + Vite Frontend
  |
  v
Python / FastAPI Backend
  |
  +--> Dataset Processing
  |      +--> Data Profiling
  |      +--> Aggregation and Filtering
  |      +--> Analysis Results
  |
  +--> AI Provider
         +--> Ollama
         +--> Gemini
  |
  v
Dashboard and Natural-Language Insights
```

## Project Structure

```text
InsightAI/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   └── main.py
│   ├── tests/
│   │   └── test_analysis.py
│   ├── test_insightai_providers.py
│   └── requirements.txt
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── sales.csv
├── .gitignore
└── README.md
```

## Prerequisites

Install the following before running the application:

- Python 3.10 or a compatible version supported by the backend dependencies
- Node.js and npm
- Git
- Ollama, if using the local AI provider
- A Gemini API key, if using the Gemini provider

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/vedant-5723/InsightAI.git
cd InsightAI
```

### 2. Set up the Python backend

Open a terminal in the project root and run:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the backend:

```powershell
python -m uvicorn app.main:app --reload
```

Keep this terminal open while using the application.

### 3. Set up the frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite in your terminal, commonly `http://localhost:5173`.

### 4. Configure an AI provider

**Ollama**

Install and start Ollama, then download the model configured for the project. The current development setup uses `qwen2.5:3b`.

```bash
ollama pull qwen2.5:3b
```

Ensure the model is available locally and that the application's provider configuration matches it.

**Gemini**

Select the Gemini provider and configure your own API key using the application's supported configuration. Do not commit real API keys to GitHub or place them in source code.

## Running Tests

From the backend directory, activate the virtual environment and run:

```powershell
python -m pytest -v
```

The latest recorded development test run passed all 15 tests. Run the tests again against your current checkout to verify its status.

To check Python syntax:

```powershell
python -m compileall app
```

To build the frontend:

```powershell
cd ..\frontend
npm run build
```

## Sample Dataset

The repository includes `sales.csv`, a synthetic sales dataset intended for demonstration and testing.

Use it to explore data summaries, product and regional revenue, dashboard charts, and natural-language analysis.

## Data and Security Notes

- Do not commit API keys, passwords, access tokens, or private environment files.
- Use synthetic or appropriately authorized datasets.
- Review `.gitignore` before committing local files.
- The current development application uses in-memory dataset handling; do not assume uploaded datasets persist across backend restarts.
- Run the application locally and verify provider configuration before relying on analysis results.

## Current Status

InsightAI is an actively developed local application. The project includes dataset analysis, AI-assisted question answering, dashboard visualizations, and backend tests.

Public deployment has not been established as part of this setup. Feature availability and compatibility should be verified against the current codebase.

## Future Improvements

- More configurable chart types and dashboard controls
- Improved conversational context for follow-up questions
- Downloadable analysis reports
- Enhanced validation and error messages
- More comprehensive testing and deployment documentation

---

**Built as a practical project combining data analysis, dashboard development, Python backend engineering, and large language model integration.**
