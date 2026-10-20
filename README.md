# InsightAI — AI Data Analyst & Dashboard Generator

InsightAI is an AI-powered data analytics application that helps users explore datasets, ask questions in natural language, and generate interactive dashboards. It combines AI-assisted analysis with backend data-processing tools to turn raw data into actionable insights.

## Key Features

- **Dataset Upload:** Import CSV and Excel datasets.
- **Data Profiling:** Inspect data types, missing values, distinct values, and dataset dimensions.
- **AI Data Analyst:** Ask natural-language questions about uploaded datasets.
- **Tool-Assisted Analysis:** Perform summaries, aggregations, grouping, top-value analysis, and filtered calculations.
- **Automated Dashboard Generation:** Generate KPI cards, visualizations, and key findings from uploaded data.
- **Interactive Charts:** Switch between supported visualization types.
- **Multiple AI Providers:** Use a local Ollama model or Google Gemini with your own API key.
- **Responsive Interface:** Navigate between Overview, Datasets, and AI Analyst.

## Tech Stack

| Component | Technologies |
|---|---|
| Frontend | React, Vite, JavaScript, CSS |
| Backend | Python, FastAPI |
| Data Processing | Pandas |
| Visualization | Recharts |
| Icons | Lucide React |
| AI Integration | Ollama, Google Gemini |
| Testing | pytest |

## Architecture

```text
User
 |
 v
React + Vite Frontend
 |
 v
FastAPI Backend
 |
 +-- Dataset Profiling
 |
 +-- Data Analysis Tools
 |
 +-- Dashboard Generation
 |
 +-- Ollama / Google Gemini
```

The backend processes datasets and performs numerical calculations. AI models assist with interpreting user questions and planning dashboard content, while analysis tools provide data-grounded results.

## Project Structure

```text
InsightAI/
├── backend/
│   ├── app/
│   │   └── main.py
│   └── tests/
│       └── test_analysis.py
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   └── index.css
    └── package.json
```

## Getting Started

### Prerequisites

- Python 3.10 or a compatible version
- Node.js and npm
- Git
- Ollama for local model execution, or a Google Gemini API key

### 1. Clone the repository

```bash
git clone https://github.com/vedant-5723/InsightAI.git
cd InsightAI
```

### 2. Start the backend

Open a terminal:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies using the dependency file included in the repository. If it is named `requirements.txt`:

```powershell
pip install -r requirements.txt
```

If the FastAPI application is exposed as `app` in `app/main.py`, start it with:

```powershell
uvicorn app.main:app --reload
```

### 3. Start the frontend

Open a second terminal from the repository root:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL displayed by Vite, usually `http://localhost:5173`.

### 4. Configure an AI provider

**Ollama:** Ensure Ollama is running and that the model configured in the backend is installed.

**Google Gemini:** Select Gemini in the application and provide your API key. Never commit API keys to GitHub.

### 5. Run backend tests

From the backend environment:

```powershell
pytest
```

## Example Workflow

1. Upload a CSV or Excel dataset.
2. Review the dataset profile and preview.
3. Ask the AI Analyst a question about the data.
4. Generate a dashboard.
5. Explore KPIs, charts, and key findings.

### Example Questions

- What is the total revenue?
- Which product generates the highest revenue?
- Compare revenue across regions.
- Identify missing values in the dataset.
- Summarize the important patterns in the data.

## Testing

The project includes backend analysis tests. Validate the application after code changes by running the test suite and checking the complete upload, analysis, and dashboard-generation workflow.

## Future Improvements

- Dashboard and analysis exports
- Improved dataset session management
- Automated frontend and API integration tests
- Deployment configuration and production security hardening

## Author

**Vedant Nikam**

GitHub: https://github.com/vedant-5723

---

*InsightAI — turning data into decisions through AI-assisted analytics.*
