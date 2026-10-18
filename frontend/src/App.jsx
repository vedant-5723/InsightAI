import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useEffect, useState } from "react";
import {
  Bot,
  Database,
  FileSpreadsheet,
  LayoutDashboard,
  Send,
  ShieldCheck,
  Sparkles,
  Upload,
  BarChart3,
  RefreshCw,
} from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import "./index.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

const CHART_COLORS = [
  "#5471e8",
  "#16a085",
  "#f39c12",
  "#e76f51",
  "#8e63ce",
  "#2d9cdb",
  "#d35400",
  "#7f8c8d",
];

async function readResponse(response) {
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(
      data.detail || data.message || `Request failed (${response.status})`
    );
  }

  return data;
}

export default function App() {
  const [apiStatus, setApiStatus] = useState("Checking...");
  const [message, setMessage] = useState("");
  const [reply, setReply] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const [file, setFile] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [preview, setPreview] = useState(null);
  const [profile, setProfile] = useState(null);
  const [uploading, setUploading] = useState(false);

  const [dashboard, setDashboard] = useState(null);
  const [dashboardLoading, setDashboardLoading] = useState(false);
  const [activeView, setActiveView] = useState("overview");
  const [chartOverrides, setChartOverrides] = useState({});
  const [provider, setProvider] = useState("ollama");
  const [geminiKey, setGeminiKey] = useState("");
  const [geminiModel, setGeminiModel] = useState("gemini-3.8-flash");

  useEffect(() => {
    let cancelled = false;

    fetch(`${API}/api/health`)
      .then((response) => {
        if (!response.ok) {
          throw new Error("API unavailable");
        }
        return response.json();
      })
      .then(() => {
        if (!cancelled) setApiStatus("Connected");
      })
      .catch(() => {
        if (!cancelled) setApiStatus("Offline");
      });

    return () => {
      cancelled = true;
    };
  }, []);

  async function sendMessage(event) {
    event.preventDefault();

    if (!message.trim() || busy) return;

    setBusy(true);
    setError("");
    setReply("");

    try {
      const endpoint = dataset
        ? `${API}/api/agent/analyze`
        : `${API}/api/chat`;

      const payload = dataset
        ? {
            datasetId: dataset.datasetId,
            question: message.trim(),
            provider,
            ...(provider === "gemini" ? { apiKey: geminiKey, model: geminiModel } : {}),
          }
        : { message: message.trim(), provider, ...(provider === "gemini" ? { apiKey: geminiKey, model: geminiModel } : {}) };

      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await readResponse(response);

      if (dataset) {
        const toolSummary = (data.toolCalls || [])
          .map((item) => `${item.tool}: ${item.status}`)
          .join("\n");

        setReply(
          `${data.answer || "No answer received."}\n\nTools executed:\n${
            toolSummary || "No tools were called."
          }`
        );
      } else {
        setReply(data.reply || "No response received.");
      }
    } catch (err) {
      setError(err.message || "Could not complete the request.");
    } finally {
      setBusy(false);
    }
  }

  async function uploadDataset() {
    if (!file || uploading) return;

    setUploading(true);
    setError("");
    setReply("");
    setDataset(null);
    setPreview(null);
    setProfile(null);
    setDashboard(null);
    setChartOverrides({});

    try {
      const body = new FormData();
      body.append("file", file);

      const response = await fetch(`${API}/api/datasets`, {
        method: "POST",
        body,
      });

      const data = await readResponse(response);

      if (!data.datasetId) {
        throw new Error("The backend did not return a dataset ID.");
      }

      const [previewResponse, profileResponse] = await Promise.all([
        fetch(`${API}/api/datasets/${data.datasetId}/preview`),
        fetch(`${API}/api/datasets/${data.datasetId}/profile`),
      ]);

      const [previewData, profileData] = await Promise.all([
        readResponse(previewResponse),
        readResponse(profileResponse),
      ]);

      setDataset(data);
      setPreview(previewData);
      setProfile(profileData);
      setMessage("");
    } catch (err) {
      setError(err.message || "Could not upload dataset.");
    } finally {
      setUploading(false);
    }
  }

  
  async function loadDashboard() {
    if (!dataset?.datasetId || dashboardLoading) return;

    setDashboardLoading(true);
    setError("");

    try {
      const response = await fetch(
        `${API}/api/datasets/${dataset.datasetId}/dashboard`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            provider,
            ...(provider === "gemini"
              ? { apiKey: geminiKey, model: geminiModel }
              : {}),
          }),
        }
      );

      const data = await readResponse(response);
      setDashboard(data);
    } catch (err) {
      setError(err.message || "Could not generate dashboard.");
    } finally {
      setDashboardLoading(false);
    }
  }


  function handleFileChange(event) {
    setFile(event.target.files?.[0] || null);
    setDataset(null);
    setPreview(null);
    setProfile(null);
    setDashboard(null);
    setChartOverrides({});
    setReply("");
    setError("");
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Sparkles size={21} />
          </div>
          <div>
            <strong>InsightAI</strong>
            <span>Data Intelligence</span>
          </div>
        </div>

        <p className="nav-label">WORKSPACE</p>

        <button type="button" className={`nav-item ${activeView === "overview" ? "active" : ""}`} onClick={() => setActiveView("overview")} aria-current={activeView === "overview" ? "page" : undefined}>
          <LayoutDashboard size={18} /> Overview
        </button>
        <button type="button" className={`nav-item ${activeView === "datasets" ? "active" : ""}`} onClick={() => setActiveView("datasets")} aria-current={activeView === "datasets" ? "page" : undefined}>
          <Database size={18} /> Datasets
        </button>
        <button type="button" className={`nav-item ${activeView === "analyst" ? "active" : ""}`} onClick={() => setActiveView("analyst")} aria-current={activeView === "analyst" ? "page" : undefined}>
          <Bot size={18} /> AI Analyst
        </button>

        <div className="sidebar-bottom">
          <ShieldCheck size={17} />
          <span>Local-first AI foundation</span>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <span>Workspace / {activeView === "overview" ? "Overview" : activeView === "datasets" ? "Datasets" : "AI Analyst"}</span>

          <div
            className={`status ${
              apiStatus === "Connected" ? "online" : ""
            }`}
          >
            <span className="status-dot" />
            API {apiStatus}
          </div>
        </header>

        <section className="welcome">
          <span className="eyebrow">YOUR AI ANALYTICS WORKSPACE</span>
          <h1>
            Turn your data into <span>decisions.</span>
          </h1>
          <p>
            Explore datasets, discover patterns, and build dashboards
            with an AI analyst designed to use real analysis tools.
          </p>
        </section>

        <section className="overview-grid" style={{ display: activeView === "overview" ? undefined : "none" }}>
          <article className="stat-card">
            <Database size={20} />
            <span>Datasets</span>
            <strong>{dataset ? 1 : 0}</strong>
            <small>
              {dataset ? "Dataset loaded" : "Ready to get started"}
            </small>
          </article>

          <article className="stat-card">
            <Bot size={20} />
            <span>Agent status</span>
            <strong>{dataset ? "Ready" : "Foundation"}</strong>
            <small>
              {dataset
                ? "Dataset-aware analysis available"
                : "Upload data to begin"}
            </small>
          </article>

          <article className="stat-card">
            <LayoutDashboard size={20} />
            <span>Dashboards</span>
            <strong>{dashboard ? 1 : 0}</strong>
            <small>
              {dashboard
                ? "Dashboard generated"
                : "No dashboards generated yet"}
            </small>
          </article>
        </section>

        {dataset && (
          <section className="dashboard-section" style={{ display: activeView === "overview" ? undefined : "none" }}>
            <div className="dashboard-heading">
              <div>
                <span className="eyebrow">DATA-DRIVEN ANALYSIS</span>
                <h2>Dashboard Generator</h2>
                <p>
                  Generate KPIs, charts, and findings from {dataset.filename}.
                </p>
              </div>

              <button
                type="button"
                onClick={loadDashboard}
                disabled={dashboardLoading}
              >
                {dashboardLoading ? (
                  "Generating..."
                ) : dashboard ? (
                  <>
                    <RefreshCw size={15} />
                    Refresh dashboard
                  </>
                ) : (
                  <>
                    <BarChart3 size={16} />
                    Generate dashboard
                  </>
                )}
              </button>
            </div>

            {dashboard && (
              <>
                <div className="dashboard-kpis">
                  {(dashboard.kpis || []).map((kpi, index) => (
                    <article
                      className="stat-card"
                      key={`${kpi.label}-${index}`}
                    >
                      <span>{kpi.label}</span>
                      <strong>
                        {typeof kpi.value === "number"
                          ? kpi.value.toLocaleString(undefined, {
                              maximumFractionDigits: 2,
                            })
                          : String(kpi.value ?? "—")}
                      </strong>
                    </article>
                  ))}
                </div>

                <div className="dashboard-charts">
                  {(dashboard.charts || []).map((chart, index) => {
                    const chartData = (chart.data || []).map((item) => ({
                      ...item,
                      category: String(item.category ?? item.name ?? "Other"),
                      value: Number(item.value) || 0,
                    }));

                    const chartType = chartOverrides[index] || String(chart.type || "bar").toLowerCase();

                    return (
                      <article
                        className="panel chart-card"
                        key={`${chart.title || "Chart"}-${index}`}
                      >
                        <div className="chart-title-row">
                          <h3>{chart.title || "Data chart"}</h3>
                          <label className="chart-type-control">
                            <span>Chart</span>
                            <select aria-label={`Chart type for ${chart.title || "chart"}`} value={chartType} onChange={(event) => setChartOverrides((current) => ({ ...current, [index]: event.target.value }))}>
                              <option value="bar">Bar</option>
                              <option value="horizontal">Horizontal bar</option>
                              <option value="line">Line</option>
                              <option value="area">Area</option>
                              <option value="pie">Pie</option>
                              <option value="donut">Donut</option>
                            </select>
                          </label>
                        </div>

                        {chartData.length === 0 ? (
                          <p>No chart data available.</p>
                        ) : (
                          <div className="chart-container">
                            <ResponsiveContainer width="100%" height="100%">
                              {chartType === "pie" || chartType === "donut" ? (
                                <PieChart>
                                  <Pie
                                    data={chartData}
                                    dataKey="value"
                                    nameKey="category"
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={chartType === "donut" ? 58 : 0}
                                    outerRadius={100}
                                    label={({ name, percent }) =>
                                      `${name}: ${(percent * 100).toFixed(0)}%`
                                    }
                                  >
                                    {chartData.map((entry, i) => (
                                      <Cell
                                        key={`${entry.category}-${i}`}
                                        fill={CHART_COLORS[i % CHART_COLORS.length]}
                                      />
                                    ))}
                                  </Pie>
                                  <Tooltip />
                                  <Legend />
                                </PieChart>
                              ) : chartType === "line" ? (
                                <LineChart data={chartData}>
                                  <CartesianGrid strokeDasharray="3 3" />
                                  <XAxis dataKey="category" />
                                  <YAxis />
                                  <Tooltip />
                                  <Legend />
                                  <Line
                                    type="monotone"
                                    dataKey="value"
                                    name={chart.metric || chart.title || "Value"}
                                    stroke="#5471e8"
                                    strokeWidth={3}
                                    dot={{ r: 4 }}
                                  />
                                </LineChart>
                              ) : chartType === "area" ? (
                                <AreaChart data={chartData}>
                                  <CartesianGrid strokeDasharray="3 3" />
                                  <XAxis dataKey="category" />
                                  <YAxis />
                                  <Tooltip />
                                  <Legend />
                                  <Area type="monotone" dataKey="value" name={chart.metric || chart.title || "Value"} stroke="#5471e8" fill="#5471e8" fillOpacity={0.18} />
                                </AreaChart>
                              ) : (
                                <BarChart data={chartData} layout={chartType === "horizontal" ? "vertical" : "horizontal"}>
                                  <CartesianGrid strokeDasharray="3 3" />
                                  {chartType === "horizontal" ? <><XAxis type="number" /><YAxis type="category" dataKey="category" width={100} /></> : <><XAxis dataKey="category" /><YAxis /></>}
                                  <Tooltip />
                                  <Legend />
                                  <Bar dataKey="value" name={chart.metric || chart.title || "Value"} fill="#5471e8" radius={[5, 5, 0, 0]} />
                                </BarChart>
                              )}
                            </ResponsiveContainer>
                          </div>
                        )}
                      </article>
                    );
                  })}
                </div>

                <article className="panel insights-card">
                  <h3>Key findings</h3>
                  {dashboard.insights?.length > 0 ? (
                    <ul>
                      {dashboard.insights.map((insight, index) => (
                        <li key={index}>{insight}</li>
                      ))}
                    </ul>
                  ) : (
                    <p>No numeric insights are available for this dataset.</p>
                  )}
                </article>

                {dashboard.planner && (
                  <p className="planner-status">
                    Dashboard planner: {dashboard.planner}
                  </p>
                )}
              </>
            )}
          </section>
        )}

        <section className={`workspace-grid view-${activeView}`}>
          <article className="panel upload-panel">
            <div className="panel-heading">
              <div>
                <h2>Start with your data</h2>
                <p>Upload a CSV or Excel file to analyze your data.</p>
              </div>
              <FileSpreadsheet size={22} />
            </div>

            <div className="upload-placeholder">
              <div className="upload-icon">
                <Upload size={24} />
              </div>

              <strong>Upload your dataset</strong>
              <span>CSV or Excel · Maximum 10 MB</span>

              <input
                type="file"
                accept=".csv,.xlsx,.xls"
                onChange={handleFileChange}
              />

              <button
                type="button"
                disabled={!file || uploading}
                onClick={uploadDataset}
              >
                {uploading ? "Processing..." : "Upload and analyze"}
              </button>
            </div>

            {dataset && profile && preview && (
              <div className="dataset-results">
                <h3>{dataset.filename}</h3>

                <div className="dataset-metrics">
                  <div>
                    <strong>
                      {Number(profile.rowCount || 0).toLocaleString()}
                    </strong>
                    <span>Rows</span>
                  </div>
                  <div>
                    <strong>{profile.columnCount ?? 0}</strong>
                    <span>Columns</span>
                  </div>
                  <div>
                    <strong>
                      {Number(profile.totalMissing || 0).toLocaleString()}
                    </strong>
                    <span>Missing values</span>
                  </div>
                </div>

                <h3>Column profile</h3>
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Column</th>
                        <th>Type</th>
                        <th>Missing</th>
                        <th>Distinct</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(profile.columns || []).map((column) => (
                        <tr key={column.name}>
                          <td>{column.name}</td>
                          <td>{column.dtype}</td>
                          <td>{column.missing}</td>
                          <td>{column.distinct}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <h3>Data preview</h3>
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        {(preview.columns || []).map((column) => (
                          <th key={column}>{column}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {(preview.rows || []).map((row, index) => (
                        <tr key={index}>
                          {(preview.columns || []).map((column) => (
                            <td key={column}>
                              {row[column] == null ? "—" : String(row[column])}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </article>

          <article className="panel analyst-panel">
            <div className="panel-heading">
              <div>
                <h2>Meet your AI analyst</h2>
                <p>
                  {dataset
                    ? "Ask questions about your uploaded dataset."
                    : "Test the connection to your local AI model."}
                </p>
              </div>
              <Bot size={22} />
            </div>

            <div className="provider-settings">
              <label htmlFor="ai-provider">AI provider</label>
              <select id="ai-provider" value={provider} onChange={(event) => setProvider(event.target.value)}>
                <option value="ollama">Ollama (local)</option>
                <option value="gemini">Google Gemini (your API key)</option>
              </select>
              {provider === "gemini" && (
                <>
                  <label htmlFor="gemini-model">Gemini model</label>
                  <select id="gemini-model" value={geminiModel} onChange={(event) => setGeminiModel(event.target.value)}>
                    <option value="gemini-3.8-flash">Gemini 3.8 Flash</option>
                    <option value="gemini-2.5-pro">Gemini 2.5 Pro</option>
                  </select>
                  <label htmlFor="gemini-key">Gemini API key</label>
                  <input id="gemini-key" type="password" autoComplete="off" value={geminiKey} onChange={(event) => setGeminiKey(event.target.value)} placeholder="Paste your Gemini API key" />
                  <small>Your key is sent to this backend for the request and is not saved by this UI.</small>
                </>
              )}
            </div>

            <form onSubmit={sendMessage} className="chat-form">
              <textarea
                value={message}
                onChange={(event) => setMessage(event.target.value)}
                placeholder={
                  dataset
                    ? "Ask about revenue, products, regions..."
                    : "Ask a question about data analysis..."
                }
                rows={4}
                maxLength={2000}
              />

              <button type="submit" disabled={busy || !message.trim()}>
                {busy ? "Thinking..." : "Send message"}
                <Send size={16} />
              </button>
            </form>

            {reply && (
              <div className="response">
                <strong>AI response</strong>
                <div className="markdown-response">
                   <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {reply}
                  </ReactMarkdown>
                </div>
              </div>
            )}

            {error && (
              <p className="error" role="alert">
                {error}
              </p>
            )}
          </article>
        </section>

        <footer>
          InsightAI · Data analysis and dashboard generation
        </footer>
      </main>
    </div>
  );
}
