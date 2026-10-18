import io

import json

import os

import uuid

from typing import Any

import ollama

import pandas as pd

from fastapi import FastAPI, File, HTTPException, UploadFile

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

# ==========================================

# INSIGHTAI CONFIGURATION

# ==========================================

app = FastAPI(

    title="InsightAI Data Analyst API",

    version="0.3.0",

)

app.add_middleware(

    CORSMiddleware,

    allow_origins=[

        "http://localhost:5173",

        "http://127.0.0.1:5173",

    ],

    allow_credentials=False,

    allow_methods=["GET", "POST"],

    allow_headers=["Content-Type"],

)

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

MAX_FILE_SIZE = 10 * 1024 * 1024

MAX_ROWS = 100_000

MAX_COLUMNS = 200

# Prototype storage. Data is lost when the backend restarts.

datasets: dict[str, dict[str, Any]] = {}

# ==========================================

# REQUEST MODELS

# ==========================================

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    provider: str = Field(default="ollama", pattern="^(ollama|gemini)$")
    apiKey: str | None = Field(default=None, max_length=500)
    model: str | None = Field(default=None, max_length=120)

class AgentRequest(BaseModel):
    datasetId: str
    question: str = Field(min_length=1, max_length=2000)
    provider: str = Field(default="ollama", pattern="^(ollama|gemini)$")
    apiKey: str | None = Field(default=None, max_length=500)
    model: str | None = Field(default=None, max_length=120)

class DashboardRequest(BaseModel):
    provider: str = Field(default="ollama", pattern="^(ollama|gemini)$")
    apiKey: str | None = Field(default=None, max_length=500)
    model: str | None = Field(default=None, max_length=120)


def generate_text(provider: str, messages: list[dict[str, str]], api_key: str | None = None, model: str | None = None, json_mode: bool = False) -> str:
    """Generate language only; all numeric analysis remains in Pandas tools."""
    if provider == "gemini":
        if not api_key or not api_key.strip():
            raise ValueError("Enter your Gemini API key or switch to Ollama.")
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("Gemini support is not installed. Run: pip install google-genai") from exc
        client = genai.Client(api_key=api_key.strip())
        prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        config = {"temperature": 0}
        if json_mode:
            config["response_mime_type"] = "application/json"
        response = client.models.generate_content(
            model=(model or "gemini-3.8-flash"),
            contents=prompt,
            config=config,
        )
        return response.text or ""
    kwargs = {
        "model": (model or MODEL),
        "options": {"temperature": 0},
        "messages": messages,
    }
    if json_mode:
        kwargs["format"] = "json"
    response = ollama.chat(**kwargs)
    return response.message.content or ""

# ==========================================

# SHARED HELPERS

# ==========================================

def get_dataset(dataset_id: str) -> dict[str, Any]:

    if dataset_id not in datasets:

        raise HTTPException(

            status_code=404,

            detail="Dataset not found. Upload the dataset again.",

        )

    return datasets[dataset_id]

def clean_value(value: Any) -> Any:

    """Convert Pandas and NumPy scalar values into JSON-safe values."""

    if value is None:

        return None

    try:

        if pd.isna(value):

            return None

    except (TypeError, ValueError):

        pass

    if hasattr(value, "isoformat"):

        return value.isoformat()

    if hasattr(value, "item"):

        return value.item()

    return value

def dashboard_number(value: Any) -> int | float | None:

    """Return a finite, JSON-safe numeric value."""

    if value is None or pd.isna(value):

        return None

    number = float(value)

    if not pd.notna(number) or number in (float("inf"), float("-inf")):

        return None

    if number.is_integer():

        return int(number)

    return round(number, 2)

def is_numeric_column(series: pd.Series) -> bool:

    return (

        pd.api.types.is_numeric_dtype(series)

        and not pd.api.types.is_bool_dtype(series)

    )

# ==========================================

# BASIC API ROUTES

# ==========================================

@app.get("/")

def home():

    return {

        "project": "InsightAI",

        "status": "running",

        "version": "0.3.0",

    }

@app.get("/api/health")

def health():

    return {

        "api": "online",

        "model": MODEL,

        "agent": "tool-using",

    }

@app.post("/api/chat")

def chat(request: ChatRequest):

    try:

        reply = generate_text(
            request.provider,
            [
                {"role": "system", "content": (
                    "You are InsightAI, a helpful data analyst assistant. "
                    "Do not claim to have analyzed a dataset unless its data "
                    "or computed results were provided."
                )},
                {"role": "user", "content": request.message},
            ],
            api_key=request.apiKey,
            model=request.model,
        )
        return {"reply": reply, "model": request.model or ("gemini-3.8-flash" if request.provider == "gemini" else MODEL), "provider": request.provider}

    except Exception as exc:

        print(f"Ollama chat error: {type(exc).__name__}: {exc}")

        raise HTTPException(

            status_code=503,

            detail="AI provider request failed. Check the selected provider, API key, model, and backend logs.",

        ) from exc

# ==========================================

# DATASET UPLOAD

# ==========================================

@app.post("/api/datasets")

async def upload_dataset(file: UploadFile = File(...)):

    filename = file.filename or ""

    extension = os.path.splitext(filename)[1].lower()

    if extension not in {".csv", ".xlsx", ".xls"}:

        raise HTTPException(

            status_code=400,

            detail="Upload a CSV or Excel file.",

        )

    content = await file.read(MAX_FILE_SIZE + 1)

    if len(content) > MAX_FILE_SIZE:

        raise HTTPException(

            status_code=413,

            detail="File exceeds the 10 MB limit.",

        )

    if not content:

        raise HTTPException(

            status_code=400,

            detail="The uploaded file is empty.",

        )

    try:

        if extension == ".csv":

            df = pd.read_csv(

                io.BytesIO(content),

                nrows=MAX_ROWS + 1,

            )

        elif extension == ".xlsx":

            df = pd.read_excel(

                io.BytesIO(content),

                nrows=MAX_ROWS + 1,

                engine="openpyxl",

            )

        else:

            # Legacy .xls files require the xlrd package.

            df = pd.read_excel(

                io.BytesIO(content),

                nrows=MAX_ROWS + 1,

                engine="xlrd",

            )

    except Exception as exc:

        print(f"Dataset parsing error: {type(exc).__name__}: {exc}")

        raise HTTPException(

            status_code=400,

            detail=(

                f"Could not read file: {type(exc).__name__}: {exc}"

            ),

        ) from exc

    if len(df) > MAX_ROWS:

        raise HTTPException(

            status_code=400,

            detail=f"Datasets are limited to {MAX_ROWS:,} rows.",

        )

    if len(df.columns) == 0:

        raise HTTPException(

            status_code=400,

            detail="The file has no columns.",

        )

    if len(df.columns) > MAX_COLUMNS:

        raise HTTPException(

            status_code=400,

            detail=f"Datasets are limited to {MAX_COLUMNS} columns.",

        )

    df.columns = [str(column).strip() for column in df.columns]

    if any(not column for column in df.columns):

        raise HTTPException(

            status_code=400,

            detail="Every column must have a name.",

        )

    if df.columns.duplicated().any():

        raise HTTPException(

            status_code=400,

            detail="Duplicate column names are not supported.",

        )

    dataset_id = str(uuid.uuid4())

    datasets[dataset_id] = {

        "filename": filename,

        "df": df,

    }

    return {

        "datasetId": dataset_id,

        "filename": filename,

        "rowCount": len(df),

        "columnCount": len(df.columns),

        "columns": [str(column) for column in df.columns],

    }

# ==========================================

# DATA PREVIEW

# ==========================================

@app.get("/api/datasets/{dataset_id}/preview")

def preview_dataset(dataset_id: str):

    item = get_dataset(dataset_id)

    df = item["df"]

    rows = []

    for row in df.head(10).to_dict(orient="records"):

        rows.append({

            str(column): clean_value(value)

            for column, value in row.items()

        })

    return {

        "datasetId": dataset_id,

        "filename": item["filename"],

        "columns": [str(column) for column in df.columns],

        "rows": rows,

    }

# ==========================================

# DATA PROFILING

# ==========================================

@app.get("/api/datasets/{dataset_id}/profile")

def profile_dataset(dataset_id: str):

    item = get_dataset(dataset_id)

    df = item["df"]

    columns = []

    for name in df.columns:

        series = df[name]

        missing = int(series.isna().sum())

        columns.append({

            "name": str(name),

            "dtype": str(series.dtype),

            "missing": missing,

            "missingPercentage": round(

                missing / len(df) * 100, 2

            ) if len(df) else 0,

            "distinct": int(series.nunique()),

        })

    numeric = df.select_dtypes(include="number")

    statistics = {}

    if not numeric.empty:

        for column, values in numeric.describe().to_dict().items():

            statistics[str(column)] = {

                key: clean_value(value)

                for key, value in values.items()

            }

    return {

        "datasetId": dataset_id,

        "filename": item["filename"],

        "rowCount": len(df),

        "columnCount": len(df.columns),

        "totalMissing": int(df.isna().sum().sum()),

        "columns": columns,

        "numericStatistics": statistics,

    }

# ==========================================

# APPROVED READ-ONLY ANALYSIS TOOLS

# ==========================================

AGENT_TOOLS = [

    {

        "type": "function",

        "function": {

            "name": "profile_dataset",

            "description": "Inspect dataset columns, types, row count and missing values.",

            "parameters": {

                "type": "object",

                "properties": {},

                "required": [],

            },

        },

    },

    {

        "type": "function",

        "function": {

            "name": "summarize_column",

            "description": "Calculate count, sum, mean, minimum and maximum for a numeric column.",

            "parameters": {

                "type": "object",

                "properties": {

                    "column": {"type": "string"},

                },

                "required": ["column"],

            },

        },

    },

    {

        "type": "function",

        "function": {

            "name": "group_by",

            "description": "Aggregate a column by category. Use sum or avg for numeric metrics, and count for record counts.",

            "parameters": {

                "type": "object",

                "properties": {

                    "category_column": {"type": "string"},

                    "value_column": {"type": "string"},

                    "aggregation": {

                        "type": "string",

                        "enum": ["sum", "avg", "count"],

                    },

                },

                "required": [

                    "category_column",

                    "value_column",

                    "aggregation",

                ],

            },

        },

    },

    {

        "type": "function",

        "function": {

            "name": "top_values",

            "description": "Find the most frequent values in a column.",

            "parameters": {

                "type": "object",

                "properties": {

                    "column": {"type": "string"},

                    "limit": {"type": "integer"},

                },

                "required": ["column"],

            },

        },

    },

    {

        "type": "function",

        "function": {

            "name": "missing_values",

            "description": "Inspect missing values in every dataset column.",

            "parameters": {

                "type": "object",

                "properties": {},

                "required": [],

            },

        },

    },

]

def execute_analysis_tool(

    tool_name: str,

    arguments: dict[str, Any],

    df: pd.DataFrame,

):

    """Run an explicitly approved tool. Never execute model-generated code."""

    if tool_name == "profile_dataset":

        return {

            "rows": len(df),

            "columns": [

                {

                    "name": str(column),

                    "type": str(df[column].dtype),

                    "distinct": int(df[column].nunique()),

                    "missing": int(df[column].isna().sum()),

                }

                for column in df.columns

            ],

        }

    if tool_name == "summarize_column":

        column = arguments.get("column")

        if column not in df.columns:

            raise ValueError(f"Unknown column: {column}")

        series = pd.to_numeric(

            df[column], errors="coerce"

        ).dropna()

        if series.empty:

            raise ValueError(f"{column} has no usable numeric values.")

        return {

            "column": column,

            "count": int(series.count()),

            "sum": dashboard_number(series.sum()),

            "mean": dashboard_number(series.mean()),

            "min": dashboard_number(series.min()),

            "max": dashboard_number(series.max()),

        }

    if tool_name == "group_by":

        category = arguments.get("category_column")

        value = arguments.get("value_column")

        aggregation = arguments.get("aggregation")

        if category not in df.columns or value not in df.columns:

            raise ValueError("Choose columns that exist in the dataset.")

        if aggregation not in {"sum", "avg", "count"}:

            raise ValueError("Unsupported aggregation.")

        working = df[[category, value]].copy()

        if aggregation in {"sum", "avg"}:

            working[value] = pd.to_numeric(

                working[value], errors="coerce"

            )

        grouped_data = working.groupby(

            category, dropna=False

        )

        if aggregation == "sum":

            grouped = grouped_data[value].sum(min_count=1)

        elif aggregation == "avg":

            grouped = grouped_data[value].mean()

        else:

            # Count records per category, including records with a

            # missing value in the selected value column.

            grouped = grouped_data.size()

        results = []

        for key, result in grouped.items():

            results.append({

                "group": (

                    str(key) if not pd.isna(key) else "(Missing)"

                ),

                "value": dashboard_number(result),

            })

        results.sort(

            key=lambda row: (

                row["value"] is not None,

                row["value"] if row["value"] is not None else 0,

            ),

            reverse=True,

        )

        return {

            "category_column": category,

            "value_column": value,

            "aggregation": aggregation,

            "results": results[:100],

        }

    if tool_name == "top_values":

        column = arguments.get("column")

        if column not in df.columns:

            raise ValueError(f"Unknown column: {column}")

        try:

            limit = int(arguments.get("limit", 10))

        except (TypeError, ValueError):

            limit = 10

        limit = max(1, min(limit, 100))

        counts = df[column].value_counts(dropna=False).head(limit)

        return {

            "column": column,

            "results": [

                {

                    "value": (

                        str(value) if not pd.isna(value) else "(Missing)"

                    ),

                    "count": int(count),

                }

                for value, count in counts.items()

            ],

        }

    if tool_name == "missing_values":

        return {

            "totalMissing": int(df.isna().sum().sum()),

            "columns": [

                {

                    "column": str(column),

                    "missing": int(df[column].isna().sum()),

                    "percentage": round(

                        float(df[column].isna().mean() * 100), 2

                    ),

                }

                for column in df.columns

            ],

        }

    raise ValueError("This analysis tool is not permitted.")

# ==========================================

# TOOL-USING AI AGENT

# ==========================================





def run_analysis_agent(dataset_id: str, question: str, provider: str = "ollama", api_key: str | None = None, model: str | None = None):

    """Analyze a dataset using approved tools and verified calculations."""

    import re



    item = get_dataset(dataset_id)

    df = item["df"]

    q = re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()



    tool_history = []

    numeric_columns = [

        str(c) for c in df.columns if is_numeric_column(df[c])

    ]

    categorical_columns = [

        str(c) for c in df.columns if not is_numeric_column(df[c])

    ]



    verified_results = {

        "filename": item["filename"],

        "row_count": len(df),

        "columns": [str(c) for c in df.columns],

        "profile": None,

        "numeric_summaries": [],

        "grouped_results": [],

        "top_values": [],

        "missing_values": None,

        "record_extremes": [],

    }



    def run_tool(name, arguments):

        try:

            result = execute_analysis_tool(name, arguments, df)

            tool_history.append({

                "tool": name,

                "status": "completed",

                "arguments": arguments,

            })

            return result

        except (ValueError, TypeError, KeyError) as exc:

            tool_history.append({

                "tool": name,

                "status": "failed",

                "arguments": arguments,

                "error": str(exc),

            })

            return {"error": str(exc)}



    def has(*terms):

        return any(

            re.search(r"\b" + re.escape(term) + r"\b", q)

            for term in terms

        )



    def normalize(value):

        return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()



    def fmt(value):

        if value is None:

            return "N/A"

        try:

            number = float(value)

            if not pd.notna(number):

                return "N/A"

            if number.is_integer():

                return f"{int(number):,}"

            return f"{number:,.2f}"

        except (TypeError, ValueError):

            return str(value)



    def find_column(candidates, available):

        """Find a column by semantic name, not column position alone."""

        for candidate in candidates:

            for column in available:

                if candidate in normalize(column).split():

                    return column

        return None



    def find_mentioned(columns):

        for column in columns:

            name = normalize(column)

            if name and re.search(

                r"\b" + re.escape(name) + r"\b", q

            ):

                return column

        return None



    # Identify common business metrics and category columns.

    metric_column = find_column(

        ["revenue", "sales", "profit", "amount", "value"],

        numeric_columns,

    )

    if metric_column is None:

        metric_column = find_mentioned(numeric_columns)



    if metric_column is None and len(numeric_columns) == 1:

        metric_column = numeric_columns[0]



    product_column = find_column(

        ["product", "item"], categorical_columns

    )

    region_column = find_column(

        ["region", "territory", "area"], categorical_columns

    )



    selected_category = find_mentioned(categorical_columns)



    if selected_category is None:
        if has("product", "products", "item", "items"):
            selected_category = product_column
        elif has("region", "regions", "territory", "area"):
            selected_category = region_column

    # Detect a specific category value mentioned in the question, e.g. "Phone"
    # in "What is the total revenue for Phone?".
    selected_category_value = None
    selected_category_value_column = None
    for category_candidate in categorical_columns:
        normalized_question = f" {q} "
        for raw_value in df[category_candidate].dropna().unique():
            value_text = str(raw_value).strip()
            normalized_value = normalize(value_text)
            if not normalized_value:
                continue
            if re.search(r"\b" + re.escape(normalized_value) + r"\b", normalized_question):
                if selected_category_value is None or len(normalized_value) > len(normalize(selected_category_value)):
                    selected_category_value = value_text
                    selected_category_value_column = category_candidate

    if selected_category_value is not None:
        selected_category = selected_category_value_column

    asks_average = has("average", "avg", "mean")

    asks_total = has("total", "sum", "overall")

    asks_missing = has("missing", "null", "empty", "incomplete")

    asks_frequency = has("frequency", "frequent", "common", "occurrences")

    asks_count = has("count", "counts", "frequency", "occurrences")



    asks_high = has("highest", "maximum", "max", "most", "top", "best")

    asks_low = has("lowest", "minimum", "min", "least", "worst")

    asks_rank = has("rank", "ranking", "compare", "comparison", "breakdown")

    asks_grouping = (

        selected_category is not None

        or has("by", "group", "compare", "comparison", "breakdown")

    )



    full_analysis = (

        has("complete", "comprehensive", "everything", "dashboard", "insights")

        or (

            has("analyze", "analyse")

            and has("dataset", "data", "report")

        )

    )



    # Always collect dataset metadata.

    verified_results["profile"] = run_tool("profile_dataset", {})



    if asks_missing or full_analysis:

        verified_results["missing_values"] = run_tool(

            "missing_values", {}

        )



    # Generate exact individual-record evidence for numeric columns.

    for column in numeric_columns[:10]:

        values = pd.to_numeric(df[column], errors="coerce")

        valid = values.dropna()



        if valid.empty:

            continue



        def matching_rows(target):

            positions = [

                i for i, value in enumerate(values.tolist())

                if pd.notna(value) and value == target

            ]

            return {

                "match_count": len(positions),

                "records": [

                    {

                        "row_number": i + 1,

                        "values": {

                            str(c): clean_value(v)

                            for c, v in df.iloc[i].items()

                        },

                    }

                    for i in positions[:10]

                ],

            }



        verified_results["record_extremes"].append({

            "column": column,

            "minimum": {

                "value": dashboard_number(valid.min()),

                **matching_rows(valid.min()),

            },

            "maximum": {

                "value": dashboard_number(valid.max()),

                **matching_rows(valid.max()),

            },

        })



    # Choose the metric required by the question.

    metrics_to_summarize = []

    mentioned_metric = find_mentioned(numeric_columns)



    if mentioned_metric:

        metrics_to_summarize = [mentioned_metric]

    elif metric_column and (

        asks_average or asks_total or asks_high or asks_low

        or asks_grouping or full_analysis

        or has("revenue", "sales", "profit", "amount", "value")

    ):

        metrics_to_summarize = [metric_column]

    elif full_analysis:

        metrics_to_summarize = numeric_columns[:10]



    for column in metrics_to_summarize:

        result = run_tool("summarize_column", {"column": column})

        if "error" not in result:

            verified_results["numeric_summaries"].append(result)



    # Group data when the question asks for a product/region comparison.

    category_columns = []



    if selected_category:

        category_columns = [selected_category]

    elif full_analysis:

        category_columns = categorical_columns[:5]

    elif asks_grouping and categorical_columns:

        category_columns = categorical_columns[:1]



    metrics_for_grouping = metrics_to_summarize[:3]



    for category in category_columns:

        for metric in metrics_for_grouping:

            aggregation = (

                "avg" if asks_average

                else "count" if asks_count and not (

                    asks_total or asks_average or asks_high or asks_low

                )

                else "sum"

            )

            result = run_tool(

                "group_by",

                {

                    "category_column": category,

                    "value_column": metric,

                    "aggregation": aggregation,

                },

            )

            if "error" not in result:

                verified_results["grouped_results"].append(result)



    if asks_frequency or full_analysis:

        columns = (

            [selected_category]

            if selected_category

            else categorical_columns[:5]

        )

        for column in columns:

            result = run_tool(

                "top_values", {"column": column, "limit": 10}

            )

            if "error" not in result:

                verified_results["top_values"].append(result)



    # Answer recognized numerical questions from verified results.

    # Do not ask the language model to recalculate these values.

    answer = None

    # Detect requested fields missing from the dataset before answering a
    # different metric by default.
    requested_field_groups = {
        "salary": {"salary", "salaries", "pay", "wage", "wages", "compensation"},
        "employee": {"employee", "employees", "staff", "worker", "workers"},
        "age": {"age", "ages"},
        "gender": {"gender", "sex"},
        "department": {"department", "departments", "team", "division"},
        "customer": {"customer", "customers", "client", "clients"},
        "profit": {"profit", "profits"},
        "revenue": {"revenue", "sales", "turnover"},
    }
    question_terms = set(q.split())
    normalized_column_words = {
        word for column in df.columns for word in normalize(column).split()
    }
    missing_requested_fields = []
    for field_name, aliases in requested_field_groups.items():
        if question_terms & aliases and not (aliases & normalized_column_words):
            missing_requested_fields.append(field_name)

    # "Employee salary" is one missing requested field, not two separate ones.
    if "salary" in missing_requested_fields:
        missing_requested_fields = [
            field for field in missing_requested_fields if field != "employee"
        ]

    if missing_requested_fields:
        available_text = ", ".join(str(column) for column in df.columns)
        missing_text = ", ".join(sorted(set(missing_requested_fields)))
        answer = (
            "I can't answer that accurately because the dataset doesn't contain "
            f"the requested field(s): **{missing_text}**.\n\n"
            f"Available columns: {available_text}."
        )



    # Highest/lowest individual record, distinct from group totals.

    asks_individual = (

        has("individual", "recorded", "single", "record")

        or (

            (asks_high or asks_low)

            and not selected_category

            and not asks_grouping

        )

    )



    if (

        asks_individual

        and (asks_high or asks_low)

        and metric_column

    ):

        extreme = next(

            (

                entry for entry in verified_results["record_extremes"]

                if entry["column"] == metric_column

            ),

            None,

        )



        if extreme:

            key = "maximum" if asks_high else "minimum"

            data = extreme[key]

            records = data["records"]



            if records:

                record = records[0]

                details = ", ".join(

                    f"{name}: {value}"

                    for name, value in record["values"].items()

                    if value is not None

                )

                answer = (

                    f"**{'Highest' if asks_high else 'Lowest'} "

                    f"individual {metric_column}: {fmt(data['value'])}**\n\n"

                    f"Record details: {details}\n\n"

                    f"Dataset row: {record['row_number']}."

                )



    # Answer specific category-value questions with a filtered calculation before
    # the generic whole-dataset total/average handler.
    if (
        answer is None
        and selected_category_value is not None
        and selected_category_value_column is not None
        and metric_column is not None
        and (asks_total or asks_average or has("revenue", "sales", "profit", "amount", "value"))
    ):
        category_series = df[selected_category_value_column].astype("string").str.strip().str.casefold()
        target_value = selected_category_value.casefold()
        filtered_values = pd.to_numeric(
            df.loc[category_series == target_value, metric_column],
            errors="coerce",
        ).dropna()

        if filtered_values.empty:
            answer = (
                f"No numeric {metric_column} values were found for "
                f"**{selected_category_value}** in {selected_category_value_column}."
            )
        elif asks_average:
            answer = (
                f"**Average {metric_column} for {selected_category_value}: "
                f"{fmt(filtered_values.mean())}** "
                f"(based on {fmt(len(filtered_values))} valid records)."
            )
        else:
            answer = (
                f"**Total {metric_column} for {selected_category_value}: "
                f"{fmt(filtered_values.sum())}** "
                f"(based on {fmt(len(filtered_values))} matching records)."
            )

    # Group totals or averages: answer using actual group results.

    if answer is None and selected_category and metric_column:

        matching_group = next(

            (

                result for result in verified_results["grouped_results"]

                if result.get("category_column") == selected_category

                and result.get("value_column") == metric_column

            ),

            None,

        )



        if matching_group and matching_group.get("results"):

            rows = matching_group["results"]

            valid_rows = [

                row for row in rows if row.get("value") is not None

            ]





        if valid_rows and (asks_high or asks_low):

            extreme_value = (

                max(row["value"] for row in valid_rows)

                if asks_high

                else min(row["value"] for row in valid_rows)

            )



            winners = [

                row for row in valid_rows

                if row["value"] == extreme_value

            ]



            direction = "highest" if asks_high else "lowest"

            aggregation = matching_group["aggregation"]



            if len(winners) > 1:

                winner_names = ", ".join(

                    f"**{row['group']}**" for row in winners

                )

                answer = (

                    f"**Tie:** {winner_names} share the {direction} "

                    f"{aggregation} of {metric_column}: "

                    f"**{fmt(extreme_value)}**.\n\n"

                    f"Results are grouped by {selected_category}."

                )

            else:

                answer = (

                    f"**{winners[0]['group']}** has the {direction} "

                    f"{aggregation} {metric_column}: "

                    f"**{fmt(extreme_value)}**.\n\n"

                    f"Results are grouped by {selected_category}."

                )





    # Simple totals and averages.

    if answer is None and metric_column:

        summary = next(

            (

                result for result in verified_results["numeric_summaries"]

                if result.get("column") == metric_column

            ),

            None,

        )



        if summary and asks_average:

            answer = (

                f"**Average {metric_column}: "

                f"{fmt(summary.get('mean'))}** "

                f"(based on {fmt(summary.get('count'))} valid values)."

            )

        elif summary and asks_total and not asks_grouping:

            answer = (

                f"**Total {metric_column}: {fmt(summary.get('sum'))}**."

            )



    # Reliable multi-filter and filtered-ranking calculations.
    # These calculations run directly against the uploaded DataFrame and take
    # precedence over a model's interpretation of precomputed whole-dataset totals.
    try:
        def mentioned_values_for(column_name):
            found = []
            if not column_name or column_name not in df.columns:
                return found
            padded_question = f" {q} "
            for raw_value in df[column_name].dropna().unique():
                value_text = str(raw_value).strip()
                normalized_value = normalize(value_text)
                if normalized_value and re.search(
                    r"\b" + re.escape(normalized_value) + r"\b",
                    padded_question,
                ):
                    found.append(value_text)
            return found

        # Resolve common business columns from the actual uploaded schema.
        product_col = find_column(["product", "item"], categorical_columns)
        region_col = find_column(["region", "territory", "area"], categorical_columns)
        metric_col = find_column(["revenue", "sales", "profit", "amount", "value"], numeric_columns)
        if metric_col is None:
            metric_col = metric_column

        # Example: "Which product has maximum revenue in the West?"
        # Filter Region=West, then aggregate Revenue by Product and rank it.
        asks_filtered_ranking = (
            metric_col is not None
            and product_col is not None
            and (asks_high or asks_low)
            and has("product", "products", "item", "items")
            and region_col is not None
            and bool(mentioned_values_for(region_col))
        )
        if asks_filtered_ranking:
            region_values = mentioned_values_for(region_col)
            filtered = df[df[region_col].astype("string").str.strip().str.casefold().isin(
                {value.casefold() for value in region_values}
            )].copy()
            numeric_metric = pd.to_numeric(filtered[metric_col], errors="coerce")
            grouped = numeric_metric.groupby(filtered[product_col], dropna=False).sum(min_count=1).dropna()
            if not grouped.empty:
                target = grouped.max() if asks_high else grouped.min()
                winners = grouped[grouped == target]
                direction = "highest" if asks_high else "lowest"
                names = ", ".join(f"**{str(name)}**" for name in winners.index)
                region_label = ", ".join(region_values)
                if len(winners) > 1:
                    answer = (
                        f"**Tie:** {names} share the {direction} total {metric_col} "
                        f"in {region_col}={region_label}: **{fmt(target)}**."
                    )
                else:
                    answer = (
                        f"**{str(winners.index[0])}** has the {direction} total {metric_col} "
                        f"in {region_col}={region_label}: **{fmt(target)}**."
                    )
                tool_history.append({
                    "tool": "filtered_group_by",
                    "status": "completed",
                    "arguments": {
                        "filter_column": region_col,
                        "filter_values": region_values,
                        "category_column": product_col,
                        "value_column": metric_col,
                        "aggregation": "sum",
                    },
                })

        # Example: "Total revenue for Laptop and Phone".
        # Detect every product value mentioned, filter to those products, and sum.
        if (
            metric_col is not None
            and product_col is not None
            and (
                asks_total
                or has("revenue", "sales", "profit", "amount", "value")
            )
        ):
            product_values = mentioned_values_for(product_col)
            if len(product_values) >= 2:
                product_mask = df[product_col].astype("string").str.strip().str.casefold().isin(
                    {value.casefold() for value in product_values}
                )
                values = pd.to_numeric(df.loc[product_mask, metric_col], errors="coerce").dropna()
                if not values.empty:
                    answer = (
                        f"**Total {metric_col} for {', '.join(product_values)}: "
                        f"{fmt(values.sum())}** (based on {fmt(len(values))} valid matching records)."
                    )
                    tool_history.append({
                        "tool": "filtered_sum",
                        "status": "completed",
                        "arguments": {
                            "filter_column": product_col,
                            "filter_values": product_values,
                            "value_column": metric_col,
                            "aggregation": "sum",
                        },
                    })
    except (KeyError, TypeError, ValueError) as exc:
        # Preserve the normal agent flow if a dataset has unusual column types.
        print(f"Deterministic filtered analysis skipped: {type(exc).__name__}: {exc}")

    # Missing-value answers are also deterministic.



    if answer is None:

        try:

            answer = generate_text(
                provider,
                [

                    {

                        "role": "system",

                        "content": (

                            "You are InsightAI, a precise data analyst. "

                            "Use only the verified results provided. "

                            "Never invent or alter numbers or row details. "

                            "Distinguish individual records from grouped "

                            "totals. If evidence is missing, say so. "

                            "Use readable Markdown."

                        ),

                    },

                    {

                        "role": "user",

                        "content": (

                            f"Dataset: {item['filename']}\n"

                            f"Question: {question}\n\n"

                            "Verified results:\n"

                            + json.dumps(

                                verified_results,

                                default=str,

                                ensure_ascii=False,

                            )

                        ),

                    },

                ],
                api_key=api_key,
                model=model,
            )
            if not answer:
                answer = "Analysis completed, but no explanation was returned."

        except Exception as exc:

            print(

                f"Ollama explanation error: {type(exc).__name__}: {exc}"

            )

            answer = (

                "The requested analysis could not be explained by Ollama. "

                "Check the Ollama service and backend logs."

            )



    return {

        "answer": answer,

        "toolCalls": tool_history,

        "datasetId": dataset_id,

        "verifiedResults": verified_results,

    }





@app.post("/api/agent/analyze")

def analyze_with_agent(request: AgentRequest):

    try:

        get_dataset(request.datasetId)

        return run_analysis_agent(
            request.datasetId,
            request.question.strip(),
            provider=request.provider,
            api_key=request.apiKey,
            model=request.model,
        )

    except HTTPException:

        raise

    except Exception as exc:

        print(f"Agent error: {type(exc).__name__}: {exc}")

        raise HTTPException(

            status_code=503,

            detail="Agent analysis failed. Check the selected AI provider, credentials, and backend logs.",

        ) from exc

# ==========================================

# AI DASHBOARD PLANNER

# ==========================================

def create_dashboard_plan(
    df: pd.DataFrame,
    provider: str = "ollama",
    api_key: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Ask Ollama to recommend charts using dataset metadata only."""

    columns = []

    for column in df.columns:

        series = df[column]

        columns.append({

            "name": str(column),

            "dtype": str(series.dtype),

            "numeric": is_numeric_column(series),

            "distinct": int(series.nunique()),

            "missing": int(series.isna().sum()),

            "examples": [

                str(value)

                for value in series.dropna().head(3).tolist()

            ],

        })

    prompt = {

        "dataset_rows": len(df),

        "dataset_columns": columns,

        "requirements": [

            "Recommend up to four useful charts.",

            "Use only exact column names supplied.",

            "Choose from bar, horizontal, line, area, pie, and donut chart types.",
            "Use horizontal charts for categories with long labels or many groups.",
            "Use area charts for meaningful ordered or time-based data.",
            "Use donut charts for part-to-whole comparisons with at most eight categories.",

            "Choose sum, avg, or count aggregation.",

            "Use sum or avg only with numeric metrics.",

            "Use count for record counts.",

            "Prefer bar charts for categorical comparisons.",

            "Use pie charts only when there are at most eight categories.",

            "Use line charts only for meaningful ordered or time data.",

            "Do not calculate or invent chart values.",

        ],

        "required_json_shape": {

            "charts": [

                {

                    "title": "Revenue by Region",

                    "type": "bar",

                    "category_column": "Region",

                    "metric_column": "Revenue",

                    "aggregation": "sum",

                }

            ]

        },

    }

    response_text = generate_text(
        provider=provider,
        api_key=api_key,
        model=model,
        json_mode=True,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a business intelligence dashboard planner. "
                    "Return valid JSON only, matching the requested shape. "
                    "Choose useful visualizations based on column names, "
                    "types, and example values. Do not invent data values."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt),
            },
        ],
    )

    plan = json.loads(response_text or "{}")

    if not isinstance(plan.get("charts"), list):

        raise ValueError("The AI returned an invalid dashboard plan.")

    return plan

def build_dashboard_charts(

    df: pd.DataFrame,

    plan: dict[str, Any],

) -> list[dict[str, Any]]:

    """Validate chart recommendations and calculate values with Pandas."""

    numeric_columns = [

        column for column in df.columns

        if is_numeric_column(df[column])

    ]

    charts = []

    used = set()

    for item in plan.get("charts", [])[:4]:

        if not isinstance(item, dict):

            continue

        category = item.get("category_column")

        metric = item.get("metric_column")

        aggregation = item.get("aggregation")

        chart_type = item.get("type")

        title = item.get("title")

        if category not in df.columns or metric not in df.columns:

            continue

        if aggregation not in {"sum", "avg", "count"}:

            continue

        if chart_type not in {
            "bar",
            "horizontal",
            "line",
            "area",
            "pie",
            "donut",
        }:
            continue

        if aggregation in {"sum", "avg"} and metric not in numeric_columns:

            continue

        if chart_type in {"pie", "donut"} and df[category].nunique() > 8:

            continue

        # Avoid misleading line charts without an ordered category.

        if chart_type in {"line", "area"}:

            is_datetime = pd.api.types.is_datetime64_any_dtype(

                df[category]

            )

            is_numeric = is_numeric_column(df[category])

            if not (is_datetime or is_numeric):

                continue

        # Numeric pie charts should show metric distribution, not row counts.
        if (
            chart_type in {"pie", "donut"}
            and metric in numeric_columns
            and aggregation == "count"
        ):
            aggregation = "sum"

        signature = (category, metric, aggregation, chart_type)

        if signature in used:

            continue

        used.add(signature)

        working = df[[category, metric]].copy()

        if aggregation in {"sum", "avg"}:

            working[metric] = pd.to_numeric(

                working[metric], errors="coerce"

            )

        grouped_data = working.groupby(

            category, dropna=False

        )

        if aggregation == "sum":

            grouped = grouped_data[metric].sum(min_count=1)

        elif aggregation == "avg":

            grouped = grouped_data[metric].mean()

        else:

            grouped = grouped_data.size()

        grouped = grouped.dropna()

        if chart_type == "line":

            grouped = grouped.sort_index()

        else:

            grouped = grouped.sort_values(ascending=False)

        grouped = grouped.head(15)

        if grouped.empty:

            continue

        data = [

            {

                "category": (

                    str(key) if not pd.isna(key) else "(Missing)"

                ),

                "value": dashboard_number(value),

            }

            for key, value in grouped.items()

        ]

        charts.append({

            "title": (

                title[:100].strip()

                if isinstance(title, str) and title.strip()

                else f"{metric} by {category}"

            ),

            "type": chart_type,

            "category": str(category),

            "metric": str(metric),

            "aggregation": aggregation,

            "data": data,

        })

    return charts


def build_fallback_charts(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Generate useful, deterministic dashboard charts when AI planning fails."""
    numeric_columns = [
        column
        for column in df.columns
        if is_numeric_column(df[column])
    ]

    if not numeric_columns or df.empty:
        return []

    datetime_columns = [
        column
        for column in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[column])
    ]

    categorical_columns = [
        column
        for column in df.columns
        if column not in numeric_columns
        and column not in datetime_columns
        and df[column].nunique(dropna=True) > 1
    ]

    charts: list[dict[str, Any]] = []
    used: set[tuple[str, str, str]] = set()

    def add_chart(
        category: Any,
        metric: Any,
        chart_type: str,
        aggregation: str = "sum",
    ) -> None:
        if len(charts) >= 4:
            return

        signature = (str(category), str(metric), aggregation)
        if signature in used:
            return

        if category not in df.columns or metric not in numeric_columns:
            return

        working = df[[category, metric]].copy()
        working[metric] = pd.to_numeric(
            working[metric], errors="coerce"
        )

        grouped_data = working.groupby(category, dropna=False)[metric]

        if aggregation == "avg":
            grouped = grouped_data.mean()
        else:
            grouped = grouped_data.sum(min_count=1)

        grouped = grouped.dropna()

        if grouped.empty:
            return

        if chart_type == "line":
            grouped = grouped.sort_index()
            # Keep the most recent 15 points while preserving time order.
            grouped = grouped.tail(15)
        else:
            grouped = grouped.sort_values(ascending=False).head(15)

        data = [
            {
                "category": (
                    str(key) if not pd.isna(key) else "(Missing)"
                ),
                "value": dashboard_number(value),
            }
            for key, value in grouped.items()
        ]

        data = [row for row in data if row["value"] is not None]
        if not data:
            return

        used.add(signature)
        charts.append({
            "title": f"{metric} by {category}",
            "type": chart_type,
            "category": str(category),
            "metric": str(metric),
            "aggregation": aggregation,
            "data": data,
        })

    # 1. Prioritize time-series charts for real datetime columns.
    for category in datetime_columns:
        if len(charts) >= 2:
            break

        if df[category].nunique(dropna=True) < 2:
            continue

        for metric in numeric_columns:
            add_chart(category, metric, "line")
            if len(charts) >= 2:
                break

    # 2. Compare numeric metrics across categorical dimensions.
    for category in categorical_columns:
        if len(charts) >= 4:
            break

        distinct_count = int(df[category].nunique(dropna=True))
        labels = df[category].dropna().astype(str)
        average_label_length = (
            labels.str.len().mean() if not labels.empty else 0
        )

        chart_type = (
            "horizontal"
            if distinct_count > 8 or average_label_length > 18
            else "bar"
        )

        for metric in numeric_columns:
            add_chart(category, metric, chart_type)
            if len(charts) >= 4:
                break

    # 3. If the dataset has no categorical dimensions, compare numeric
    # columns against one another using a deterministic row index.
    if not charts and len(numeric_columns) >= 2:
        working = df[numeric_columns].copy()
        working.insert(0, "Record", range(1, len(working) + 1))

        for metric in numeric_columns[:4]:
            values = pd.to_numeric(
                working[metric], errors="coerce"
            ).dropna().head(15)

            if values.empty:
                continue

            charts.append({
                "title": f"{metric} by Record",
                "type": "line",
                "category": "Record",
                "metric": str(metric),
                "aggregation": "sum",
                "data": [
                    {
                        "category": str(index),
                        "value": dashboard_number(value),
                    }
                    for index, value in values.items()
                ],
            })

            if len(charts) >= 4:
                break

    return charts


# ==========================================

# DASHBOARD API

# ==========================================

@app.post("/api/datasets/{dataset_id}/dashboard")
def generate_dashboard(
    dataset_id: str,
    request: DashboardRequest,
):

    item = get_dataset(dataset_id)

    df = item["df"]

    numeric_columns = [

        column for column in df.columns

        if is_numeric_column(df[column])

    ]

    # Compute KPI values directly from the dataset.

    kpis = [

        {

            "label": "Total records",

            "value": len(df),

        }

    ]

    for column in numeric_columns[:3]:

        values = pd.to_numeric(

            df[column], errors="coerce"

        ).dropna()

        if values.empty:

            continue

        kpis.append({

            "label": f"Total {column}",

            "value": dashboard_number(values.sum()),

        })

        kpis.append({

            "label": f"Average {column}",

            "value": dashboard_number(values.mean()),

        })

    # Ask AI for a chart plan. Use a deterministic fallback on failure.

    try:

        plan = create_dashboard_plan(
            df,
            provider=request.provider,
            api_key=request.apiKey,
            model=request.model,
        )

        charts = build_dashboard_charts(df, plan)

        planner = "AI"

        if not charts:

            charts = build_fallback_charts(df)

            planner = "Rules-based fallback"

    except Exception as exc:

        print(

            f"Dashboard planner error: {type(exc).__name__}: {exc}"

        )

        charts = build_fallback_charts(df)

        planner = "Rules-based fallback"


    insights = []

    for chart in charts:
        valid_data = [
            row for row in chart.get("data", [])
            if row.get("value") is not None
        ]

        if not valid_data:
            continue

        aggregation = chart.get("aggregation", "sum")
        metric = chart.get("metric", "value")
        category = chart.get("category", "category")
        top = max(valid_data, key=lambda row: row["value"])

        if aggregation == "count":
            finding = (
                f"{top['category']} has the most records "
                f"among the displayed {category} groups: "
                f"{dashboard_number(top['value'])}."
            )
        elif aggregation == "avg":
            finding = (
                f"{top['category']} has the highest average "
                f"{metric} among the displayed {category} groups: "
                f"{dashboard_number(top['value'])}."
            )
        else:
            finding = (
                f"{top['category']} has the highest total "
                f"{metric} among the displayed {category} groups: "
                f"{dashboard_number(top['value'])}."
            )

        insights.append(finding)

    for column in numeric_columns[:2]:
        values = pd.to_numeric(df[column], errors="coerce").dropna()

        if not values.empty:
            insights.append(
                f"{column}: total {dashboard_number(values.sum())}; "
                f"average {dashboard_number(values.mean())} across "
                f"{len(values)} non-missing numeric records."
            )

    return {
        "datasetId": dataset_id,
        "filename": item["filename"],
        "planner": planner,
        "rowCount": len(df),
        "columnCount": len(df.columns),
        "missingValues": int(df.isna().sum().sum()),
        "kpis": kpis,
        "charts": charts,
        "insights": insights[:8],
    }
