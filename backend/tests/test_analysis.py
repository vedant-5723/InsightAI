
import pandas as pd
import pytest

from app.main import datasets, run_analysis_agent, execute_analysis_tool


@pytest.fixture(autouse=True)
def clear_datasets():
    """Keep each test isolated from other datasets."""
    datasets.clear()
    yield
    datasets.clear()


def register_dataset(df):
    """Register a test dataset using InsightAI's existing storage."""
    dataset_id = "test-dataset"
    datasets[dataset_id] = {
        "filename": "test_sales.csv",
        "df": df,
    }
    return dataset_id


def test_total_revenue():
    df = pd.DataFrame({"Revenue": [50000, 40000, 30000, 35000, 25000]})
    dataset_id = register_dataset(df)

    result = run_analysis_agent(dataset_id, "What is the total revenue?")

    assert "180,000" in result["answer"]


def test_average_revenue_per_record():
    df = pd.DataFrame({"Revenue": [50000, 40000, 30000, 35000, 25000]})
    dataset_id = register_dataset(df)

    result = run_analysis_agent(
        dataset_id, "What is the average revenue per record?"
    )

    assert "36,000" in result["answer"]


def test_highest_region_total():
    df = pd.DataFrame({
        "Region": ["West", "West", "East"],
        "Revenue": [60000, 45000, 75000],
    })
    dataset_id = register_dataset(df)

    result = run_analysis_agent(
        dataset_id, "Which region has the highest total revenue?"
    )

    assert "West" in result["answer"]
    assert "105,000" in result["answer"]


def test_lowest_product_total():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet"],
        "Revenue": [95000, 65000, 20000],
    })
    dataset_id = register_dataset(df)

    result = run_analysis_agent(
        dataset_id, "Which product has the lowest total revenue?"
    )

    assert "Tablet" in result["answer"]
    assert "20,000" in result["answer"]


def test_highest_individual_record():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet"],
        "Region": ["West", "East", "West"],
        "Revenue": [50000, 30000, 20000],
    })
    dataset_id = register_dataset(df)

    result = run_analysis_agent(
        dataset_id,
        "Show the highest individual revenue record with product and region."
    )

    assert "50,000" in result["answer"]
    assert "Laptop" in result["answer"]
    assert "West" in result["answer"]


def test_missing_values():
    df = pd.DataFrame({
        "Product": ["Laptop", None, "Tablet"],
        "Revenue": [50000, 30000, None],
    })
    dataset_id = register_dataset(df)

    result = run_analysis_agent(
        dataset_id, "Are there any missing values in this dataset?"
    )

    assert "2" in result["answer"]


def test_group_by_tool():
    df = pd.DataFrame({
        "Region": ["West", "West", "East"],
        "Revenue": [60000, 45000, 75000],
    })

    result = execute_analysis_tool(
        "group_by",
        {
            "category_column": "Region",
            "value_column": "Revenue",
            "aggregation": "sum",
        },
        df,
    )

    totals = {row["group"]: row["value"] for row in result["results"]}

    assert totals["West"] == 105000
    assert totals["East"] == 75000


def test_group_by_rejects_unknown_column():
    df = pd.DataFrame({
        "Region": ["West", "East"],
        "Revenue": [100, 200],
    })

    with pytest.raises(ValueError, match="Choose columns"):
        execute_analysis_tool(
            "group_by",
            {
                "category_column": "Unknown",
                "value_column": "Revenue",
                "aggregation": "sum",
            },
            df,
        )


def test_group_by_rejects_unsupported_aggregation():
    df = pd.DataFrame({
        "Region": ["West", "East"],
        "Revenue": [100, 200],
    })

    with pytest.raises(ValueError, match="Unsupported aggregation"):
        execute_analysis_tool(
            "group_by",
            {
                "category_column": "Region",
                "value_column": "Revenue",
                "aggregation": "median",
            },
            df,
        )


def test_average_ignores_missing_numeric_values():
    df = pd.DataFrame({
        "Revenue": [100, None, 300],
    })

    result = execute_analysis_tool(
        "summarize_column",
        {"column": "Revenue"},
        df,
    )

    assert result["count"] == 2
    assert result["mean"] == 200
    assert result["sum"] == 400


def test_group_by_preserves_tied_totals():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet"],
        "Revenue": [100, 100, 50],
    })

    result = execute_analysis_tool(
        "group_by",
        {
            "category_column": "Product",
            "value_column": "Revenue",
            "aggregation": "sum",
        },
        df,
    )

    totals = {
        row["group"]: row["value"]
        for row in result["results"]
    }

    assert totals["Laptop"] == 100
    assert totals["Phone"] == 100
    assert totals["Tablet"] == 50


def test_group_by_handles_negative_values():
    df = pd.DataFrame({
        "Region": ["North", "South", "West"],
        "Profit": [-10, -30, -5],
    })

    result = execute_analysis_tool(
        "group_by",
        {
            "category_column": "Region",
            "value_column": "Profit",
            "aggregation": "sum",
        },
        df,
    )

    totals = {
        row["group"]: row["value"]
        for row in result["results"]
    }

    assert totals["North"] == -10
    assert totals["South"] == -30
    assert totals["West"] == -5




def test_agent_reports_all_tied_highest_products():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet"],
        "Revenue": [100, 100, 50],
    })

    dataset_id = "test-dataset"
    datasets[dataset_id] = {
        "filename": "tie_test.csv",
        "df": df,
    }

    result = run_analysis_agent(
        dataset_id,
        "Which product has the highest total revenue?",
    )

    answer = result["answer"].lower()

    assert "laptop" in answer
    assert "phone" in answer
    assert "100" in answer


def test_agent_does_not_invent_missing_columns():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet"],
        "Revenue": [100, 200, 50],
    })

    dataset_id = "test-dataset"
    datasets[dataset_id] = {
        "filename": "test.csv",
        "df": df,
    }

    result = run_analysis_agent(
        dataset_id,
        "What is the average employee salary?",
    )

    answer = result["answer"].lower()

    assert "employee salary" not in answer or any(
        phrase in answer
        for phrase in [
            "not available",
            "not found",
            "cannot determine",
            "no evidence",
            "does not contain",
            "could not",
            "unable to",
        ]
    )


def test_agent_rejects_unavailable_employee_salary():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet"],
        "Region": ["West", "East", "West"],
        "Revenue": [50000, 30000, 20000],
    })

    dataset_id = "test-dataset"
    datasets[dataset_id] = {
        "filename": "sales.csv",
        "df": df,
    }

    result = run_analysis_agent(
        dataset_id,
        "What is the average employee salary?",
    )

    answer = result["answer"].lower()

    assert "salary" in answer
    assert "revenue" not in answer or "available columns" in answer
    assert "50000" not in answer


def test_agent_uses_explicitly_requested_numeric_column():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet"],
        "Revenue": [50000, 30000, 20000],
        "Units Sold": [5, 7, 15],
    })
    dataset_id = register_dataset(df)

    result = run_analysis_agent(
        dataset_id,
        "What is the average Units Sold?",
    )

    assert "9" in result["answer"]
    assert "Units Sold" in result["answer"]


def test_agent_ranks_groups_using_explicitly_requested_metric():
    df = pd.DataFrame({
        "Market": ["North", "North", "South"],
        "Revenue": [50000, 30000, 20000],
        "Units Sold": [5, 7, 15],
    })
    dataset_id = register_dataset(df)

    result = run_analysis_agent(
        dataset_id,
        "Which market has the highest total Units Sold?",
    )

    assert "South" in result["answer"]
    assert "15" in result["answer"]
    assert "Units Sold" in result["answer"]


def test_agent_groups_by_explicitly_mentioned_custom_category():
    df = pd.DataFrame({
        "Sales Channel": ["Online", "Retail", "Online"],
        "Transaction Value": [100, 200, 300],
    })
    dataset_id = register_dataset(df)

    result = run_analysis_agent(
        dataset_id,
        "Which Sales Channel has the highest total Transaction Value?",
    )

    assert "Online" in result["answer"]
    assert "400" in result["answer"]
    assert "Transaction Value" in result["answer"]


def test_agent_uses_custom_category_for_grouped_summary():
    df = pd.DataFrame({
        "Branch": ["Pune", "Mumbai", "Pune"],
        "Units Sold": [10, 20, 15],
    })
    dataset_id = register_dataset(df)

    result = run_analysis_agent(
        dataset_id,
        "Show total Units Sold by Branch",
    )

    assert "Pune" in result["answer"]
    assert "Mumbai" in result["answer"]
    assert "25" in result["answer"]
    assert "20" in result["answer"]


def test_agent_asks_for_clarification_when_grouping_is_ambiguous():
    df = pd.DataFrame({
        "Branch": ["Pune", "Mumbai", "Pune"],
        "Department": ["Sales", "IT", "IT"],
        "Sales": [100, 200, 300],
    })
    dataset_id = register_dataset(df)

    result = run_analysis_agent(
        dataset_id,
        "Show total Sales by category",
    )

    answer = result["answer"]
    assert "Which column should I group by?" in answer
    assert "Branch" in answer
    assert "Department" in answer
