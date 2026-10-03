import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CSV Chat AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# SESSION STATE
# =========================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "file_name" not in st.session_state:
    st.session_state.file_name = ""

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        opacity: 0.65;
        margin-bottom: 20px;
    }

    .user-box {
        padding: 12px 16px;
        border-radius: 18px;
        margin: 8px 0 8px 15%;
        border: 1px solid rgba(128,128,128,0.25);
    }

    .assistant-box {
        padding: 12px 16px;
        border-radius: 18px;
        margin: 8px 15% 8px 0;
        border: 1px solid rgba(128,128,128,0.25);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# BASIC FUNCTIONS
# =========================================================

def normalize_text(text):
    text = str(text).strip().lower()
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def compact_text(text):
    text = normalize_text(text)
    return re.sub(r"[^a-z0-9]", "", text)


def format_value(value):
    if pd.isna(value):
        return "Missing"

    if isinstance(value, (float, np.floating)):
        if np.isfinite(value):
            return f"{value:.4f}".rstrip("0").rstrip(".")

    return str(value)


def numeric_values(series):
    return pd.to_numeric(series, errors="coerce")


def is_numeric_column(series):
    return pd.api.types.is_numeric_dtype(series)


# =========================================================
# COLUMN DETECTION
# =========================================================

def find_column(query, columns):

    query_normal = normalize_text(query)
    query_compact = compact_text(query)

    sorted_columns = sorted(
        columns,
        key=lambda x: len(str(x)),
        reverse=True
    )

    # Exact match
    for col in sorted_columns:
        if normalize_text(col) == query_normal:
            return col

    # Compact exact match
    for col in sorted_columns:
        if compact_text(col) == query_compact:
            return col

    # Column exists inside query
    for col in sorted_columns:
        col_normal = normalize_text(col)
        col_compact = compact_text(col)

        if col_normal in query_normal:
            return col

        if col_compact in query_compact:
            return col

    return None


# =========================================================
# ROW NUMBER DETECTION
# =========================================================

def get_single_row_number(query):

    q = normalize_text(query)

    patterns = [
        r"\bshow\s+(\d+)\s*(?:no|number|num)\s+row\b",
        r"\bshow\s+row\s+(\d+)\b",
        r"\bshow\s+row\s+number\s+(\d+)\b",
        r"\bshow\s+(\d+)(?:st|nd|rd|th)\s+row\b",
        r"\b(\d+)\s*(?:no|number|num)\s+row\b",
        r"\b(\d+)(?:st|nd|rd|th)\s+row\b",
        r"\brow\s+(\d+)\s+information\b",
        r"\brow\s+(\d+)\s+info\b",
        r"\brow\s+number\s+(\d+)\b"
    ]

    for pattern in patterns:
        match = re.search(pattern, q)

        if match:
            return int(match.group(1))

    return None


# =========================================================
# ROW RANGE DETECTION
# =========================================================

def get_row_range(query):

    q = normalize_text(query)

    patterns = [
        r"\bshow\s+rows?\s+(\d+)\s+(?:to|-)\s+(\d+)",
        r"\brows?\s+(\d+)\s+(?:to|-)\s+(\d+)",
        r"\bshow\s+(\d+)\s+(?:to|-)\s+(\d+)\s+rows?"
    ]

    for pattern in patterns:
        match = re.search(pattern, q)

        if match:
            return (
                int(match.group(1)),
                int(match.group(2))
            )

    return None


# =========================================================
# DISPLAY FUNCTIONS
# =========================================================

def show_table(data, title=None, height=450):

    if title:
        st.subheader(title)

    st.dataframe(
        data,
        use_container_width=True,
        height=height
    )


def show_full_row(data, row_number, title):

    st.subheader(title)

    st.caption(
        f"Dataset row number: {row_number}"
    )

    row = data.iloc[row_number - 1]

    result = pd.DataFrame(
        {
            "Column": list(row.index),
            "Value": [
                format_value(value)
                for value in row.values
            ]
        }
    )

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# QUERY ENGINE
# =========================================================

def process_query(query, data):

    q = normalize_text(query)
    columns = list(data.columns)

    if not q:
        st.warning("Please enter a question.")
        return


    # =====================================================
    # ALL COLUMN NAMES
    # =====================================================

    all_column_commands = [
        "show all column name",
        "show all column names",
        "show all columns",
        "display all columns",
        "display column names",
        "list all columns",
        "list columns",
        "show column names",
        "what are the columns",
        "what are all columns",
        "give me all columns"
    ]

    if any(command in q for command in all_column_commands):

        result = pd.DataFrame(
            {
                "No.": range(1, len(columns) + 1),
                "Column Name": columns,
                "Data Type": [
                    str(data[col].dtype)
                    for col in columns
                ],
                "Non-Null": [
                    int(data[col].notna().sum())
                    for col in columns
                ],
                "Missing": [
                    int(data[col].isna().sum())
                    for col in columns
                ],
                "Unique": [
                    int(data[col].nunique())
                    for col in columns
                ]
            }
        )

        st.success(
            f"Found {len(columns)} columns."
        )

        show_table(
            result,
            "📋 All Column Names",
            500
        )

        return


    # =====================================================
    # SINGLE ROW
    # =====================================================

    row_number = get_single_row_number(q)

    if row_number is not None:

        if row_number < 1 or row_number > len(data):

            st.error(
                f"Row {row_number} does not exist. "
                f"Available rows: 1 to {len(data)}."
            )

            return

        show_full_row(
            data,
            row_number,
            f"📌 Row {row_number} - Complete Information"
        )

        return


    # =====================================================
    # ROW RANGE
    # =====================================================

    row_range = get_row_range(q)

    if row_range:

        start, end = row_range

        if start > end:
            start, end = end, start

        if start < 1:
            start = 1

        if end > len(data):
            end = len(data)

        if start > len(data):

            st.error(
                f"Row {start} does not exist."
            )

            return

        result = data.iloc[start - 1:end]

        show_table(
            result,
            f"📋 Rows {start} to {end}",
            500
        )

        return


    # =====================================================
    # FIRST N ROWS
    # =====================================================

    match = re.search(
        r"\bfirst\s+(\d+)\s+rows?\b",
        q
    )

    if match:

        n = int(match.group(1))
        n = min(n, len(data))

        show_table(
            data.head(n),
            f"📋 First {n} Rows",
            500
        )

        return


    # =====================================================
    # LAST N ROWS
    # =====================================================

    match = re.search(
        r"\blast\s+(\d+)\s+rows?\b",
        q
    )

    if match:

        n = int(match.group(1))
        n = min(n, len(data))

        show_table(
            data.tail(n),
            f"📋 Last {n} Rows",
            500
        )

        return


    # =====================================================
    # RANDOM N ROWS
    # =====================================================

    match = re.search(
        r"\brandom\s+(\d+)\s+rows?\b",
        q
    )

    if match:

        n = int(match.group(1))
        n = min(n, len(data))

        result = data.sample(
            n=n,
            random_state=None
        )

        show_table(
            result,
            f"🎲 Random {n} Rows",
            500
        )

        return


    # =====================================================
    # DATASET SIZE
    # =====================================================

    if (
        "how many rows" in q
        or "total rows" in q
        or "number of rows" in q
        or "row count" in q
    ):

        st.metric(
            "Total Rows",
            len(data)
        )

        return


    if (
        "how many columns" in q
        or "total columns" in q
        or "number of columns" in q
        or "column count" in q
    ):

        st.metric(
            "Total Columns",
            len(columns)
        )

        return


    # =====================================================
    # MISSING
    # =====================================================

    if (
        "missing values" in q
        or "missing data" in q
        or "null values" in q
        or "null data" in q
    ):

        result = pd.DataFrame(
            {
                "Column": columns,
                "Missing": [
                    int(data[col].isna().sum())
                    for col in columns
                ]
            }
        )

        result = result[
            result["Missing"] > 0
        ]

        if result.empty:

            st.success(
                "There are no missing values."
            )

        else:

            show_table(
                result,
                "⚠️ Missing Values"
            )

        return


    # =====================================================
    # DUPLICATES
    # =====================================================

    if (
        "duplicate" in q
        or "duplicates" in q
    ):

        duplicates = data[
            data.duplicated(
                keep=False
            )
        ]

        st.metric(
            "Duplicate Rows",
            len(duplicates)
        )

        if not duplicates.empty:

            show_table(
                duplicates,
                "Duplicate Rows"
            )

        return


    # =====================================================
    # FIND COLUMN
    # =====================================================

    column = find_column(
        q,
        columns
    )


    # =====================================================
    # COLUMN OPERATIONS
    # =====================================================

    if column:

        series = data[column]


        # =================================================
        # FIRST N COLUMN VALUES
        # =================================================

        match = re.search(
            r"\bfirst\s+(\d+)\b",
            q
        )

        if (
            match
            and "row" not in q
            and "rows" not in q
        ):

            n = int(match.group(1))
            n = min(n, len(data))

            result = data[[column]].head(n)

            show_table(
                result,
                f"📋 First {n} Values of `{column}`"
            )

            return


        # =================================================
        # LAST N COLUMN VALUES
        # =================================================

        match = re.search(
            r"\blast\s+(\d+)\b",
            q
        )

        if (
            match
            and "row" not in q
            and "rows" not in q
        ):

            n = int(match.group(1))
            n = min(n, len(data))

            result = data[[column]].tail(n)

            show_table(
                result,
                f"📋 Last {n} Values of `{column}`"
            )

            return


        # =================================================
        # RANDOM N COLUMN VALUES
        # =================================================

        match = re.search(
            r"\brandom\s+(\d+)\b",
            q
        )

        if (
            match
            and "row" not in q
            and "rows" not in q
        ):

            n = int(match.group(1))
            n = min(n, len(data))

            result = data[[column]].sample(
                n=n
            )

            show_table(
                result,
                f"🎲 Random {n} Values of `{column}`"
            )

            return


        # =================================================
        # HIGHEST
        # =================================================

        if any(
            word in q
            for word in [
                "highest",
                "maximum",
                "largest",
                "max value",
                "highest value"
            ]
        ):

            if not is_numeric_column(series):

                st.warning(
                    f"`{column}` is not numeric."
                )

                return

            values = numeric_values(series)

            if values.dropna().empty:

                st.warning(
                    f"No numeric values found in `{column}`."
                )

                return

            index = values.idxmax()

            highest = values.loc[index]

            st.metric(
                f"Highest Value of {column}",
                format_value(highest)
            )

            if any(
                word in q
                for word in [
                    "information",
                    "info",
                    "details",
                    "detail",
                    "full row",
                    "complete row"
                ]
            ):

                position = (
                    data.index.get_loc(index) + 1
                )

                show_full_row(
                    data,
                    position,
                    f"🏆 Row with Highest `{column}`"
                )

            return


        # =================================================
        # LOWEST
        # =================================================

        if any(
            word in q
            for word in [
                "lowest",
                "minimum",
                "smallest",
                "min value",
                "lowest value"
            ]
        ):

            if not is_numeric_column(series):

                st.warning(
                    f"`{column}` is not numeric."
                )

                return

            values = numeric_values(series)

            if values.dropna().empty:

                st.warning(
                    f"No numeric values found in `{column}`."
                )

                return

            index = values.idxmin()

            lowest = values.loc[index]

            st.metric(
                f"Lowest Value of {column}",
                format_value(lowest)
            )

            if any(
                word in q
                for word in [
                    "information",
                    "info",
                    "details",
                    "detail",
                    "full row",
                    "complete row"
                ]
            ):

                position = (
                    data.index.get_loc(index) + 1
                )

                show_full_row(
                    data,
                    position,
                    f"📉 Row with Lowest `{column}`"
                )

            return


        # =================================================
        # AVERAGE
        # =================================================

        if (
            "average" in q
            or "avg" in q
            or "mean" in q
        ):

            if not is_numeric_column(series):

                st.warning(
                    f"`{column}` is not numeric."
                )

                return

            value = numeric_values(
                series
            ).mean()

            st.metric(
                f"Average Value of {column}",
                format_value(value)
            )

            return


        # =================================================
        # MEDIAN
        # =================================================

        if "median" in q:

            if not is_numeric_column(series):

                st.warning(
                    f"`{column}` is not numeric."
                )

                return

            value = numeric_values(
                series
            ).median()

            st.metric(
                f"Median of {column}",
                format_value(value)
            )

            return


        # =================================================
        # SUM
        # =================================================

        if (
            "sum" in q
            or "total value" in q
            or "total of" in q
        ):

            if not is_numeric_column(series):

                st.warning(
                    f"`{column}` is not numeric."
                )

                return

            value = numeric_values(
                series
            ).sum()

            st.metric(
                f"Total of {column}",
                format_value(value)
            )

            return


        # =================================================
        # COUNT
        # =================================================

        if (
            "count" in q
            or "how many values" in q
            or "number of values" in q
        ):

            st.metric(
                f"Non-Missing Values in {column}",
                int(series.notna().sum())
            )

            return


        # =================================================
        # UNIQUE
        # =================================================

        if (
            "unique" in q
            or "distinct" in q
        ):

            unique_values = (
                series
                .dropna()
                .unique()
            )

            result = pd.DataFrame(
                {
                    column: unique_values
                }
            )

            st.success(
                f"{len(unique_values)} unique values found."
            )

            show_table(
                result,
                f"Unique Values of `{column}`",
                500
            )

            return


        # =================================================
        # FILTER
        # =================================================

        filter_patterns = [
            (
                r"(?:greater than|more than|above|over)\s+(-?\d+(?:\.\d+)?)",
                ">"
            ),
            (
                r"(?:less than|below|under)\s+(-?\d+(?:\.\d+)?)",
                "<"
            ),
            (
                r"(?:equal to|equals)\s+(-?\d+(?:\.\d+)?)",
                "=="
            ),
            (
                r"(?:greater than or equal to|at least)\s+(-?\d+(?:\.\d+)?)",
                ">="
            ),
            (
                r"(?:less than or equal to|at most)\s+(-?\d+(?:\.\d+)?)",
                "<="
            )
        ]

        for pattern, operator in filter_patterns:

            match = re.search(
                pattern,
                q
            )

            if match:

                if not is_numeric_column(series):

                    st.warning(
                        f"`{column}` must be numeric."
                    )

                    return

                number = float(
                    match.group(1)
                )

                values = numeric_values(
                    series
                )

                if operator == ">":
                    mask = values > number

                elif operator == "<":
                    mask = values < number

                elif operator == "==":
                    mask = values == number

                elif operator == ">=":
                    mask = values >= number

                else:
                    mask = values <= number

                result = data[mask]

                show_table(
                    result,
                    f"`{column}` {operator} {number} — {len(result)} Rows",
                    500
                )

                return


        # =================================================
        # SORT DESCENDING
        # =================================================

        if (
            "sort" in q
            and (
                "descending" in q
                or "desc" in q
                or "highest first" in q
            )
        ):

            result = data.sort_values(
                by=column,
                ascending=False
            )

            show_table(
                result,
                f"`{column}` - Descending"
            )

            return


        # =================================================
        # SORT ASCENDING
        # =================================================

        if (
            "sort" in q
            and (
                "ascending" in q
                or "asc" in q
                or "lowest first" in q
            )
        ):

            result = data.sort_values(
                by=column,
                ascending=True
            )

            show_table(
                result,
                f"`{column}` - Ascending"
            )

            return


        # =================================================
        # COLUMN INFORMATION
        # =================================================

        if any(
            word in q
            for word in [
                "information",
                "info",
                "details",
                "detail",
                "about"
            ]
        ):

            st.subheader(
                f"📌 Information about `{column}`"
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Data Type",
                str(series.dtype)
            )

            c2.metric(
                "Total",
                len(series)
            )

            c3.metric(
                "Missing",
                int(series.isna().sum())
            )

            c4.metric(
                "Unique",
                int(series.nunique())
            )

            if is_numeric_column(series):

                values = numeric_values(series)

                st.divider()

                c1, c2, c3, c4, c5 = st.columns(5)

                c1.metric(
                    "Minimum",
                    format_value(values.min())
                )

                c2.metric(
                    "Maximum",
                    format_value(values.max())
                )

                c3.metric(
                    "Average",
                    format_value(values.mean())
                )

                c4.metric(
                    "Median",
                    format_value(values.median())
                )

                c5.metric(
                    "Sum",
                    format_value(values.sum())
                )

            return


        # =================================================
        # DEFAULT COLUMN DISPLAY
        # =================================================

        result = data[[column]]

        show_table(
            result,
            f"📋 Full Data of `{column}`",
            500
        )

        return


    # =====================================================
    # FALLBACK
    # =====================================================

    st.warning(
        "I could not understand that command."
    )

    st.info(
        "Use the examples below."
    )

    st.code(
        "show 10 no row\n"
        "show row 10\n"
        "show 10th row\n"
        "show 5 to 10 rows\n"
        "show first 10 rows\n"
        "show last 10 rows\n"
        "show random 10 rows\n"
        "show all column names\n"
        "show [column]\n"
        "show all [column]\n"
        "show first 10 [column]\n"
        "show last 10 [column]\n"
        "show random 10 [column]\n"
        "highest value of [column]\n"
        "lowest value of [column]\n"
        "average value of [column]\n"
        "median [column]\n"
        "sum [column]\n"
        "count [column]\n"
        "highest [column] information\n"
        "lowest [column] information\n"
        "[column] greater than 50\n"
        "[column] less than 50\n"
        "sort [column] ascending\n"
        "sort [column] descending"
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📂 CSV Chat AI")

    uploaded_file = st.file_uploader(
        "Upload your CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            data = pd.read_csv(
                uploaded_file
            )

            data.columns = [
                str(col).strip()
                for col in data.columns
            ]

            st.session_state.df = data
            st.session_state.file_name = uploaded_file.name

        except Exception as error:

            st.error(
                f"CSV reading error: {error}"
            )

    st.divider()

    if st.session_state.df is not None:

        current_data = st.session_state.df

        st.write(
            f"**File:** {st.session_state.file_name}"
        )

        st.write(
            f"**Rows:** {len(current_data):,}"
        )

        st.write(
            f"**Columns:** {len(current_data.columns):,}"
        )

        st.divider()

        st.subheader("⚡ Quick Commands")

        if st.button(
            "📋 All Columns",
            use_container_width=True
        ):

            st.session_state.quick_query = (
                "show all column names"
            )

        if st.button(
            "🔟 Row 10",
            use_container_width=True
        ):

            st.session_state.quick_query = (
                "show 10 no row"
            )

        if st.button(
            "📊 First 10 Rows",
            use_container_width=True
        ):

            st.session_state.quick_query = (
                "show first 10 rows"
            )

        if st.button(
            "📊 Last 10 Rows",
            use_container_width=True
        ):

            st.session_state.quick_query = (
                "show last 10 rows"
            )

        st.divider()

        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True
        ):

            st.session_state.messages = []

            st.rerun()


# =========================================================
# NO CSV
# =========================================================

if st.session_state.df is None:

    st.markdown(
        "<div class='main-title'>🤖 CSV Chat AI</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>Upload a CSV and chat with your dataset.</div>",
        unsafe_allow_html=True
    )

    st.info(
        "👈 Upload a CSV file from the sidebar."
    )

    st.markdown("### 💡 Example Commands")

    st.code(
        "show 10 no row\n"
        "show first 10 rows\n"
        "show last 10 rows\n"
        "show all column names\n"
        "show [column]\n"
        "highest value of [column]\n"
        "lowest value of [column]\n"
        "average value of [column]\n"
        "highest [column] information"
    )

    st.stop()


# =========================================================
# DATA PREPARATION
# =========================================================

df = st.session_state.df.copy()


# Convert numeric-looking object columns
for col in df.columns:

    if df[col].dtype == "object":

        converted = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        original_non_null = df[col].notna().sum()

        if original_non_null > 0:

            ratio = (
                converted.notna().sum()
                / original_non_null
            )

            if ratio >= 0.90:

                df[col] = converted


# =========================================================
# HEADER
# =========================================================

st.markdown(
    "<div class='main-title'>🤖 CSV Chat AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Chat naturally with your CSV dataset</div>",
    unsafe_allow_html=True
)


# =========================================================
# TOP METRICS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Rows",
    f"{len(df):,}"
)

c2.metric(
    "Columns",
    f"{len(df.columns):,}"
)

c3.metric(
    "Missing",
    f"{int(df.isna().sum().sum()):,}"
)

c4.metric(
    "Duplicates",
    f"{int(df.duplicated().sum()):,}"
)


# =========================================================
# TABS
# =========================================================

chat_tab, dataset_tab, chart_tab, ml_tab = st.tabs(
    [
        "💬 Chat",
        "📊 Dataset",
        "📈 Visualization",
        "🤖 Machine Learning"
    ]
)


# =========================================================
# CHAT TAB
# =========================================================

with chat_tab:

    st.subheader(
        "💬 Chat with your CSV"
    )

    st.caption(
        "Use the actual column names from your uploaded CSV."
    )

    # -----------------------------------------------------
    # Display chat history
    # -----------------------------------------------------

    for message in st.session_state.messages:

        if message["role"] == "user":

            st.markdown(
                f"<div class='user-box'><b>👤 You</b><br>{message['text']}</div>",
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"<div class='assistant-box'><b>🤖 CSV AI</b><br>{message['text']}</div>",
                unsafe_allow_html=True
            )


    # -----------------------------------------------------
    # Quick examples
    # -----------------------------------------------------

    with st.expander(
        "💡 Example Commands"
    ):

        st.code(
            "show 10 no row\n"
            "show row 10\n"
            "show 10th row\n"
            "show 5 to 10 rows\n"
            "show first 10 rows\n"
            "show last 10 rows\n"
            "show random 10 rows\n"
            "show all column names\n"
            "show X\n"
            "show all X\n"
            "show first 10 X\n"
            "show last 10 X\n"
            "show random 10 X\n"
            "highest value of X\n"
            "lowest value of X\n"
            "average value of X\n"
            "median X\n"
            "sum X\n"
            "count X\n"
            "highest X information\n"
            "lowest X information\n"
            "show unique X\n"
            "X greater than 50\n"
            "X less than 50\n"
            "sort X ascending\n"
            "sort X descending"
        )


    # -----------------------------------------------------
    # Quick buttons
    # -----------------------------------------------------

    q1, q2, q3, q4 = st.columns(4)

    with q1:

        if st.button(
            "📋 All Columns",
            key="chat_all_columns",
            use_container_width=True
        ):

            st.session_state.quick_query = (
                "show all column names"
            )

    with q2:

        if st.button(
            "🔟 Row 10",
            key="chat_row_10",
            use_container_width=True
        ):

            st.session_state.quick_query = (
                "show 10 no row"
            )

    with q3:

        if st.button(
            "📊 First 10",
            key="chat_first_10",
            use_container_width=True
        ):

            st.session_state.quick_query = (
                "show first 10 rows"
            )

    with q4:

        if st.button(
            "📊 Last 10",
            key="chat_last_10",
            use_container_width=True
        ):

            st.session_state.quick_query = (
                "show last 10 rows"
            )


    # -----------------------------------------------------
    # Chat input
    # -----------------------------------------------------

    typed_query = st.chat_input(
        "Ask something about your CSV..."
    )

    quick_query = st.session_state.pop(
        "quick_query",
        None
    )

    final_query = typed_query

    if quick_query:
        final_query = quick_query


    # -----------------------------------------------------
    # Process query
    # -----------------------------------------------------

    if final_query:

        st.session_state.messages.append(
            {
                "role": "user",
                "text": final_query
            }
        )

        # Assistant placeholder
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": "I found the following result:"
            }
        )

        # Display current result
        st.markdown(
            f"<div class='user-box'><b>👤 You</b><br>{final_query}</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<div class='assistant-box'><b>🤖 CSV AI</b><br>I found the following result:</div>",
            unsafe_allow_html=True
        )

        process_query(
            final_query,
            df
        )


# =========================================================
# DATASET TAB
# =========================================================

with dataset_tab:

    st.subheader(
        "📊 Dataset Explorer"
    )

    preview_type = st.radio(
        "Preview",
        [
            "First 10",
            "Last 10",
            "Random 10",
            "All"
        ],
        horizontal=True
    )

    if preview_type == "First 10":

        preview = df.head(10)

    elif preview_type == "Last 10":

        preview = df.tail(10)

    elif preview_type == "Random 10":

        preview = df.sample(
            n=min(10, len(df))
        )

    else:

        preview = df

    st.dataframe(
        preview,
        use_container_width=True,
        height=500
    )

    st.divider()

    st.subheader(
        "📋 Column Information"
    )

    information = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [
                str(df[col].dtype)
                for col in df.columns
            ],
            "Total": [
                len(df[col])
                for col in df.columns
            ],
            "Non-Null": [
                int(df[col].notna().sum())
                for col in df.columns
            ],
            "Missing": [
                int(df[col].isna().sum())
                for col in df.columns
            ],
            "Unique": [
                int(df[col].nunique())
                for col in df.columns
            ]
        }
    )

    st.dataframe(
        information,
        use_container_width=True,
        hide_index=True
    )

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download CSV",
        data=csv_data,
        file_name="processed_dataset.csv",
        mime="text/csv"
    )


# =========================================================
# VISUALIZATION TAB
# =========================================================

with chart_tab:

    st.subheader(
        "📈 Visualization"
    )

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    if not numeric_columns:

        st.warning(
            "No numeric columns found."
        )

    else:

        selected_column = st.selectbox(
            "Select Numeric Column",
            numeric_columns
        )

        chart_type = st.selectbox(
            "Chart Type",
            [
                "Histogram",
                "Box Plot",
                "Line Chart"
            ]
        )

        values = numeric_values(
            df[selected_column]
        ).dropna()

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        if chart_type == "Histogram":

            ax.hist(values)

            ax.set_title(
                f"Distribution of {selected_column}"
            )

            ax.set_xlabel(
                selected_column
            )

            ax.set_ylabel(
                "Frequency"
            )

        elif chart_type == "Box Plot":

            ax.boxplot(values)

            ax.set_title(
                f"Box Plot of {selected_column}"
            )

            ax.set_ylabel(
                selected_column
            )

        else:

            ax.plot(values.values)

            ax.set_title(
                f"{selected_column} by Row"
            )

            ax.set_xlabel(
                "Row Number"
            )

            ax.set_ylabel(
                selected_column
            )

        st.pyplot(
            fig,
            clear_figure=True
        )


# =========================================================
# MACHINE LEARNING TAB
# =========================================================

with ml_tab:

    st.subheader(
        "🤖 Machine Learning"
    )

    st.info(
        "Select any column as target. "
        "The system automatically detects classification or regression."
    )

    target_column = st.selectbox(
        "🎯 Target Column",
        df.columns,
        key="target_column"
    )

    target = df[target_column]

    unique_count = target.nunique(
        dropna=True
    )

    if (
        not is_numeric_column(target)
        or unique_count <= 20
    ):

        problem_type = "Classification"

    else:

        problem_type = "Regression"

    st.metric(
        "Detected Problem",
        problem_type
    )


    if st.button(
        "🚀 Train Models",
        type="primary",
        use_container_width=True
    ):

        ml_data = df.dropna(
            subset=[target_column]
        ).copy()

        if len(ml_data) < 10:

            st.error(
                "Not enough usable rows for ML."
            )

        else:

            X = ml_data.drop(
                columns=[target_column]
            )

            y = ml_data[target_column]

            X = X.dropna(
                axis=1,
                how="all"
            )

            if X.shape[1] == 0:

                st.error(
                    "No usable feature columns found."
                )

            else:

                numeric_features = (
                    X.select_dtypes(
                        include=np.number
                    )
                    .columns
                    .tolist()
                )

                categorical_features = (
                    X.select_dtypes(
                        exclude=np.number
                    )
                    .columns
                    .tolist()
                )

                transformers = []

                if numeric_features:

                    numeric_pipeline = Pipeline(
                        steps=[
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="median"
                                )
                            ),
                            (
                                "scaler",
                                StandardScaler()
                            )
                        ]
                    )

                    transformers.append(
                        (
                            "numeric",
                            numeric_pipeline,
                            numeric_features
                        )
                    )

                if categorical_features:

                    categorical_pipeline = Pipeline(
                        steps=[
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="most_frequent"
                                )
                            ),
                            (
                                "encoder",
                                OneHotEncoder(
                                    handle_unknown="ignore"
                                )
                            )
                        ]
                    )

                    transformers.append(
                        (
                            "categorical",
                            categorical_pipeline,
                            categorical_features
                        )
                    )

                preprocessor = ColumnTransformer(
                    transformers=transformers
                )


                # =========================================
                # CLASSIFICATION
                # =========================================

                if problem_type == "Classification":

                    if y.nunique() < 2:

                        st.error(
                            "Target must have at least two classes."
                        )

                    else:

                        try:

                            X_train, X_test, y_train, y_test = (
                                train_test_split(
                                    X,
                                    y,
                                    test_size=0.20,
                                    random_state=42,
                                    stratify=y
                                )
                            )

                        except Exception:

                            X_train, X_test, y_train, y_test = (
                                train_test_split(
                                    X,
                                    y,
                                    test_size=0.20,
                                    random_state=42
                                )
                            )

                        models = {
                            "Logistic Regression":
                                LogisticRegression(
                                    max_iter=1000
                                ),

                            "Random Forest":
                                RandomForestClassifier(
                                    n_estimators=150,
                                    random_state=42
                                ),

                            "Gradient Boosting":
                                GradientBoostingClassifier(
                                    random_state=42
                                )
                        }

                        results = []

                        for name, model in models.items():

                            try:

                                pipeline = Pipeline(
                                    steps=[
                                        (
                                            "preprocessor",
                                            preprocessor
                                        ),
                                        (
                                            "model",
                                            model
                                        )
                                    ]
                                )

                                pipeline.fit(
                                    X_train,
                                    y_train
                                )

                                prediction = pipeline.predict(
                                    X_test
                                )

                                results.append(
                                    {
                                        "Model": name,
                                        "Accuracy": round(
                                            accuracy_score(
                                                y_test,
                                                prediction
                                            ),
                                            4
                                        ),
                                        "Precision": round(
                                            precision_score(
                                                y_test,
                                                prediction,
                                                average="weighted",
                                                zero_division=0
                                            ),
                                            4
                                        ),
                                        "Recall": round(
                                            recall_score(
                                                y_test,
                                                prediction,
                                                average="weighted",
                                                zero_division=0
                                            ),
                                            4
                                        ),
                                        "F1 Score": round(
                                            f1_score(
                                                y_test,
                                                prediction,
                                                average="weighted",
                                                zero_division=0
                                            ),
                                            4
                                        )
                                    }
                                )

                            except Exception as error:

                                st.warning(
                                    f"{name} failed: {error}"
                                )

                        if results:

                            st.dataframe(
                                pd.DataFrame(results),
                                use_container_width=True,
                                hide_index=True
                            )


                # =========================================
                # REGRESSION
                # =========================================

                else:

                    X_train, X_test, y_train, y_test = (
                        train_test_split(
                            X,
                            y,
                            test_size=0.20,
                            random_state=42
                        )
                    )

                    models = {
                        "Linear Regression":
                            LinearRegression(),

                        "Random Forest":
                            RandomForestRegressor(
                                n_estimators=150,
                                random_state=42
                            ),

                        "Gradient Boosting":
                            GradientBoostingRegressor(
                                random_state=42
                            )
                    }

                    results = []

                    for name, model in models.items():

                        try:

                            pipeline = Pipeline(
                                steps=[
                                    (
                                        "preprocessor",
                                        preprocessor
                                    ),
                                    (
                                        "model",
                                        model
                                    )
                                ]
                            )

                            pipeline.fit(
                                X_train,
                                y_train
                            )

                            prediction = pipeline.predict(
                                X_test
                            )

                            mae = mean_absolute_error(
                                y_test,
                                prediction
                            )

                            rmse = np.sqrt(
                                mean_squared_error(
                                    y_test,
                                    prediction
                                )
                            )

                            r2 = r2_score(
                                y_test,
                                prediction
                            )

                            results.append(
                                {
                                    "Model": name,
                                    "MAE": round(
                                        mae,
                                        4
                                    ),
                                    "RMSE": round(
                                        rmse,
                                        4
                                    ),
                                    "R²": round(
                                        r2,
                                        4
                                    )
                                }
                            )

                        except Exception as error:

                            st.warning(
                                f"{name} failed: {error}"
                            )

                    if results:

                        st.dataframe(
                            pd.DataFrame(results),
                            use_container_width=True,
                            hide_index=True
                        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🤖 CSV Chat AI | Dynamic CSV Analysis | Statistics | Visualization | Machine Learning"
)
