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

if "last_file_id" not in st.session_state:
    st.session_state.last_file_id = None

if "quick_query" not in st.session_state:
    st.session_state.quick_query = None


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
# FIND COLUMN
# =========================================================

def find_column(query, columns):

    q_normal = normalize_text(query)
    q_compact = compact_text(query)

    sorted_columns = sorted(
        columns,
        key=lambda x: len(str(x)),
        reverse=True
    )

    for col in sorted_columns:
        if normalize_text(col) == q_normal:
            return col

    for col in sorted_columns:
        if compact_text(col) == q_compact:
            return col

    for col in sorted_columns:

        normal_col = normalize_text(col)
        compact_col = compact_text(col)

        if normal_col and normal_col in q_normal:
            return col

        if compact_col and compact_col in q_compact:
            return col

    return None


# =========================================================
# ROW NUMBER
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
# ROW RANGE
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
# CHAT STORAGE
# =========================================================

def add_user_message(text):

    st.session_state.messages.append(
        {
            "role": "user",
            "type": "text",
            "text": text
        }
    )


def add_text_answer(text):

    st.session_state.messages.append(
        {
            "role": "assistant",
            "type": "text",
            "text": text
        }
    )


def add_table_answer(text, data):

    if isinstance(data, pd.DataFrame):
        saved_data = data.copy()
    else:
        saved_data = pd.DataFrame(data)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "type": "table",
            "text": text,
            "data": saved_data
        }
    )


def add_metric_answer(label, value, extra_text=""):

    st.session_state.messages.append(
        {
            "role": "assistant",
            "type": "metric",
            "label": label,
            "value": str(value),
            "text": extra_text
        }
    )


def add_error_answer(text):

    st.session_state.messages.append(
        {
            "role": "assistant",
            "type": "error",
            "text": text
        }
    )


# =========================================================
# RENDER CHAT MESSAGE
# =========================================================

def render_message(message):

    role = message.get("role")

    if role == "user":

        with st.chat_message("user"):
            st.write(
                message.get("text", "")
            )

        return

    with st.chat_message("assistant"):

        message_type = message.get(
            "type",
            "text"
        )

        if message_type == "text":

            st.write(
                message.get("text", "")
            )

        elif message_type == "error":

            st.error(
                message.get("text", "")
            )

        elif message_type == "metric":

            extra_text = message.get(
                "text",
                ""
            )

            if extra_text:
                st.write(extra_text)

            st.metric(
                message.get(
                    "label",
                    "Result"
                ),
                message.get(
                    "value",
                    ""
                )
            )

        elif message_type == "table":

            text = message.get(
                "text",
                ""
            )

            if text:
                st.write(text)

            table_data = message.get(
                "data"
            )

            if isinstance(
                table_data,
                pd.DataFrame
            ):

                st.dataframe(
                    table_data,
                    use_container_width=True,
                    height=400
                )


# =========================================================
# READ CSV
# =========================================================

def read_csv_file(uploaded_file):

    encodings = [
        "utf-8",
        "utf-8-sig",
        "latin1"
    ]

    last_error = None

    for encoding in encodings:

        try:

            data = pd.read_csv(
                uploaded_file,
                encoding=encoding
            )

            data.columns = [
                str(col).strip()
                for col in data.columns
            ]

            return data

        except Exception as error:

            last_error = error

    raise last_error


# =========================================================
# QUERY ENGINE
# =========================================================

def process_query(query, data):

    q = normalize_text(query)
    columns = list(data.columns)

    if not q:

        add_error_answer(
            "Please enter a question."
        )

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

    if any(
        command in q
        for command in all_column_commands
    ):

        result = pd.DataFrame(
            {
                "No.": range(
                    1,
                    len(columns) + 1
                ),
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

        add_table_answer(
            f"Found {len(columns)} columns.",
            result
        )

        return

    # =====================================================
    # SINGLE ROW
    # =====================================================

    row_number = get_single_row_number(q)

    if row_number is not None:

        if (
            row_number < 1
            or row_number > len(data)
        ):

            add_error_answer(
                f"Row {row_number} does not exist. "
                f"Available rows: 1 to {len(data)}."
            )

            return

        row = data.iloc[
            row_number - 1
        ]

        result = pd.DataFrame(
            {
                "Column": list(row.index),
                "Value": [
                    format_value(value)
                    for value in row.values
                ]
            }
        )

        add_table_answer(
            f"📌 Complete information for row {row_number}.",
            result
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

        start = max(1, start)
        end = min(len(data), end)

        if start > len(data):

            add_error_answer(
                f"Row {start} does not exist."
            )

            return

        result = data.iloc[
            start - 1:end
        ].copy()

        add_table_answer(
            f"📋 Rows {start} to {end}.",
            result
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

        add_table_answer(
            f"📋 First {n} rows.",
            data.head(n)
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

        add_table_answer(
            f"📋 Last {n} rows.",
            data.tail(n)
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

        add_table_answer(
            f"🎲 Random {n} rows.",
            data.sample(n=n)
        )

        return

    # =====================================================
    # ROW COUNT
    # =====================================================

    if (
        "how many rows" in q
        or "total rows" in q
        or "number of rows" in q
        or "row count" in q
    ):

        add_metric_answer(
            "Total Rows",
            f"{len(data):,}"
        )

        return

    # =====================================================
    # COLUMN COUNT
    # =====================================================

    if (
        "how many columns" in q
        or "total columns" in q
        or "number of columns" in q
        or "column count" in q
    ):

        add_metric_answer(
            "Total Columns",
            f"{len(columns):,}"
        )

        return

    # =====================================================
    # MISSING VALUES
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

            add_text_answer(
                "✅ There are no missing values."
            )

        else:

            add_table_answer(
                "⚠️ Missing values by column.",
                result
            )

        return

    # =====================================================
    # DUPLICATES
    # =====================================================

    if (
        "duplicate" in q
        or "duplicates" in q
    ):

        duplicate_count = int(
            data.duplicated().sum()
        )

        add_metric_answer(
            "Duplicate Rows",
            duplicate_count
        )

        return

    # =====================================================
    # FIND COLUMN
    # =====================================================

    column = find_column(
        q,
        columns
    )

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

            add_table_answer(
                f"📋 First {n} values of `{column}`.",
                data[[column]].head(n)
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

            add_table_answer(
                f"📋 Last {n} values of `{column}`.",
                data[[column]].tail(n)
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

            add_table_answer(
                f"🎲 Random {n} values of `{column}`.",
                data[[column]].sample(n=n)
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

                add_error_answer(
                    f"`{column}` is not a numeric column."
                )

                return

            values = numeric_values(series)

            if values.dropna().empty:

                add_error_answer(
                    f"No numeric values found in `{column}`."
                )

                return

            index = values.idxmax()
            highest = values.loc[index]

            wants_information = any(
                word in q
                for word in [
                    "information",
                    "info",
                    "details",
                    "detail",
                    "full row",
                    "complete row"
                ]
            )

            if wants_information:

                position = (
                    data.index.get_loc(index) + 1
                )

                row_data = data.iloc[
                    position - 1
                ]

                result = pd.DataFrame(
                    {
                        "Column": list(row_data.index),
                        "Value": [
                            format_value(v)
                            for v in row_data.values
                        ]
                    }
                )

                add_metric_answer(
                    f"Highest {column}",
                    format_value(highest),
                    f"🏆 Complete information for dataset row {position}."
                )

                add_table_answer(
                    f"Complete row information for row {position}.",
                    result
                )

            else:

                add_metric_answer(
                    f"Highest Value of {column}",
                    format_value(highest)
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

                add_error_answer(
                    f"`{column}` is not a numeric column."
                )

                return

            values = numeric_values(series)

            if values.dropna().empty:

                add_error_answer(
                    f"No numeric values found in `{column}`."
                )

                return

            index = values.idxmin()
            lowest = values.loc[index]

            wants_information = any(
                word in q
                for word in [
                    "information",
                    "info",
                    "details",
                    "detail",
                    "full row",
                    "complete row"
                ]
            )

            if wants_information:

                position = (
                    data.index.get_loc(index) + 1
                )

                row_data = data.iloc[
                    position - 1
                ]

                result = pd.DataFrame(
                    {
                        "Column": list(row_data.index),
                        "Value": [
                            format_value(v)
                            for v in row_data.values
                        ]
                    }
                )

                add_metric_answer(
                    f"Lowest {column}",
                    format_value(lowest),
                    f"📉 Complete information for dataset row {position}."
                )

                add_table_answer(
                    f"Complete row information for row {position}.",
                    result
                )

            else:

                add_metric_answer(
                    f"Lowest Value of {column}",
                    format_value(lowest)
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

                add_error_answer(
                    f"`{column}` is not numeric."
                )

                return

            value = numeric_values(
                series
            ).mean()

            add_metric_answer(
                f"Average of {column}",
                format_value(value)
            )

            return

        # =================================================
        # MEDIAN
        # =================================================

        if "median" in q:

            if not is_numeric_column(series):

                add_error_answer(
                    f"`{column}` is not numeric."
                )

                return

            value = numeric_values(
                series
            ).median()

            add_metric_answer(
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

                add_error_answer(
                    f"`{column}` is not numeric."
                )

                return

            value = numeric_values(
                series
            ).sum()

            add_metric_answer(
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

            count = int(
                series.notna().sum()
            )

            add_metric_answer(
                f"Non-Missing Values in {column}",
                count
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

            add_table_answer(
                f"Found {len(unique_values)} unique values in `{column}`.",
                result
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

                    add_error_answer(
                        f"`{column}` must be numeric for this filter."
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

                add_table_answer(
                    f"`{column}` {operator} {number} — {len(result)} rows found.",
                    result
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

            add_table_answer(
                f"Sorted `{column}` in descending order.",
                result
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

            add_table_answer(
                f"Sorted `{column}` in ascending order.",
                result
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

            result = pd.DataFrame(
                {
                    "Property": [
                        "Column",
                        "Data Type",
                        "Total Values",
                        "Non-Null",
                        "Missing",
                        "Unique"
                    ],
                    "Value": [
                        column,
                        str(series.dtype),
                        len(series),
                        int(series.notna().sum()),
                        int(series.isna().sum()),
                        int(series.nunique())
                    ]
                }
            )

            if is_numeric_column(series):

                values = numeric_values(series)

                extra = pd.DataFrame(
                    {
                        "Property": [
                            "Minimum",
                            "Maximum",
                            "Average",
                            "Median",
                            "Sum",
                            "Standard Deviation"
                        ],
                        "Value": [
                            format_value(values.min()),
                            format_value(values.max()),
                            format_value(values.mean()),
                            format_value(values.median()),
                            format_value(values.sum()),
                            format_value(values.std())
                        ]
                    }
                )

                result = pd.concat(
                    [
                        result,
                        extra
                    ],
                    ignore_index=True
                )

            add_table_answer(
                f"📌 Information about `{column}`.",
                result
            )

            return

        # =================================================
        # DEFAULT COLUMN
        # =================================================

        add_table_answer(
            f"📋 Full data of `{column}`.",
            data[[column]]
        )

        return

    # =====================================================
    # FALLBACK
    # =====================================================

    sample_columns = columns[:10]

    column_text = ", ".join(
        str(col)
        for col in sample_columns
    )

    if len(columns) > 10:
        column_text += ", ..."

    add_text_answer(
        "I could not understand that question.\n\n"
        "Available columns: "
        + column_text
        + "\n\n"
        "Try:\n"
        "• show 10 no row\n"
        "• show first 10 rows\n"
        "• show all column names\n"
        "• highest value of [column]\n"
        "• lowest value of [column]\n"
        "• average value of [column]\n"
        "• highest [column] information\n"
        "• show first 10 [column]"
    )


# =========================================================
# CSS
# =========================================================

st.markdown(
    "<style>"
    ".main-title{font-size:32px;font-weight:700;}"
    ".sub-title{opacity:0.65;margin-bottom:20px;}"
    "</style>",
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🤖 CSV Chat AI")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        current_file_id = (
            uploaded_file.name
            + "_"
            + str(uploaded_file.size)
        )

        if (
            st.session_state.last_file_id
            != current_file_id
        ):

            try:

                loaded_df = read_csv_file(
                    uploaded_file
                )

                st.session_state.df = loaded_df

                st.session_state.file_name = (
                    uploaded_file.name
                )

                st.session_state.messages = []

                st.session_state.last_file_id = (
                    current_file_id
                )

                st.rerun()

            except Exception as error:

                st.error(
                    f"Could not read CSV: {error}"
                )

    if st.session_state.df is not None:

        current_df = st.session_state.df

        st.divider()

        st.write(
            f"**File:** {st.session_state.file_name}"
        )

        st.write(
            f"**Rows:** {len(current_df):,}"
        )

        st.write(
            f"**Columns:** {len(current_df.columns):,}"
        )

        st.divider()

        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True
        ):

            st.session_state.messages = []

            st.rerun()

        st.divider()

        st.subheader(
            "⚡ Quick Commands"
        )

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


# =========================================================
# NO CSV
# =========================================================

if st.session_state.df is None:

    st.markdown(
        "<div class='main-title'>🤖 CSV Chat AI</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='sub-title'>Upload any CSV and chat with its data.</div>",
        unsafe_allow_html=True
    )

    st.info(
        "👈 Upload a CSV file from the sidebar to start."
    )

    st.subheader(
        "💡 Example"
    )

    st.code(
        "show 10 no row\n"
        "show first 10 rows\n"
        "show all column names\n"
        "highest value of Salary\n"
        "average Age"
    )

    st.stop()


# =========================================================
# DATA
# =========================================================

df = st.session_state.df.copy()


# =========================================================
# AUTO NUMERIC DETECTION
# =========================================================

for col in df.columns:

    if df[col].dtype == "object":

        converted = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        original_count = int(
            df[col].notna().sum()
        )

        if original_count > 0:

            ratio = (
                converted.notna().sum()
                / original_count
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
    "<div class='sub-title'>Ask questions about your uploaded CSV.</div>",
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
        "All previous questions and answers remain in this chat."
    )

    # -----------------------------------------------------
    # DISPLAY FULL HISTORY
    # -----------------------------------------------------

    for message in st.session_state.messages:

        render_message(
            message
        )

    # -----------------------------------------------------
    # EXAMPLES
    # -----------------------------------------------------

    with st.expander(
        "💡 Example Questions"
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
            "show Age\n"
            "show all Salary\n"
            "show first 10 Age\n"
            "show last 10 Salary\n"
            "show random 10 Age\n"
            "highest value of Salary\n"
            "lowest value of Salary\n"
            "average Salary\n"
            "median Age\n"
            "sum Salary\n"
            "count Age\n"
            "show unique City\n"
            "highest Salary information\n"
            "lowest Age information\n"
            "Salary greater than 50000\n"
            "Age less than 30\n"
            "sort Salary descending\n"
            "sort Age ascending"
        )

    # -----------------------------------------------------
    # QUICK BUTTONS
    # -----------------------------------------------------

    b1, b2, b3, b4 = st.columns(4)

    with b1:

        if st.button(
            "📋 All Columns",
            key="all_columns_chat",
            use_container_width=True
        ):

            st.session_state.quick_query = (
                "show all column names"
            )

    with b2:

        if st.button(
            "🔟 Row 10",
            key="row10_chat",
            use_container_width=True
        ):

            st.session_state.quick_query = (
                "show 10 no row"
            )

    with b3:

        if st.button(
            "📊 First 10",
            key="first10_chat",
            use_container_width=True
        ):

            st.session_state.quick_query = (
                "show first 10 rows"
            )

    with b4:

        if st.button(
            "📊 Last 10",
            key="last10_chat",
            use_container_width=True
        ):

            st.session_state.quick_query = (
                "show last 10 rows"
            )

    # -----------------------------------------------------
    # CHAT INPUT
    # -----------------------------------------------------

    typed_query = st.chat_input(
        "Ask anything about your CSV..."
    )

    quick_query = st.session_state.pop(
        "quick_query",
        None
    )

    final_query = (
        typed_query
        if typed_query
        else quick_query
    )

    # -----------------------------------------------------
    # NEW QUESTION
    # -----------------------------------------------------

    if final_query:

        add_user_message(
            final_query
        )

        process_query(
            final_query,
            df
        )

        st.rerun()


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
            n=min(
                10,
                len(df)
            )
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
            numeric_columns,
            key="visual_column"
        )

        chart_type = st.selectbox(
            "Chart Type",
            [
                "Histogram",
                "Box Plot",
                "Line Chart"
            ],
            key="chart_type"
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

            ax.plot(
                values.values
            )

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
        key="ml_target"
    )

    target = df[
        target_column
    ]

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
            subset=[
                target_column
            ]
        ).copy()

        if len(ml_data) < 10:

            st.error(
                "Not enough usable rows for ML."
            )

        else:

            X = ml_data.drop(
                columns=[
                    target_column
                ]
            )

            y = ml_data[
                target_column
            ]

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
                                pd.DataFrame(
                                    results
                                ),
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
                            pd.DataFrame(
                                results
                            ),
                            use_container_width=True,
                            hide_index=True
                        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🤖 CSV Chat AI | Persistent Chat | Dynamic CSV Analysis | Visualization | Machine Learning"
)
