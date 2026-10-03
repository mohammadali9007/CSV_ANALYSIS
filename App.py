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
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 10px;
}

.chat-header {
    padding: 18px 22px;
    border-radius: 16px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 20px;
}

.chat-header h1 {
    margin: 0;
    font-size: 30px;
}

.chat-header p {
    margin-top: 6px;
    opacity: 0.65;
}

.user-message {
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0 8px 20%;
    border: 1px solid rgba(128,128,128,0.20);
}

.ai-message {
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 20% 8px 0;
    border: 1px solid rgba(128,128,128,0.20);
}

.small-text {
    font-size: 13px;
    opacity: 0.60;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "file_name" not in st.session_state:
    st.session_state.file_name = None


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def normalize_text(text):
    """
    Convert different formats into comparable text.

    Example:
    Blood_Pressure
    Blood Pressure
    blood-pressure

    become similar.
    """

    text = str(text).strip().lower()

    text = text.replace("_", " ")
    text = text.replace("-", " ")

    text = re.sub(r"\s+", " ", text)

    return text


def compact_text(text):
    """
    Remove spaces and symbols.
    """

    text = normalize_text(text)

    return re.sub(
        r"[^a-z0-9]",
        "",
        text
    )


def format_value(value):

    if pd.isna(value):
        return "Missing"

    if isinstance(value, (float, np.floating)):

        if np.isfinite(value):

            return (
                f"{value:.4f}"
                .rstrip("0")
                .rstrip(".")
            )

    return str(value)


def is_numeric(series):

    return pd.api.types.is_numeric_dtype(series)


def numeric_series(series):

    return pd.to_numeric(
        series,
        errors="coerce"
    )


# =========================================================
# FIND COLUMN
# =========================================================

def find_column_from_text(text, columns):

    """
    Dynamically detect actual CSV column.

    No fixed column names.
    """

    text_normal = normalize_text(text)
    text_compact = compact_text(text)

    # -----------------------------------------------------
    # Exact normal match
    # -----------------------------------------------------

    for col in columns:

        col_normal = normalize_text(col)

        if col_normal == text_normal:
            return col

    # -----------------------------------------------------
    # Exact compact match
    # -----------------------------------------------------

    for col in columns:

        col_compact = compact_text(col)

        if col_compact == text_compact:
            return col

    # -----------------------------------------------------
    # Column inside query
    # Longest first
    # -----------------------------------------------------

    matches = []

    for col in columns:

        col_normal = normalize_text(col)
        col_compact = compact_text(col)

        if col_normal in text_normal:
            matches.append(col)

        elif col_compact in text_compact:
            matches.append(col)

    if matches:

        matches.sort(
            key=lambda x: len(str(x)),
            reverse=True
        )

        return matches[0]

    return None


# =========================================================
# EXTRACT COLUMN
# =========================================================

def extract_column(query, columns):

    """
    Find the real CSV column name from a natural language query.
    """

    q_normal = normalize_text(query)
    q_compact = compact_text(query)

    # Longest column first
    sorted_columns = sorted(
        columns,
        key=lambda x: len(str(x)),
        reverse=True
    )

    for col in sorted_columns:

        col_normal = normalize_text(col)
        col_compact = compact_text(col)

        # Exact phrase
        if col_normal in q_normal:

            return col

        # Compact form
        if col_compact and col_compact in q_compact:

            return col

    return None


# =========================================================
# ROW NUMBER DETECTOR
# =========================================================

def extract_single_row_number(query):

    q = normalize_text(query)

    patterns = [

        # show 10 no row
        r"\bshow\s+(\d+)\s*(?:no|number|num)\s+row\b",

        # show row 10
        r"\bshow\s+row\s+(\d+)\b",

        # show row number 10
        r"\bshow\s+row\s+number\s+(\d+)\b",

        # show 10th row
        r"\bshow\s+(\d+)(?:st|nd|rd|th)\s+row\b",

        # 10 no row
        r"\b(\d+)\s*(?:no|number|num)\s+row\b",

        # 10th row
        r"\b(\d+)(?:st|nd|rd|th)\s+row\b",

        # row 10 information
        r"\brow\s+(\d+)\s+(?:information|info|details?)\b",

        # row number 10
        r"\brow\s+number\s+(\d+)\b"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            q
        )

        if match:

            return int(
                match.group(1)
            )

    return None


# =========================================================
# ROW RANGE
# =========================================================

def extract_row_range(query):

    q = normalize_text(query)

    patterns = [

        r"\b(?:show|display|give me)?\s*rows?\s+(\d+)\s*(?:to|-)\s*(\d+)",

        r"\bshow\s+(\d+)\s*(?:to|-)\s*(\d+)\s+rows?",

        r"\b(\d+)\s*(?:to|-)\s*(\d+)\s+rows?"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            q
        )

        if match:

            return (
                int(match.group(1)),
                int(match.group(2))
            )

    return None


# =========================================================
# DISPLAY FULL ROW
# =========================================================

def display_full_row(
    row,
    row_number=None,
    title="Complete Row Information"
):

    st.markdown(
        f"### 📌 {title}"
    )

    if row_number is not None:

        st.caption(
            f"Dataset Row Number: {row_number}"
        )

    result = pd.DataFrame({
        "Column": [
            str(x)
            for x in row.index
        ],
        "Value": [
            format_value(x)
            for x in row.values
        ]
    })

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# SHOW DATAFRAME
# =========================================================

def display_dataframe(
    data,
    message=None,
    height=450
):

    if message:

        st.markdown(
            f"### 📊 {message}"
        )

    st.dataframe(
        data,
        use_container_width=True,
        height=height
    )


# =========================================================
# QUERY PROCESSOR
# =========================================================

def process_query(query, data):

    query_original = query.strip()

    q = normalize_text(
        query_original
    )

    columns = list(
        data.columns
    )

    # =====================================================
    # EMPTY QUERY
    # =====================================================

    if not q:

        st.warning(
            "Please write a question."
        )

        return


    # =====================================================
    # 1. SHOW ALL COLUMN NAMES
    # =====================================================

    column_commands = [

        "show all column name",
        "show all column names",
        "show all columns",
        "display all columns",
        "list all columns",
        "list columns",
        "column names",
        "what are the columns",
        "what are all columns",
        "give me all columns",
        "show column names"
    ]

    if any(
        command in q
        for command in column_commands
    ):

        st.markdown(
            "### 📋 All Column Names"
        )

        result = pd.DataFrame({

            "No.": range(
                1,
                len(columns) + 1
            ),

            "Column Name": columns,

            "Data Type": [
                str(data[c].dtype)
                for c in columns
            ],

            "Non-Null": [
                int(data[c].notna().sum())
                for c in columns
            ],

            "Missing": [
                int(data[c].isna().sum())
                for c in columns
            ]
        })

        st.success(
            f"Total {len(columns)} columns found."
        )

        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )

        return


    # =====================================================
    # 2. ROW RANGE
    # =====================================================

    row_range = extract_row_range(
        q
    )

    if row_range:

        start, end = row_range

        if start < 1:
            start = 1

        if end > len(data):
            end = len(data)

        if start > len(data):

            st.error(
                f"Row {start} does not exist."
            )

            return

        if start > end:

            start, end = end, start

        result = data.iloc[
            start - 1:end
        ]

        display_dataframe(
            result,
            f"Rows {start} to {end}"
        )

        return


    # =====================================================
    # 3. SINGLE ROW
    # =====================================================

    row_number = extract_single_row_number(
        q
    )

    if row_number is not None:

        if (
            row_number < 1
            or row_number > len(data)
        ):

            st.error(
                f"Row {row_number} does not exist. "
                f"Available rows: 1 to {len(data)}."
            )

            return

        row = data.iloc[
            row_number - 1
        ]

        display_full_row(
            row,
            row_number,
            f"Row {row_number} — Complete Information"
        )

        return


    # =====================================================
    # 4. FIRST N ROWS
    # =====================================================

    match = re.search(
        r"\bfirst\s+(\d+)\s+rows?\b",
        q
    )

    if match:

        n = int(
            match.group(1)
        )

        n = min(
            n,
            len(data)
        )

        display_dataframe(
            data.head(n),
            f"First {n} Rows"
        )

        return


    # =====================================================
    # 5. LAST N ROWS
    # =====================================================

    match = re.search(
        r"\blast\s+(\d+)\s+rows?\b",
        q
    )

    if match:

        n = int(
            match.group(1)
        )

        n = min(
            n,
            len(data)
        )

        display_dataframe(
            data.tail(n),
            f"Last {n} Rows"
        )

        return


    # =====================================================
    # 6. RANDOM N ROWS
    # =====================================================

    match = re.search(
        r"\brandom\s+(\d+)\s+rows?\b",
        q
    )

    if match:

        n = int(
            match.group(1)
        )

        n = min(
            n,
            len(data)
        )

        display_dataframe(
            data.sample(n=n),
            f"Random {n} Rows"
        )

        return


    # =====================================================
    # 7. DATASET QUESTIONS
    # =====================================================

    if (
        "how many rows" in q
        or "number of rows" in q
        or "total rows" in q
        or "row count" in q
    ):

        st.metric(
            "Total Rows",
            len(data)
        )

        return


    if (
        "how many columns" in q
        or "number of columns" in q
        or "total columns" in q
        or "column count" in q
    ):

        st.metric(
            "Total Columns",
            len(columns)
        )

        return


    # =====================================================
    # 8. MISSING VALUES
    # =====================================================

    if (
        "missing values" in q
        or "missing data" in q
        or "null values" in q
        or "null data" in q
    ):

        missing = pd.DataFrame({

            "Column": columns,

            "Missing": [
                int(data[c].isna().sum())
                for c in columns
            ]
        })

        missing = missing[
            missing["Missing"] > 0
        ]

        if missing.empty:

            st.success(
                "No missing values found."
            )

        else:

            display_dataframe(
                missing,
                "Missing Values"
            )

        return


    # =====================================================
    # 9. DUPLICATES
    # =====================================================

    if (
        "duplicate rows" in q
        or "duplicates" in q
        or "duplicate data" in q
    ):

        duplicate_data = data[
            data.duplicated(
                keep=False
            )
        ]

        st.metric(
            "Duplicate Rows",
            len(duplicate_data)
        )

        if len(duplicate_data) > 0:

            display_dataframe(
                duplicate_data,
                "Duplicate Rows"
            )

        return


    # =====================================================
    # 10. DETECT COLUMN
    # =====================================================

    column = extract_column(
        q,
        columns
    )


    # =====================================================
    # 11. COLUMN COMMANDS
    # =====================================================

    if column:

        series = data[column]


        # =================================================
        # SHOW COLUMN
        # =================================================

        # Remove generic command words
        command_words = [
            "show",
            "display",
            "give me",
            "get",
            "what is",
            "what are",
            "tell me",
            "please"
        ]

        cleaned_query = q

        for word in command_words:

            cleaned_query = cleaned_query.replace(
                word,
                ""
            )

        cleaned_query = cleaned_query.strip()


        # =================================================
        # FIRST N COLUMN DATA
        # =================================================

        first_match = re.search(
            r"\bfirst\s+(\d+)\b",
            q
        )

        if (
            first_match
            and "row" not in q
            and "rows" not in q
        ):

            n = int(
                first_match.group(1)
            )

            n = min(
                n,
                len(data)
            )

            result = data[
                [column]
            ].head(n)

            display_dataframe(
                result,
                f"First {n} Values of `{column}`"
            )

            return


        # =================================================
        # LAST N COLUMN DATA
        # =================================================

        last_match = re.search(
            r"\blast\s+(\d+)\b",
            q
        )

        if (
            last_match
            and "row" not in q
            and "rows" not in q
        ):

            n = int(
                last_match.group(1)
            )

            n = min(
                n,
                len(data)
            )

            result = data[
                [column]
            ].tail(n)

            display_dataframe(
                result,
                f"Last {n} Values of `{column}`"
            )

            return


        # =================================================
        # RANDOM N COLUMN DATA
        # =================================================

        random_match = re.search(
            r"\brandom\s+(\d+)\b",
            q
        )

        if (
            random_match
            and "row" not in q
            and "rows" not in q
        ):

            n = int(
                random_match.group(1)
            )

            n = min(
                n,
                len(data)
            )

            result = data[
                [column]
            ].sample(
                n=n
            )

            display_dataframe(
                result,
                f"Random {n} Values of `{column}`"
            )

            return


        # =================================================
        # HIGHEST VALUE
        # =================================================

        highest_words = [
            "highest",
            "maximum",
            "max value",
            "largest",
            "highest value"
        ]

        if any(
            word in q
            for word in highest_words
        ):

            if not is_numeric(series):

                st.warning(
                    f"`{column}` is not a numeric column."
                )

                return

            numeric = numeric_series(
                series
            )

            if numeric.dropna().empty:

                st.warning(
                    f"No numeric data found in `{column}`."
                )

                return

            idx = numeric.idxmax()

            highest_value = numeric.loc[idx]

            st.metric(
                f"Highest Value of {column}",
                format_value(
                    highest_value
                )
            )

            # Full row if information/details requested
            if any(
                word in q
                for word in [
                    "information",
                    "info",
                    "details",
                    "detail",
                    "full row",
                    "complete row",
                    "complete information"
                ]
            ):

                row_position = (
                    data.index.get_loc(idx) + 1
                )

                display_full_row(
                    data.loc[idx],
                    row_position,
                    f"Complete Row — Highest `{column}`"
                )

            return


        # =================================================
        # LOWEST VALUE
        # =================================================

        lowest_words = [
            "lowest",
            "minimum",
            "min value",
            "smallest",
            "lowest value"
        ]

        if any(
            word in q
            for word in lowest_words
        ):

            if not is_numeric(series):

                st.warning(
                    f"`{column}` is not a numeric column."
                )

                return

            numeric = numeric_series(
                series
            )

            if numeric.dropna().empty:

                st.warning(
                    f"No numeric data found in `{column}`."
                )

                return

            idx = numeric.idxmin()

            lowest_value = numeric.loc[idx]

            st.metric(
                f"Lowest Value of {column}",
                format_value(
                    lowest_value
                )
            )

            if any(
                word in q
                for word in [
                    "information",
                    "info",
                    "details",
                    "detail",
                    "full row",
                    "complete row",
                    "complete information"
                ]
            ):

                row_position = (
                    data.index.get_loc(idx) + 1
                )

                display_full_row(
                    data.loc[idx],
                    row_position,
                    f"Complete Row — Lowest `{column}`"
                )

            return


        # =================================================
        # AVERAGE
        # =================================================

        if (
            "average" in q
            or "avg" in q
            or "mean" in q
            or "average value" in q
        ):

            if not is_numeric(series):

                st.warning(
                    f"`{column}` is not numeric."
                )

                return

            numeric = numeric_series(
                series
            )

            value = numeric.mean()

            st.metric(
                f"Average Value of {column}",
                format_value(value)
            )

            return


        # =================================================
        # MEDIAN
        # =================================================

        if "median" in q:

            if not is_numeric(series):

                st.warning(
                    f"`{column}` is not numeric."
                )

                return

            value = numeric_series(
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

            if not is_numeric(series):

                st.warning(
                    f"`{column}` is not numeric."
                )

                return

            value = numeric_series(
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
            or "how many" in q
            or "number of values" in q
        ):

            st.metric(
                f"Non-Missing Values of {column}",
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

            st.success(
                f"{len(unique_values)} unique values found."
            )

            result = pd.DataFrame({

                column:
                    unique_values

            })

            display_dataframe(
                result,
                f"Unique Values of `{column}`"
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

            display_dataframe(
                result,
                f"`{column}` Sorted Descending"
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

            display_dataframe(
                result,
                f"`{column}` Sorted Ascending"
            )

            return


        # =================================================
        # GREATER / LESS / EQUAL
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

                if not is_numeric(series):

                    st.warning(
                        f"`{column}` must be numeric for this operation."
                    )

                    return

                number = float(
                    match.group(1)
                )

                numeric = numeric_series(
                    series
                )

                if operator == ">":

                    mask = numeric > number

                elif operator == "<":

                    mask = numeric < number

                elif operator == "==":

                    mask = numeric == number

                elif operator == ">=":

                    mask = numeric >= number

                else:

                    mask = numeric <= number

                result = data[
                    mask
                ]

                display_dataframe(
                    result,
                    f"{column} {operator} {number} — {len(result)} Rows"
                )

                return


        # =================================================
        # COLUMN INFORMATION
        # =================================================

        information_words = [
            "information",
            "info",
            "details",
            "detail",
            "about"
        ]

        if any(
            word in q
            for word in information_words
        ):

            st.markdown(
                f"### 📌 Information about `{column}`"
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

            if is_numeric(series):

                numeric = numeric_series(
                    series
                )

                st.divider()

                c1, c2, c3, c4, c5 = st.columns(5)

                c1.metric(
                    "Min",
                    format_value(
                        numeric.min()
                    )
                )

                c2.metric(
                    "Max",
                    format_value(
                        numeric.max()
                    )
                )

                c3.metric(
                    "Average",
                    format_value(
                        numeric.mean()
                    )
                )

                c4.metric(
                    "Median",
                    format_value(
                        numeric.median()
                    )
                )

                c5.metric(
                    "Sum",
                    format_value(
                        numeric.sum()
                    )
                )

            return


        # =================================================
        # DEFAULT SHOW COLUMN
        # =================================================

        # If user simply says:
        # show Age
        # show Product
        # give me Revenue
        # display ID

        if (
            q.startswith("show ")
            or q.startswith("display ")
            or q.startswith("give me ")
            or q.startswith("get ")
            or q.startswith("what is ")
            or q.startswith("what are ")
            or q == normalize_text(column)
        ):

            result = data[
                [column]
            ]

            display_dataframe(
                result,
                f"Full Data of `{column}`"
            )

            return


        # =================================================
        # DIRECT COLUMN NAME
        # =================================================

        result = data[
            [column]
        ]

        display_dataframe(
            result,
            f"Data of `{column}`"
        )

        return


    # =====================================================
    # 12. GENERAL FALLBACK
    # =====================================================

    st.warning(
        "I couldn't understand this command."
    )

    st.markdown(
        "### 💡 Try these formats"
    )

    st.code("""
show 10 no row
show row 10
show 10th row

show 5 to 10 rows

show first 10 rows
show last 10 rows
show random 10 rows

show all column names

show [column]
show all [column]

show first 10 [column]
show last 10 [column]
show random 10 [column]

highest value of [column]
lowest value of [column]
average value of [column]
median [column]
sum [column]
count [column]

highest [column] information
lowest [column] information

show unique [column]

[column] greater than 50
[column] less than 50

sort [column] ascending
sort [column] descending
""")


# =========================================================
# LOAD CSV
# =========================================================

with st.sidebar:

    st.markdown("## 📂 Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(
                uploaded_file
            )

            # Clean column names
            df.columns = [
                str(c).strip()
                for c in df.columns
            ]

            st.session_state.df = df

            if (
                st.session_state.file_name
                != uploaded_file.name
            ):

                st.session_state.chat_history = []

                st.session_state.file_name = (
                    uploaded_file.name
                )

        except Exception as e:

            st.error(
                f"Could not read CSV: {e}"
            )

            st.stop()


# =========================================================
# MAIN
# =========================================================

if st.session_state.df is None:

    st.markdown("""
    ## 🤖 CSV Chat AI

    Upload any CSV file and chat with your dataset.

    ### You can ask:

    - `show 10 no row`
    - `show first 10 rows`
    - `show all column names`
    - `show all X`
    - `show first 10 X`
    - `show last 10 X`
    - `show random 10 X`
    - `highest value of X`
    - `lowest value of X`
    - `average value of X`
    - `highest X information`
    - `X greater than 50`
    - `sort X descending`

    **X automatically means the actual column name in your CSV.**
    """)

    st.stop()


# =========================================================
# DATA
# =========================================================

df = st.session_state.df.copy()


# =========================================================
# CONVERT NUMERIC-LOOKING COLUMNS
# =========================================================

for col in df.columns:

    if df[col].dtype == "object":

        converted = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        original_count = (
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

st.markdown("""
<div class="chat-header">

<h1>🤖 CSV Chat AI</h1>

<p>
Chat with your CSV • Ask questions • Explore rows •
Analyze columns • Statistics • ML
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# TOP INFO
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

chat_tab, data_tab, visual_tab, ml_tab = st.tabs(
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

    st.markdown(
        "### 💬 Chat with your CSV"
    )

    st.caption(
        "Ask questions naturally using your actual CSV column names."
    )


    # =====================================================
    # CHAT HISTORY
    # =====================================================

    for item in st.session_state.chat_history:

        if item["role"] == "user":

            st.markdown(
                f"""
                <div class="user-message">
                <b>You</b><br>
                {item["message"]}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="ai-message">
                <b>🤖 CSV AI</b><br>
                {item["message"]}
                </div>
                """,
                unsafe_allow_html=True
            )

            # Show result after message
            if item.get("result") is not None:

                if item["result_type"] == "dataframe":

                    st.dataframe(
                        item["result"],
                        use_container_width=True,
                        height=400
                    )

                elif item["result_type"] == "metric":

                    st.metric(
                        item["title"],
                        item["result"]
                    )


    # =====================================================
    # QUICK COMMANDS
    # =====================================================

    st.markdown(
        "#### ⚡ Quick Commands"
    )

    columns = list(
        df.columns
    )

    if columns:

        example_col = columns[0]

        q1, q2, q3, q4 = st.columns(4)

        with q1:

            if st.button(
                "📋 All Columns",
                use_container_width=True
            ):

                st.session_state.pending_query = (
                    "show all column names"
                )

        with q2:

            if st.button(
                "🔟 Row 10",
                use_container_width=True
            ):

                st.session_state.pending_query = (
                    "show 10 no row"
                )

        with q3:

            if st.button(
                "📊 First 10 Rows",
                use_container_width=True
            ):

                st.session_state.pending_query = (
                    "show first 10 rows"
                )

        with q4:

            if st.button(
                "📋 Show Column",
                use_container_width=True
            ):

                st.session_state.pending_query = (
                    f"show {example_col}"
                )


    # =====================================================
    # CHAT INPUT
    # =====================================================

    pending_query = st.session_state.get(
        "pending_query",
        ""
    )

    user_query = st.chat_input(
        "Ask anything about your CSV..."
    )


    if pending_query:

        user_query = pending_query

        st.session_state.pending_query = ""


    if user_query:

        # Save user message
        st.session_state.chat_history.append({
            "role": "user",
            "message": user_query
        })

        # AI message
        st.session_state.chat_history.append({
            "role": "assistant",
            "message": "Here is what I found:"
        })

        # Process
        process_query(
            user_query,
            df
        )

        # Rerun to display chat
        st.rerun()


    # =====================================================
    # HELP
    # =====================================================

    with st.expander(
        "💡 What can I ask?"
    ):

        st.markdown("""
### Row Questions

```text
show 10 no row
show row 10
show 10th row
show 10 no row information

show 5 to 10 rows

show first 10 rows
show last 10 rows
show random 10 rows
