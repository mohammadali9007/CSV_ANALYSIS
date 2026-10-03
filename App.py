import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

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
    page_title="CSV Master AI",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
}

.hero {
    padding: 28px;
    border-radius: 20px;
    border: 1px solid rgba(128,128,128,.25);
    margin-bottom: 25px;
}

.hero h1 {
    margin-bottom: 5px;
}

.hero p {
    opacity: .7;
    font-size: 16px;
}

.query-card {
    padding: 20px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,.25);
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">

<h1>📊 CSV Master AI</h1>

<p>
Dynamic CSV Analysis • Smart Query • Statistics •
Row & Column Explorer • Visualization • Machine Learning
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "df" not in st.session_state:
    st.session_state.df = None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📂 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    st.divider()

    st.markdown("### 💡 Query Examples")

    st.code("""
show row 10

show rows 5 to 15

first 10 rows

last 20 rows

random 15 rows

show all [column]

first 15 [column]

last 20 [column]

random 10 [column]

highest [column]

lowest [column]

average [column]

median [column]

sum [column]

highest [column] information

lowest [column] information

show unique [column]

count [column]

[column] greater than 50

[column] less than 20

sort [column] ascending

sort [column] descending
""")


# =========================================================
# LOAD CSV
# =========================================================

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        # Clean column names
        df.columns = [
            str(col).strip()
            for col in df.columns
        ]

        st.session_state.df = df

    except Exception as e:

        st.error(
            f"CSV file could not be read: {e}"
        )

        st.stop()


# =========================================================
# NO FILE
# =========================================================

if st.session_state.df is None:

    st.info(
        "👈 Upload a CSV file from the sidebar to start."
    )

    st.markdown("""
### 🚀 Dynamic CSV Query System

এই app-এ কোনো fixed column name নেই।

যেমন CSV-তে যদি থাকে:

`Age, Glucose, BMI, Salary`

তাহলে তুমি লিখতে পারবে:

`highest Glucose`

`average BMI`

`last 20 Age`

আবার অন্য CSV-তে যদি থাকে:

`Price, Sales, Revenue, Product`

তাহলে একই app-এ:

`highest Price`

`average Revenue`

`first 15 Product`

কাজ করবে।
""")

    st.stop()


# =========================================================
# COPY DATAFRAME
# =========================================================

df = st.session_state.df.copy()


# =========================================================
# TRY CONVERT NUMERIC COLUMNS
# =========================================================

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
# UTILITY FUNCTIONS
# =========================================================

def normalize_text(text):

    """
    Makes text easier to compare.

    Example:

    Blood Pressure
    blood_pressure
    Blood-Pressure

    will become similar.
    """

    text = str(text).lower().strip()

    text = text.replace("_", " ")
    text = text.replace("-", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


def compact_text(text):

    text = normalize_text(text)

    return re.sub(
        r"[^a-z0-9]",
        "",
        text
    )


def find_column(query, columns):

    """
    Dynamically find column from CSV.
    No fixed column name is used.
    """

    q = normalize_text(query)
    q_compact = compact_text(query)

    # Exact match
    for col in columns:

        if normalize_text(col) == q:
            return col

    # Compact exact match
    for col in columns:

        if compact_text(col) == q_compact:
            return col

    # Column name appearing in query
    matches = []

    for col in columns:

        col_normal = normalize_text(col)
        col_compact = compact_text(col)

        if (
            col_normal in q
            or col_compact in q_compact
        ):

            matches.append(col)

    if matches:

        # Longest match first
        matches.sort(
            key=lambda x: len(str(x)),
            reverse=True
        )

        return matches[0]

    return None


def format_value(value):

    if pd.isna(value):

        return "Missing"

    if isinstance(
        value,
        (float, np.floating)
    ):

        if np.isfinite(value):

            return f"{value:.4f}".rstrip("0").rstrip(".")

    return str(value)


def show_complete_row(
    row,
    title="Complete Row Information"
):

    st.success(title)

    result = pd.DataFrame({
        "Column": row.index.astype(str),
        "Value": [
            format_value(v)
            for v in row.values
        ]
    })

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True
    )


def is_numeric(series):

    return pd.api.types.is_numeric_dtype(
        series
    )


def clean_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce"
    )


# =========================================================
# SMART QUERY ENGINE
# =========================================================

def process_query(query, data):

    original_query = query.strip()

    if not original_query:

        st.warning(
            "Please enter a question."
        )

        return

    q = normalize_text(
        original_query
    )

    columns = list(data.columns)

    # =====================================================
    # 1. SPECIFIC ROW RANGE
    # =====================================================

    range_patterns = [

        r"(?:show|display|give me)?\s*rows?\s+(\d+)\s*(?:to|-)\s*(\d+)",

        r"(?:show|display|give me)?\s*(\d+)\s*(?:to|-)\s*(\d+)\s*rows?"
    ]

    for pattern in range_patterns:

        match = re.search(
            pattern,
            q
        )

        if match:

            start = int(
                match.group(1)
            )

            end = int(
                match.group(2)
            )

            if start < 1:
                start = 1

            if end > len(data):
                end = len(data)

            if start > len(data):

                st.error(
                    f"Row {start} does not exist."
                )

                return

            result = data.iloc[
                start - 1:end
            ]

            st.success(
                f"Showing rows {start} to {end}"
            )

            st.dataframe(
                result,
                use_container_width=True,
                height=450
            )

            return

    # =====================================================
    # 2. SPECIFIC ROW
    # =====================================================

    row_patterns = [

        r"show row(?: number)?\s+(\d+)",

        r"display row(?: number)?\s+(\d+)",

        r"give me row(?: number)?\s+(\d+)",

        r"row(?: number)?\s+(\d+)(?: information)?$",

        r"show me row(?: number)?\s+(\d+)"
    ]

    for pattern in row_patterns:

        match = re.search(
            pattern,
            q
        )

        if match:

            row_number = int(
                match.group(1)
            )

            if (
                row_number < 1
                or row_number > len(data)
            ):

                st.error(
                    f"Row {row_number} does not exist. "
                    f"Available rows: 1 - {len(data)}"
                )

                return

            row = data.iloc[
                row_number - 1
            ]

            show_complete_row(
                row,
                f"📌 Complete Information of Row {row_number}"
            )

            return

    # =====================================================
    # 3. FIRST N ROWS
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

        st.success(
            f"First {n} rows"
        )

        st.dataframe(
            data.head(n),
            use_container_width=True,
            height=450
        )

        return

    # =====================================================
    # 4. LAST N ROWS
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

        st.success(
            f"Last {n} rows"
        )

        st.dataframe(
            data.tail(n),
            use_container_width=True,
            height=450
        )

        return

    # =====================================================
    # 5. RANDOM N ROWS
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

        st.success(
            f"Random {n} rows"
        )

        st.dataframe(
            data.sample(n=n),
            use_container_width=True,
            height=450
        )

        return

    # =====================================================
    # 6. LIST ALL COLUMN NAMES
    # =====================================================

    if (
        "show columns" in q
        or "list columns" in q
        or "column names" in q
        or "what are the columns" in q
        or "all columns" in q
    ):

        st.success(
            f"Total Columns: {len(columns)}"
        )

        column_df = pd.DataFrame({
            "No.": range(
                1,
                len(columns) + 1
            ),
            "Column Name": columns,
            "Data Type": [
                str(data[c].dtype)
                for c in columns
            ],
            "Missing": [
                int(data[c].isna().sum())
                for c in columns
            ],
            "Unique": [
                int(data[c].nunique())
                for c in columns
            ]
        })

        st.dataframe(
            column_df,
            use_container_width=True,
            hide_index=True
        )

        return

    # =====================================================
    # 7. DATASET SIZE
    # =====================================================

    if (
        "how many rows" in q
        or "number of rows" in q
        or "total rows" in q
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
    ):

        st.metric(
            "Total Columns",
            len(columns)
        )

        return

    # =====================================================
    # FIND DYNAMIC COLUMN
    # =====================================================

    column = find_column(
        q,
        columns
    )

    # =====================================================
    # IF COLUMN FOUND
    # =====================================================

    if column:

        series = data[column]

        # -------------------------------------------------
        # SHOW ALL COLUMN
        # -------------------------------------------------

        if (
            "show all" in q
            or "display all" in q
            or "all values" in q
            or q.startswith(
                "all "
            )
        ):

            st.success(
                f"All values of `{column}`"
            )

            result = pd.DataFrame({
                column: data[column]
            })

            st.dataframe(
                result,
                use_container_width=True,
                height=500
            )

            return

        # -------------------------------------------------
        # FIRST N COLUMN VALUES
        # -------------------------------------------------

        match = re.search(
            r"\bfirst\s+(\d+)\b",
            q
        )

        if (
            match
            and "row" not in q
            and "rows" not in q
        ):

            n = int(
                match.group(1)
            )

            n = min(
                n,
                len(data)
            )

            st.success(
                f"First {n} values of `{column}`"
            )

            result = data[
                [column]
            ].head(n)

            st.dataframe(
                result,
                use_container_width=True
            )

            return

        # -------------------------------------------------
        # LAST N COLUMN VALUES
        # -------------------------------------------------

        match = re.search(
            r"\blast\s+(\d+)\b",
            q
        )

        if (
            match
            and "row" not in q
            and "rows" not in q
        ):

            n = int(
                match.group(1)
            )

            n = min(
                n,
                len(data)
            )

            st.success(
                f"Last {n} values of `{column}`"
            )

            result = data[
                [column]
            ].tail(n)

            st.dataframe(
                result,
                use_container_width=True
            )

            return

        # -------------------------------------------------
        # RANDOM N COLUMN VALUES
        # -------------------------------------------------

        match = re.search(
            r"\brandom\s+(\d+)\b",
            q
        )

        if (
            match
            and "row" not in q
            and "rows" not in q
        ):

            n = int(
                match.group(1)
            )

            n = min(
                n,
                len(data)
            )

            st.success(
                f"Random {n} values of `{column}`"
            )

            result = data[
                [column]
            ].sample(
                n=n
            )

            st.dataframe(
                result,
                use_container_width=True
            )

            return

        # -------------------------------------------------
        # HIGHEST
        # -------------------------------------------------

        if any(
            word in q
            for word in [
                "highest",
                "maximum",
                "largest",
                "max"
            ]
        ):

            if not is_numeric(series):

                st.warning(
                    f"`{column}` is not numeric. "
                    "Highest value cannot be calculated."
                )

                return

            numeric_series = clean_numeric(
                series
            )

            if numeric_series.dropna().empty:

                st.warning(
                    f"No numeric values found in `{column}`."
                )

                return

            idx = numeric_series.idxmax()

            value = numeric_series.loc[idx]

            st.metric(
                f"Highest {column}",
                format_value(value)
            )

            if any(
                word in q
                for word in [
                    "information",
                    "details",
                    "full row",
                    "complete",
                    "who"
                ]
            ):

                show_complete_row(
                    data.loc[idx],
                    f"🏆 Complete Row — Highest {column}"
                )

            return

        # -------------------------------------------------
        # LOWEST
        # -------------------------------------------------

        if any(
            word in q
            for word in [
                "lowest",
                "minimum",
                "smallest",
                "min"
            ]
        ):

            if not is_numeric(series):

                st.warning(
                    f"`{column}` is not numeric. "
                    "Lowest value cannot be calculated."
                )

                return

            numeric_series = clean_numeric(
                series
            )

            if numeric_series.dropna().empty:

                st.warning(
                    f"No numeric values found in `{column}`."
                )

                return

            idx = numeric_series.idxmin()

            value = numeric_series.loc[idx]

            st.metric(
                f"Lowest {column}",
                format_value(value)
            )

            if any(
                word in q
                for word in [
                    "information",
                    "details",
                    "full row",
                    "complete",
                    "who"
                ]
            ):

                show_complete_row(
                    data.loc[idx],
                    f"📉 Complete Row — Lowest {column}"
                )

            return

        # -------------------------------------------------
        # AVERAGE
        # -------------------------------------------------

        if (
            "average" in q
            or "avg" in q
            or "mean" in q
        ):

            if not is_numeric(series):

                st.warning(
                    f"`{column}` is not numeric. "
                    "Average cannot be calculated."
                )

                return

            numeric_series = clean_numeric(
                series
            )

            value = numeric_series.mean()

            st.metric(
                f"Average {column}",
                format_value(value)
            )

            return

        # -------------------------------------------------
        # MEDIAN
        # -------------------------------------------------

        if "median" in q:

            if not is_numeric(series):

                st.warning(
                    f"`{column}` is not numeric."
                )

                return

            value = clean_numeric(
                series
            ).median()

            st.metric(
                f"Median {column}",
                format_value(value)
            )

            return

        # -------------------------------------------------
        # SUM
        # -------------------------------------------------

        if (
            "sum" in q
            or "total" in q
        ):

            if not is_numeric(series):

                st.warning(
                    f"`{column}` is not numeric."
                )

                return

            value = clean_numeric(
                series
            ).sum()

            st.metric(
                f"Total {column}",
                format_value(value)
            )

            return

        # -------------------------------------------------
        # COUNT
        # -------------------------------------------------

        if (
            "count" in q
            or "how many" in q
        ):

            st.metric(
                f"Non-empty values in {column}",
                int(series.notna().sum())
            )

            return

        # -------------------------------------------------
        # UNIQUE
        # -------------------------------------------------

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
                f"{len(unique_values)} unique values"
            )

            result = pd.DataFrame({
                column: unique_values
            })

            st.dataframe(
                result,
                use_container_width=True,
                height=500
            )

            return

        # -------------------------------------------------
        # SORT DESCENDING
        # -------------------------------------------------

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

            st.success(
                f"Sorted `{column}` descending"
            )

            st.dataframe(
                result,
                use_container_width=True,
                height=500
            )

            return

        # -------------------------------------------------
        # SORT ASCENDING
        # -------------------------------------------------

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

            st.success(
                f"Sorted `{column}` ascending"
            )

            st.dataframe(
                result,
                use_container_width=True,
                height=500
            )

            return

        # -------------------------------------------------
        # FILTER
        # -------------------------------------------------

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
                        f"`{column}` must be numeric "
                        "for this filter."
                    )

                    return

                number = float(
                    match.group(1)
                )

                numeric_series = clean_numeric(
                    series
                )

                if operator == ">":

                    result = data[
                        numeric_series > number
                    ]

                elif operator == "<":

                    result = data[
                        numeric_series < number
                    ]

                elif operator == "==":

                    result = data[
                        numeric_series == number
                    ]

                elif operator == ">=":

                    result = data[
                        numeric_series >= number
                    ]

                else:

                    result = data[
                        numeric_series <= number
                    ]

                st.success(
                    f"Found {len(result)} matching rows."
                )

                st.dataframe(
                    result,
                    use_container_width=True,
                    height=500
                )

                return

        # -------------------------------------------------
        # COLUMN INFORMATION
        # -------------------------------------------------

        if (
            "information" in q
            or "details" in q
            or "info" in q
        ):

            st.subheader(
                f"📌 Information about `{column}`"
            )

            st.write(
                f"**Data Type:** `{series.dtype}`"
            )

            st.write(
                f"**Total Values:** {len(series)}"
            )

            st.write(
                f"**Non-Null:** {series.notna().sum()}"
            )

            st.write(
                f"**Missing:** {series.isna().sum()}"
            )

            st.write(
                f"**Unique:** {series.nunique()}"
            )

            if is_numeric(series):

                numeric_series = clean_numeric(
                    series
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Minimum",
                    format_value(
                        numeric_series.min()
                    )
                )

                c2.metric(
                    "Maximum",
                    format_value(
                        numeric_series.max()
                    )
                )

                c3.metric(
                    "Average",
                    format_value(
                        numeric_series.mean()
                    )
                )

                c4.metric(
                    "Median",
                    format_value(
                        numeric_series.median()
                    )
                )

            st.subheader(
                "📋 Values"
            )

            st.dataframe(
                data[[column]],
                use_container_width=True,
                height=500
            )

            return

    # =====================================================
    # DUPLICATES
    # =====================================================

    if "duplicate" in q:

        duplicates = data[
            data.duplicated(
                keep=False
            )
        ]

        st.success(
            f"Duplicate rows found: {len(duplicates)}"
        )

        if len(duplicates) > 0:

            st.dataframe(
                duplicates,
                use_container_width=True
            )

        return

    # =====================================================
    # MISSING VALUES
    # =====================================================

    if (
        "missing values" in q
        or "missing data" in q
        or "null values" in q
    ):

        missing = pd.DataFrame({
            "Column": columns,
            "Missing Values": [
                int(data[c].isna().sum())
                for c in columns
            ]
        })

        missing = missing[
            missing["Missing Values"] > 0
        ]

        if missing.empty:

            st.success(
                "No missing values found."
            )

        else:

            st.dataframe(
                missing,
                use_container_width=True,
                hide_index=True
            )

        return

    # =====================================================
    # FALLBACK
    # =====================================================

    st.warning(
        "I couldn't understand the query."
    )

    st.info(
        "Use one of the supported patterns shown below."
    )


# =========================================================
# TABS
# =========================================================

tab_overview, tab_explore, tab_query, tab_visual, tab_ml = st.tabs(
    [
        "📊 Overview",
        "🔎 Column Explorer",
        "💬 Ask CSV",
        "📈 Visualization",
        "🤖 Machine Learning"
    ]
)


# =========================================================
# OVERVIEW TAB
# =========================================================

with tab_overview:

    st.subheader(
        "📊 Dataset Overview"
    )

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

    st.divider()

    st.subheader(
        "📋 Data Preview"
    )

    preview = st.radio(
        "Choose preview",
        [
            "First 10",
            "Last 10",
            "Random 10",
            "All"
        ],
        horizontal=True
    )

    if preview == "First 10":

        result = df.head(10)

    elif preview == "Last 10":

        result = df.tail(10)

    elif preview == "Random 10":

        result = df.sample(
            n=min(10, len(df))
        )

    else:

        result = df

    st.dataframe(
        result,
        use_container_width=True,
        height=500
    )

    st.divider()

    st.subheader(
        "🧾 All Column Information"
    )

    info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [
            str(df[c].dtype)
            for c in df.columns
        ],
        "Total": [
            len(df[c])
            for c in df.columns
        ],
        "Non-Null": [
            int(df[c].notna().sum())
            for c in df.columns
        ],
        "Missing": [
            int(df[c].isna().sum())
            for c in df.columns
        ],
        "Unique": [
            int(df[c].nunique())
            for c in df.columns
        ]
    })

    st.dataframe(
        info,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# COLUMN EXPLORER
# =========================================================

with tab_explore:

    st.subheader(
        "🔎 Specific Column Explorer"
    )

    selected_column = st.selectbox(
        "Select any column",
        df.columns
    )

    series = df[selected_column]

    st.markdown(
        f"### 📌 `{selected_column}`"
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

        st.divider()

        numeric = clean_numeric(
            series
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "Minimum",
            format_value(
                numeric.min()
            )
        )

        c2.metric(
            "Maximum",
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

    st.divider()

    display_type = st.selectbox(
        "Show",
        [
            "All Values",
            "First N",
            "Last N",
            "Random N"
        ]
    )

    if display_type == "All Values":

        result = df[
            [selected_column]
        ]

    else:

        n = st.number_input(
            "N",
            min_value=1,
            max_value=len(df),
            value=min(10, len(df))
        )

        if display_type == "First N":

            result = df[
                [selected_column]
            ].head(n)

        elif display_type == "Last N":

            result = df[
                [selected_column]
            ].tail(n)

        else:

            result = df[
                [selected_column]
            ].sample(
                n=min(n, len(df))
            )

    st.dataframe(
        result,
        use_container_width=True,
        height=450
    )

    st.divider()

    st.subheader(
        "🔢 Value Frequency"
    )

    frequency = (
        series
        .value_counts(
            dropna=False
        )
        .reset_index()
    )

    frequency.columns = [
        selected_column,
        "Count"
    ]

    st.dataframe(
        frequency.head(100),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# QUERY TAB
# =========================================================

with tab_query:

    st.subheader(
        "💬 Ask Your CSV"
    )

    st.caption(
        "Write your question using the actual column names of your CSV."
    )

    query = st.text_input(
        "Your question",
        placeholder=(
            "Example: highest Glucose information"
        )
    )

    if st.button(
        "🔍 Run Query",
        type="primary",
        use_container_width=True
    ):

        process_query(
            query,
            df
        )

    st.divider()

    st.subheader(
        "✨ Supported Examples"
    )

    examples = [
        "show row 10",
        "show rows 5 to 15",
        "first 10 rows",
        "last 20 rows",
        "random 15 rows",
        "show all [column]",
        "first 15 [column]",
        "last 20 [column]",
        "random 10 [column]",
        "highest [column]",
        "lowest [column]",
        "average [column]",
        "median [column]",
        "sum [column]",
        "highest [column] information",
        "lowest [column] information",
        "show unique [column]",
        "[column] greater than 50",
        "[column] less than 20",
        "sort [column] descending",
        "sort [column] ascending"
    ]

    for item in examples:

        st.code(item)


# =========================================================
# VISUALIZATION
# =========================================================

with tab_visual:

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

        chart_column = st.selectbox(
            "Select numeric column",
            numeric_columns
        )

        chart_type = st.selectbox(
            "Chart",
            [
                "Histogram",
                "Box Plot",
                "Line Chart"
            ]
        )

        values = clean_numeric(
            df[chart_column]
        ).dropna()

        fig, ax = plt.subplots()

        if chart_type == "Histogram":

            ax.hist(values)

            ax.set_title(
                f"Distribution of {chart_column}"
            )

            ax.set_xlabel(
                chart_column
            )

            ax.set_ylabel(
                "Frequency"
            )

        elif chart_type == "Box Plot":

            ax.boxplot(values)

            ax.set_title(
                f"Box Plot of {chart_column}"
            )

            ax.set_ylabel(
                chart_column
            )

        else:

            ax.plot(
                values.values
            )

            ax.set_title(
                f"{chart_column} by Row"
            )

            ax.set_xlabel(
                "Row"
            )

            ax.set_ylabel(
                chart_column
            )

        st.pyplot(
            fig,
            clear_figure=True
        )


# =========================================================
# MACHINE LEARNING
# =========================================================

with tab_ml:

    st.subheader(
        "🤖 Machine Learning"
    )

    st.info(
        "Select any column as the target. "
        "The remaining columns will be used as features."
    )

    target_column = st.selectbox(
        "🎯 Target Column",
        df.columns
    )

    target = df[
        target_column
    ]

    unique_values = target.nunique(
        dropna=True
    )

    if (
        not is_numeric(target)
        or unique_values <= 20
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
        type="primary"
    ):

        ml_df = df.copy()

        ml_df = ml_df.dropna(
            subset=[
                target_column
            ]
        )

        if len(ml_df) < 10:

            st.error(
                "Not enough rows for ML."
            )

            st.stop()

        X = ml_df.drop(
            columns=[
                target_column
            ]
        )

        y = ml_df[
            target_column
        ]

        # Remove completely empty columns
        X = X.dropna(
            axis=1,
            how="all"
        )

        if X.shape[1] == 0:

            st.error(
                "No usable feature columns found."
            )

            st.stop()

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

        # =================================================
        # CLASSIFICATION
        # =================================================

        if problem_type == "Classification":

            if y.nunique() < 2:

                st.error(
                    "Target needs at least 2 classes."
                )

                st.stop()

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=0.20,
                    random_state=42,
                    stratify=y
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

                    results.append({

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
                    })

                except Exception as e:

                    st.warning(
                        f"{name} failed: {e}"
                    )

            if results:

                st.subheader(
                    "📊 Model Results"
                )

                st.dataframe(
                    pd.DataFrame(results),
                    use_container_width=True,
                    hide_index=True
                )

        # =================================================
        # REGRESSION
        # =================================================

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

                    results.append({

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
                    })

                except Exception as e:

                    st.warning(
                        f"{name} failed: {e}"
                    )

            if results:

                st.subheader(
                    "📊 Model Results"
                )

                st.dataframe(
                    pd.DataFrame(results),
                    use_container_width=True,
                    hide_index=True
                )


# =========================================================
# DOWNLOAD
# =========================================================

st.divider()

st.subheader(
    "📥 Download CSV"
)

csv_bytes = df.to_csv(
    index=False
).encode(
    "utf-8"
)

st.download_button(
    label="⬇️ Download Dataset",
    data=csv_bytes,
    file_name="dataset.csv",
    mime="text/csv",
    use_container_width=True
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<br>

<div style="text-align:center; opacity:.55;">

CSV Master AI • Dynamic CSV Analysis & Machine Learning

</div>
""", unsafe_allow_html=True)
