import os
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# The database is created by the notebook inside TestProject/, so keep it local to this folder
DB_PATH = os.path.join(os.path.dirname(__file__), "loan_database.db")

@st.cache_resource
def get_conn(db_path: str):
    return sqlite3.connect(db_path, check_same_thread=False)

@st.cache_data(show_spinner=False)
def read_sql(query: str, _conn: sqlite3.Connection) -> pd.DataFrame:
    # Leading underscore on _conn avoids Streamlit hashing issues for cache
    return pd.read_sql_query(query, _conn)

def list_tables(conn: sqlite3.Connection) -> pd.DataFrame:
    return read_sql("SELECT name as table_name FROM sqlite_master WHERE type='table' ORDER BY name", conn)

def get_schema(table: str, conn: sqlite3.Connection) -> pd.DataFrame:
    return read_sql(f"PRAGMA table_info({table})", conn)

def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if not numeric_cols:
        return pd.DataFrame()
    desc = df[numeric_cols].describe().T
    desc["missing"] = df[numeric_cols].isnull().sum()
    return desc

def main():
    st.set_page_config(page_title="ML Project Dashboard", layout="wide")
    st.title("Machine Learning Project Dashboard")
    st.caption("SQLite-backed, switchable datasets, quick EDA, and model metrics")

    if not os.path.exists(DB_PATH):
        st.error(f"Database not found at: {DB_PATH}")
        st.stop()

    conn = get_conn(DB_PATH)

    # Sidebar: database inspector
    st.sidebar.header("Database")
    tables_df = list_tables(conn)
    if tables_df.empty:
        st.warning("No tables found in database.")
        st.stop()

    default_tables = ["shopping_data_fe", "shopping_data", "training_data", "test_data", "shopping_predictions", "shopping_metrics"]
    ordered = [t for t in default_tables if t in tables_df.table_name.values] + \
              [t for t in tables_df.table_name.values if t not in default_tables]

    st.sidebar.write("Tables:")
    st.sidebar.dataframe(pd.DataFrame({"table": ordered}), use_container_width=True, hide_index=True)

    # Dataset switcher
    st.subheader("Dataset Explorer")
    dataset_table = st.selectbox(
        "Select dataset table", 
        options=[t for t in ordered if t in ("shopping_data", "training_data", "test_data")],
        index=0 if "shopping_data" in ordered else 0
    )

    # Preview and schema
    df = read_sql(f"SELECT * FROM {dataset_table} LIMIT 1000", conn)
    st.markdown(f"**Preview: `{dataset_table}`**")
    st.dataframe(df.head(50), use_container_width=True, hide_index=True)

    with st.expander(f"Schema: `{dataset_table}`", expanded=False):
        schema = get_schema(dataset_table, conn)
        st.dataframe(schema, use_container_width=True, hide_index=True)

    # Quick EDA
    st.subheader("Quick EDA")
    cols = st.multiselect("Select columns to visualize", options=df.columns.tolist())
    if cols:
        for col in cols:
            st.markdown(f"Column: `{col}`")
            if pd.api.types.is_numeric_dtype(df[col]):
                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X(col, bin=alt.Bin(maxbins=30)),
                    y='count()'
                ).properties(height=200)
                st.altair_chart(chart, use_container_width=True)
            else:
                vc = df[col].value_counts().reset_index()
                vc.columns = [col, "count"]
                chart = alt.Chart(vc).mark_bar().encode(
                    x=alt.X(col, sort='-y'),
                    y='count'
                ).properties(height=200)
                st.altair_chart(chart, use_container_width=True)

    # Numeric summary
    with st.expander("Numeric summary", expanded=False):
        st.dataframe(numeric_summary(df), use_container_width=True)

    st.markdown("---")

    # Model metrics and predictions (if available)
    st.subheader("Model Metrics and Predictions")
    available_metrics = [t for t in ordered if t.endswith("_metrics")]
    if available_metrics:
        metrics_table = st.selectbox("Select metrics table", options=available_metrics, index=0)
        metrics_df = read_sql(f"SELECT * FROM {metrics_table}", conn)
        st.markdown(f"**Metrics from `{metrics_table}`**")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # Leaderboard view
        st.markdown("### Model Leaderboard")
        acc_type = st.radio("Accuracy to compare", options=["tuned_accuracy", "baseline_accuracy"], index=0, horizontal=True)
        if acc_type not in metrics_df.columns:
            # fallback if column missing
            acc_type = "baseline_accuracy"
        leaderboard = metrics_df.copy()
        leaderboard = leaderboard.dropna(subset=[acc_type])
        if not leaderboard.empty:
            leaderboard = leaderboard.sort_values(by=acc_type, ascending=False).reset_index(drop=True)
            st.dataframe(leaderboard[["model", acc_type]], use_container_width=True, hide_index=True)

            # Chart
            chart = alt.Chart(leaderboard).mark_bar().encode(
                y=alt.Y("model:N", sort='-x', title="Model"),
                x=alt.X(f"{acc_type}:Q", title="Accuracy"),
                tooltip=["model", alt.Tooltip(f"{acc_type}:Q", format=".3f")]
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)

            best_row = leaderboard.iloc[0]
            st.success(f"Best model: {best_row['model']}  —  {acc_type.replace('_',' ')} = {best_row[acc_type]:.4f}")
        else:
            st.info("No accuracy values found in metrics table.")
    else:
        st.info("No metrics table found (e.g., `shopping_metrics`).")

    available_preds = [t for t in ordered if t.endswith("_predictions")]
    if available_preds:
        preds_table = st.selectbox("Select predictions table", options=available_preds, index=0)
        preds_df = read_sql(f"SELECT * FROM {preds_table}", conn)
        st.markdown(f"**Predictions from `{preds_table}`**")
        st.dataframe(preds_df.head(100), use_container_width=True, hide_index=True)
    else:
        st.info("No predictions table found (e.g., `shopping_predictions`).")

    # Compare two datasets (basic compare)
    st.subheader("Compare Two Datasets")
    left, right = st.columns(2)
    with left:
        table_left = st.selectbox("Left dataset", options=[t for t in ordered if t in ("shopping_data", "training_data")], index=0, key="left")
        df_left = read_sql(f"SELECT * FROM {table_left} LIMIT 1000", conn)
        st.write(f"Rows: {len(df_left)}, Cols: {len(df_left.columns)}")
    with right:
        table_right = st.selectbox("Right dataset", options=[t for t in ordered if t in ("shopping_data", "training_data")], index=1 if "training_data" in ordered else 0, key="right")
        df_right = read_sql(f"SELECT * FROM {table_right} LIMIT 1000", conn)
        st.write(f"Rows: {len(df_right)}, Cols: {len(df_right.columns)}")

    # Simple categorical overlap compare (first common categorical column)
    common_cols = [c for c in df_left.columns if c in df_right.columns]
    if common_cols:
        compare_col = st.selectbox("Compare distribution on column", options=common_cols, index=0)
        if compare_col:
            def top_counts(df_, col):
                s = df_[col].astype(str).value_counts().head(10).reset_index()
                s.columns = [col, "count"]
                return s
            lc = top_counts(df_left, compare_col)
            rc = top_counts(df_right, compare_col)
            lc["dataset"] = table_left
            rc["dataset"] = table_right
            both = pd.concat([lc, rc], ignore_index=True)
            chart = alt.Chart(both).mark_bar().encode(
                x=alt.X("count:Q"),
                y=alt.Y(f"{compare_col}:N", sort='-x'),
                color="dataset:N",
                column=alt.Column("dataset:N")
            ).resolve_scale(y='independent').properties(height=250)
            st.altair_chart(chart, use_container_width=True)

    st.markdown("---")
    st.caption(f"Database: {DB_PATH}")

if __name__ == "__main__":
    main()


