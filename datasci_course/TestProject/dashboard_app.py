import os
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# The database is created by the notebook inside TestProject/, so keep it local to this folder
DB_PATH = os.path.join(os.path.dirname(__file__), "loan_database.db")

TARGET_METADATA = {
    "shopping_predictions": {
        "target": "Subscription Status",
        "classes": {
            1: "Yes (Subscribed)",
            0: "No (Not Subscribed)"
        }
    }
}

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

def get_columns(table: str, conn: sqlite3.Connection) -> pd.DataFrame:
    schema = get_schema(table, conn)
    return schema[["name", "type"]].copy()

def diff_engineered_columns(raw_table: str, fe_table: str, conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Return a dataframe listing columns that exist in the feature-engineered table
    but not in the raw table, along with SQLite inferred types.
    """
    raw_cols = get_columns(raw_table, conn)
    fe_cols = get_columns(fe_table, conn)
    raw_set = set(raw_cols["name"].tolist())
    fe_only = fe_cols[~fe_cols["name"].isin(raw_set)].copy()
    fe_only = fe_only.rename(columns={"name": "feature", "type": "sqlite_type"})
    return fe_only.sort_values(by="feature").reset_index(drop=True)

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

    # Feature Engineering explorer
    st.subheader("Feature Engineering")
    if "shopping_data" in ordered and "shopping_data_fe" in ordered:
        col1, col2 = st.columns(2)
        with col1:
            raw_table_sel = st.selectbox(
                "Raw dataset table",
                options=[t for t in ordered if t == "shopping_data" or t.endswith("_raw") or t in ("training_data",)],
                index=0 if "shopping_data" in ordered else 0,
                key="raw_table_sel"
            )
        with col2:
            fe_table_sel = st.selectbox(
                "Feature-engineered table",
                options=[t for t in ordered if t.endswith("_fe") or t == "shopping_data_fe"],
                index=0,
                key="fe_table_sel"
            )

        try:
            fe_diff = diff_engineered_columns(raw_table_sel, fe_table_sel, conn)
        except Exception as e:
            st.error(f"Unable to compare schemas: {e}")
            fe_diff = pd.DataFrame()

        st.markdown("New engineered columns (present in FE, not in Raw):")
        if fe_diff.empty:
            st.info("No new columns detected, or schema comparison failed.")
        else:
            st.dataframe(fe_diff, use_container_width=True, hide_index=True)

        # Preview engineered columns only
        with st.expander("Preview engineered columns", expanded=False):
            if not fe_diff.empty:
                engineered_cols = fe_diff["feature"].tolist()
                preview_query = f"SELECT {', '.join([f'`{c}`' for c in engineered_cols])} FROM {fe_table_sel} LIMIT 200"
                try:
                    fe_preview = read_sql(preview_query, conn)
                except Exception:
                    # Fallback to select all if any column quoting fails
                    fe_preview = read_sql(f"SELECT * FROM {fe_table_sel} LIMIT 200", conn)
                    fe_preview = fe_preview[engineered_cols] if all(c in fe_preview.columns for c in engineered_cols) else fe_preview
                st.dataframe(fe_preview, use_container_width=True, hide_index=True)
            else:
                st.caption("No engineered columns to preview.")

        with st.expander("Schemas (raw vs FE)", expanded=False):
            left, right = st.columns(2)
            with left:
                st.markdown(f"Schema: `{raw_table_sel}`")
                st.dataframe(get_schema(raw_table_sel, conn), use_container_width=True, hide_index=True)
            with right:
                st.markdown(f"Schema: `{fe_table_sel}`")
                st.dataframe(get_schema(fe_table_sel, conn), use_container_width=True, hide_index=True)
    else:
        st.info("Feature-engineered table `shopping_data_fe` not found. Create it in your notebook to enable this view.")

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

        display_df = preds_df.copy()
        meta = TARGET_METADATA.get(preds_table)
        if meta and {'pred', 'truth'}.issubset(display_df.columns):
            label_map = meta.get("classes", {})
            display_df["pred_label"] = display_df["pred"].map(label_map).fillna(display_df["pred"].astype(str))
            display_df["truth_label"] = display_df["truth"].map(label_map).fillna(display_df["truth"].astype(str))
            st.caption(f"Target: **{meta['target']}** — " +
                       ", ".join(f"{k} = {v}" for k, v in label_map.items()))
        st.dataframe(display_df.head(100), use_container_width=True, hide_index=True)
    else:
        st.info("No predictions table found (e.g., `shopping_predictions`).")

    st.markdown("---")
    st.caption(f"Database: {DB_PATH}")

if __name__ == "__main__":
    main()


