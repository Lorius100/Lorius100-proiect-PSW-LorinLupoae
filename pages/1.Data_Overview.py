"""
Pagina 1: Data Overview
- Încărcare CSV (preîncărcat sau upload)
- Verificare dimensiuni, tipuri, valori lipsă
- Statistici descriptive
- Funcții de grup (groupby)
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import build_main_dataset, load_clubs, load_competitions

st.set_page_config(page_title="Data Overview", page_icon="📂", layout="wide")
st.title("📂 Data Overview – Inspectarea datelor")

st.markdown(
    """
    În această pagină **inspectăm structura setului de date principal** (jucători + statistici de
    performanță + cluburi + competiții). Este pasul esențial înainte de orice modelare —
    permite înțelegerea formatului, identificarea valorilor lipsă și a anomaliilor.
    """
)

# ============= Sursa datelor =============
st.subheader("📤 Sursa datelor")
data_source = st.radio(
    "Alege sursa setului de date:",
    options=["Date preîncărcate (recomandat)", "Încarcă propriul CSV"],
    horizontal=True
)

if data_source == "Date preîncărcate (recomandat)":
    df = build_main_dataset()
    st.success(
        f"✅ Set de date principal încărcat: **{df.shape[0]:,} jucători** × "
        f"**{df.shape[1]} coloane**".replace(",", ".")
    )
else:
    uploaded = st.file_uploader(
        "Încarcă un fișier CSV (format Transfermarkt sau echivalent)",
        type=["csv"]
    )
    if uploaded is None:
        st.info("⏳ Aștept fișier CSV...")
        st.stop()
    df = pd.read_csv(uploaded, low_memory=False)
    st.success(
        f"✅ Fișier încărcat: **{df.shape[0]:,} rânduri** × **{df.shape[1]} coloane**"
        .replace(",", ".")
    )

# Salvăm în session_state pentru a fi disponibil în alte pagini
st.session_state["main_df"] = df

st.divider()

# ============= Tab-uri pentru organizare =============
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Structură",
    "📊 Statistici descriptive",
    "🕳️ Valori lipsă",
    "📈 Distribuții numerice",
    "🧮 Funcții de grup"
])

# -------- TAB 1: Structură --------
with tab1:
    st.subheader("Structura setului de date")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rânduri (jucători)", f"{df.shape[0]:,}".replace(",", "."))
    c2.metric("Coloane", df.shape[1])
    c3.metric("Memorie utilizată", f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    st.markdown("**Primele rânduri (`df.head()`):**")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("**Tipuri de date pe coloană (`df.dtypes`):**")
    dtypes_df = pd.DataFrame({
        "Coloană": df.columns,
        "Tip": df.dtypes.astype(str).values,
        "Valori nenule": df.count().values,
        "Valori unice": [df[c].nunique() for c in df.columns]
    })
    st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

    # Separare coloane numerice vs categorice
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    cN, cC = st.columns(2)
    with cN:
        st.markdown(f"**🔢 Coloane numerice ({len(numeric_cols)}):**")
        st.write(numeric_cols)
    with cC:
        st.markdown(f"**🔤 Coloane categorice ({len(categorical_cols)}):**")
        st.write(categorical_cols)

# -------- TAB 2: Statistici descriptive --------
with tab2:
    st.subheader("Statistici descriptive (`df.describe()`)")
    st.markdown(
        """
        - **count** — număr observații nenule  
        - **mean** — media valorilor  
        - **std** — deviația standard (cât de împrăștiate sunt valorile)  
        - **min / 25% / 50% / 75% / max** — distribuția pe quartile (50% = mediana)
        """
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_cols = st.multiselect(
        "Alege coloanele numerice pentru analiză:",
        options=numeric_cols,
        default=[c for c in numeric_cols if c in [
            "age", "height_in_cm", "market_value_in_eur",
            "highest_market_value_in_eur", "total_goals", "total_assists",
            "total_minutes", "goals_per_90"
        ]][:8]
    )

    if selected_cols:
        st.dataframe(df[selected_cols].describe().round(2), use_container_width=True)

# -------- TAB 3: Valori lipsă --------
with tab3:
    st.subheader("Analiza valorilor lipsă")
    st.markdown(
        "Identificăm coloanele cu valori lipsă (NaN). Acestea trebuie tratate înainte de "
        "modelare — vom face asta în pagina **Data Cleaning**."
    )

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Coloană": missing.index,
        "Nr. valori lipsă": missing.values,
        "Procent (%)": missing_pct.values
    })
    missing_df = missing_df[missing_df["Nr. valori lipsă"] > 0].sort_values(
        "Procent (%)", ascending=False
    )

    if len(missing_df) == 0:
        st.success("🎉 Setul de date nu are valori lipsă!")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
        with c2:
            fig, ax = plt.subplots(figsize=(8, max(4, len(missing_df) * 0.3)))
            ax.barh(missing_df["Coloană"], missing_df["Procent (%)"], color="#ff6b6b")
            ax.set_xlabel("Procent valori lipsă (%)")
            ax.set_title("Valori lipsă pe coloană")
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

# -------- TAB 4: Distribuții numerice --------
with tab4:
    st.subheader("Distribuții pentru variabile numerice")
    st.markdown(
        "Histograma arată cum sunt distribuite valorile unei variabile. "
        "Distribuții puternic asimetrice pot indica nevoia de transformări (ex: logaritm)."
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    col_to_plot = st.selectbox(
        "Alege coloana:",
        options=numeric_cols,
        index=numeric_cols.index("market_value_in_eur") if "market_value_in_eur" in numeric_cols else 0
    )

    bins = st.slider("Număr bins:", min_value=10, max_value=100, value=40, step=5)
    log_scale = st.checkbox("📐 Scală logaritmică pe axa Y (util pentru variabile cu coadă lungă)")

    fig, ax = plt.subplots(figsize=(10, 4))
    data_clean = df[col_to_plot].dropna()
    ax.hist(data_clean, bins=bins, color="#4c72b0", edgecolor="black", alpha=0.85)
    ax.set_xlabel(col_to_plot)
    ax.set_ylabel("Frecvență (log)" if log_scale else "Frecvență")
    ax.set_title(f"Distribuția: {col_to_plot}")
    if log_scale:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    # Scurtă descriere
    st.markdown(
        f"**Statistici rapide pentru `{col_to_plot}`:** "
        f"medie = `{data_clean.mean():,.2f}`, "
        f"mediana = `{data_clean.median():,.2f}`, "
        f"min = `{data_clean.min():,.2f}`, "
        f"max = `{data_clean.max():,.2f}`, "
        f"std = `{data_clean.std():,.2f}`"
    )

# -------- TAB 5: Funcții de grup --------
with tab5:
    st.subheader("Gruparea și agregarea datelor (`pandas.groupby`)")
    st.markdown(
        "Funcțiile de grup permit calcule sumare pe categorii — esențial pentru "
        "înțelegerea structurii datelor."
    )

    # Selectarea coloanei de grupare
    group_options = [c for c in ["position", "sub_position", "country_of_citizenship",
                                  "competition_country", "foot"]
                     if c in df.columns]
    group_col = st.selectbox("Grupează după:", options=group_options, index=0)

    metric_options = [c for c in ["market_value_in_eur", "age", "height_in_cm",
                                   "total_goals", "total_assists", "goals_per_90",
                                   "total_minutes"] if c in df.columns]
    metric_col = st.selectbox("Variabila numerică pentru agregare:", options=metric_options)

    agg_funcs = st.multiselect(
        "Funcții de agregare:",
        options=["count", "mean", "median", "sum", "min", "max", "std"],
        default=["count", "mean", "median"]
    )

    if group_col and metric_col and agg_funcs:
        grouped = df.groupby(group_col)[metric_col].agg(agg_funcs).round(2)
        grouped = grouped.sort_values(by=agg_funcs[0], ascending=False).head(20)
        st.dataframe(grouped, use_container_width=True)

        # Plot pe top 15 categorii
        if len(grouped) > 1 and "mean" in agg_funcs:
            fig, ax = plt.subplots(figsize=(10, max(4, len(grouped.head(15)) * 0.35)))
            top_15 = grouped.head(15)
            ax.barh(top_15.index.astype(str), top_15["mean"], color="#55a868")
            ax.set_xlabel(f"Media: {metric_col}")
            ax.set_title(f"Top 15 categorii din `{group_col}` după media `{metric_col}`")
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

st.divider()
st.info(
    "👉 **Următorul pas:** mergi la pagina **🧹 Data Cleaning** pentru a trata valorile lipsă și outlier-ii."
)
