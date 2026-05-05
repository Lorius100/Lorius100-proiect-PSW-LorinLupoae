"""
Pagina 2: Data Cleaning
- Tratarea valorilor lipsă (mean / median / mode / drop / fill cu 0)
- Tratarea outlier-ilor (IQR / percentile capping / log transform)
- Utilizatorul ALEGE metoda
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import build_main_dataset

st.set_page_config(page_title="Data Cleaning", page_icon="🧹", layout="wide")
st.title("🧹 Data Cleaning – Tratarea datelor")

st.markdown(
    """
    În această pagină tratăm **valorile lipsă** și **outlier-ii**. Tu (utilizatorul) alegi
    metoda. Modificările se salvează automat și se propagă
    către paginile următoare.
    """
)

# ============= Verificare date =============
if "main_df" not in st.session_state:
    df = build_main_dataset()
    st.session_state["main_df"] = df
    st.info("ℹ️ Date încărcate automat din sursa preîncărcată.")
else:
    df = st.session_state["main_df"].copy()

st.success(f"📊 Set curent: **{df.shape[0]:,} jucători** × **{df.shape[1]} coloane**".replace(",", "."))

st.divider()

# ============= TAB-uri =============
tab_missing, tab_outliers, tab_summary = st.tabs([
    "🕳️ 1. Valori lipsă",
    "📍 2. Outlier-i",
    "✅ 3. Sumar final"
])

# ===================== TAB 1: VALORI LIPSĂ =====================
with tab_missing:
    st.subheader("Tratarea valorilor lipsă")
    st.markdown(
        """
        **Strategii disponibile:**
        - **Mean / Median** — pentru variabile numerice (median e mai robust la outlier-i)  
        - **Mode** — pentru variabile categorice (cea mai frecventă valoare)  
        - **Fill cu 0** — pentru variabile unde NaN înseamnă "absență" (ex: goluri pentru un jucător care n-a jucat)  
        - **Drop rows** — eliminăm rândurile cu NaN (când coloana e foarte importantă)  
        - **Drop column** — eliminăm coloana (dacă are >50% NaN și nu e critică)
        """
    )

    # Lista coloanelor cu valori lipsă
    missing = df.isnull().sum()
    cols_with_missing = missing[missing > 0].sort_values(ascending=False)

    if len(cols_with_missing) == 0:
        st.success("🎉 Setul de date nu are valori lipsă! Poți trece la tratarea outlier-ilor.")
    else:
        st.markdown("**Coloane cu valori lipsă:**")
        miss_df = pd.DataFrame({
            "Coloană": cols_with_missing.index,
            "Nr. NaN": cols_with_missing.values,
            "Procent (%)": (cols_with_missing.values / len(df) * 100).round(2)
        })
        st.dataframe(miss_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 🛠️ Aplică o strategie de tratare")

        col_to_treat = st.selectbox(
            "Alege coloana:",
            options=cols_with_missing.index.tolist()
        )

        col_dtype = df[col_to_treat].dtype
        is_numeric = pd.api.types.is_numeric_dtype(col_dtype)

        if is_numeric:
            method_options = [
                "Mediana", "Media", "Fill cu 0",
                "Drop rânduri", "Drop coloană"
            ]
        else:
            method_options = [
                "Mode (cea mai frecventă)", "Fill cu 'Unknown'",
                "Drop rânduri", "Drop coloană"
            ]

        method = st.radio(
            "Metoda de tratare:",
            options=method_options,
            horizontal=True
        )

        # Preview înainte de aplicare
        st.markdown("**Preview înainte de aplicare:**")
        preview_data = df[col_to_treat].head(10)
        st.write(preview_data.tolist())

        if st.button("🚀 Aplică tratarea", type="primary"):
            df_new = df.copy()

            if method == "Mediana":
                fill_val = df_new[col_to_treat].median()
                df_new[col_to_treat] = df_new[col_to_treat].fillna(fill_val)
                st.success(f"✅ Înlocuit NaN cu mediana = `{fill_val:.2f}`")

            elif method == "Media":
                fill_val = df_new[col_to_treat].mean()
                df_new[col_to_treat] = df_new[col_to_treat].fillna(fill_val)
                st.success(f"✅ Înlocuit NaN cu media = `{fill_val:.2f}`")

            elif method == "Fill cu 0":
                df_new[col_to_treat] = df_new[col_to_treat].fillna(0)
                st.success("✅ Înlocuit NaN cu `0`")

            elif method == "Mode (cea mai frecventă)":
                fill_val = df_new[col_to_treat].mode()[0]
                df_new[col_to_treat] = df_new[col_to_treat].fillna(fill_val)
                st.success(f"✅ Înlocuit NaN cu mode = `{fill_val}`")

            elif method == "Fill cu 'Unknown'":
                df_new[col_to_treat] = df_new[col_to_treat].fillna("Unknown")
                st.success("✅ Înlocuit NaN cu `'Unknown'`")

            elif method == "Drop rânduri":
                before = len(df_new)
                df_new = df_new.dropna(subset=[col_to_treat])
                after = len(df_new)
                st.success(f"✅ Eliminat **{before - after:,}** rânduri".replace(",", "."))

            elif method == "Drop coloană":
                df_new = df_new.drop(columns=[col_to_treat])
                st.success(f"✅ Eliminat coloana `{col_to_treat}`")

            st.session_state["main_df"] = df_new
            st.rerun()

        # Buton pentru tratare automată "smart" pe toate coloanele
        st.markdown("---")
        st.markdown("### 🪄 Sau aplică o strategie globală automată")
        if st.button("Tratează toate coloanele automat (median pentru numerice, mode pentru categorice, drop pentru >70% NaN)"):
            df_new = df.copy()
            actions = []
            for col in df_new.columns:
                pct_missing = df_new[col].isnull().sum() / len(df_new) * 100
                if pct_missing == 0:
                    continue
                if pct_missing > 70:
                    df_new = df_new.drop(columns=[col])
                    actions.append(f"❌ `{col}` (>{pct_missing:.0f}% NaN) → DROP coloană")
                elif pd.api.types.is_numeric_dtype(df_new[col]):
                    fill = df_new[col].median()
                    df_new[col] = df_new[col].fillna(fill)
                    actions.append(f"🔢 `{col}` → median = {fill:.2f}")
                else:
                    fill = df_new[col].mode()[0] if len(df_new[col].mode()) > 0 else "Unknown"
                    df_new[col] = df_new[col].fillna(fill)
                    actions.append(f"🔤 `{col}` → mode = `{fill}`")

            st.session_state["main_df"] = df_new
            for a in actions:
                st.write(a)
            st.success("✅ Tratare automată aplicată!")
            st.rerun()

# ===================== TAB 2: OUTLIER-I =====================
with tab_outliers:
    st.subheader("Tratarea outlier-ilor (valori extreme)")
    st.markdown(
        r"""
        **Outlier-ii** sunt valori atipice care se abat semnificativ de la restul observațiilor.
        Pot afecta modelele de regresie sau clusterizare.

        **Metode disponibile:**
        - **IQR (Interquartile Range)** — eliminăm valorile dincolo de `Q1 - 1.5×IQR` și `Q3 + 1.5×IQR`
        - **Percentile Capping (Winsorize)** — înlocuim valorile sub percentila 1% sau peste 99% cu acele limite (păstrăm rândurile)
        - **Log Transform** — aplicăm `log(x+1)` pentru a comprima valori extreme (util pentru `market_value`)
        - **Păstrare** — uneori outlier-ii sunt informație reală (ex: jucători de top precum Mbappé, Haaland)
        """
    )

    df = st.session_state["main_df"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.warning("Nu există coloane numerice în setul curent.")
    else:
        col_outlier = st.selectbox(
            "Alege coloana numerică pentru analiza outlier-ilor:",
            options=numeric_cols,
            index=numeric_cols.index("market_value_in_eur") if "market_value_in_eur" in numeric_cols else 0,
            key="outlier_col"
        )

        # Calculăm IQR
        Q1 = df[col_outlier].quantile(0.25)
        Q3 = df[col_outlier].quantile(0.75)
        IQR = Q3 - Q1
        lower_iqr = Q1 - 1.5 * IQR
        upper_iqr = Q3 + 1.5 * IQR
        outliers_iqr = df[(df[col_outlier] < lower_iqr) | (df[col_outlier] > upper_iqr)]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Q1 (25%)", f"{Q1:,.2f}")
        c2.metric("Q3 (75%)", f"{Q3:,.2f}")
        c3.metric("IQR", f"{IQR:,.2f}")
        c4.metric("Nr. outlier-i (IQR)", f"{len(outliers_iqr):,}".replace(",", "."))

        # Boxplot înainte
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].boxplot(df[col_outlier].dropna(), vert=False)
        axes[0].set_title(f"Boxplot: {col_outlier}")
        axes[0].set_xlabel(col_outlier)
        axes[0].grid(alpha=0.3)

        axes[1].hist(df[col_outlier].dropna(), bins=50, color="#4c72b0", edgecolor="black", alpha=0.85)
        axes[1].set_title(f"Histogramă: {col_outlier}")
        axes[1].set_xlabel(col_outlier)
        axes[1].set_ylabel("Frecvență")
        axes[1].grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")
        st.markdown("### 🛠️ Aplică o metodă")

        method = st.radio(
            "Metoda:",
            options=["IQR (eliminare)", "Percentile Capping (1%-99%)", "Log Transform", "Păstrează outlier-ii"],
            horizontal=True
        )

        if st.button("🚀 Aplică pe coloană", type="primary", key="apply_outliers"):
            df_new = st.session_state["main_df"].copy()

            if method == "IQR (eliminare)":
                before = len(df_new)
                df_new = df_new[
                    (df_new[col_outlier] >= lower_iqr) & (df_new[col_outlier] <= upper_iqr)
                ]
                after = len(df_new)
                st.success(
                    f"✅ Eliminat **{before - after:,}** outlier-i (IQR). "
                    f"Limite: [{lower_iqr:.2f}, {upper_iqr:.2f}]".replace(",", ".")
                )

            elif method == "Percentile Capping (1%-99%)":
                p1 = df_new[col_outlier].quantile(0.01)
                p99 = df_new[col_outlier].quantile(0.99)
                df_new[col_outlier] = df_new[col_outlier].clip(lower=p1, upper=p99)
                st.success(
                    f"✅ Capping aplicat: valorile sub `{p1:.2f}` și peste `{p99:.2f}` "
                    "au fost înlocuite cu aceste limite."
                )

            elif method == "Log Transform":
                # log1p pentru a evita probleme cu 0
                new_col_name = f"{col_outlier}_log"
                df_new[new_col_name] = np.log1p(df_new[col_outlier].fillna(0))
                st.success(
                    f"✅ Coloană nouă creată: `{new_col_name}` = `log(1 + {col_outlier})`. "
                    f"Coloana originală a fost păstrată."
                )

            elif method == "Păstrează outlier-ii":
                st.info("ℹ️ Outlier-ii sunt păstrați. Nu s-au făcut modificări.")

            st.session_state["main_df"] = df_new
            if method != "Păstrează outlier-ii":
                st.rerun()

# ===================== TAB 3: SUMAR FINAL =====================
with tab_summary:
    st.subheader("Sumar final după tratare")

    df = st.session_state["main_df"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Rânduri rămase", f"{df.shape[0]:,}".replace(",", "."))
    c2.metric("Coloane rămase", df.shape[1])
    total_missing = df.isnull().sum().sum()
    c3.metric("Total NaN rămase", f"{total_missing:,}".replace(",", "."))

    if total_missing > 0:
        st.warning(
            f"⚠️ Mai există **{total_missing:,}** valori lipsă. "
            "Recomandare: tratează-le înainte de modelare.".replace(",", ".")
        )
    else:
        st.success("🎉 Setul de date este complet curat — gata pentru următorul pas!")

    # Buton de reset
    if st.button("🔄 Resetează la setul original", type="secondary"):
        st.session_state["main_df"] = build_main_dataset()
        st.rerun()

st.divider()
st.info(
    "👉 **Următorul pas:** mergi la pagina **🔢 Encoding & Scaling** pentru codificare și scalare."
)
