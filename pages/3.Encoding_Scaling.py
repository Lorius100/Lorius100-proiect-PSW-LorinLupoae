"""
Pagina 3: Encoding & Scaling
- Codificare variabile categorice (One-Hot, Label, Target, Frequency)
- Scalare variabile numerice (StandardScaler, MinMaxScaler)
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from utils.data_loader import build_main_dataset

st.set_page_config(page_title="Encoding & Scaling", page_icon="🔢", layout="wide")
st.title("🔢 Encoding & Scaling – Codificare și scalare")

st.markdown(
    """
    Modelele de Machine Learning lucrează cu **numere**, nu cu text. În această pagină:
    1. **Codificăm** variabilele categorice (text) în numerice
    2. **Scalăm** variabilele numerice pentru a aduce toate valorile la aceeași scară
    """
)

# ============= Verificare date =============
if "main_df" not in st.session_state:
    df = build_main_dataset()
    st.session_state["main_df"] = df

df = st.session_state["main_df"].copy()
st.success(f"📊 Set curent: **{df.shape[0]:,} jucători** × **{df.shape[1]} coloane**".replace(",", "."))

st.divider()

tab_encode, tab_scale, tab_corr = st.tabs([
    "🔤 1. Encoding (codificare)",
    "📏 2. Scaling (scalare)",
    "🔗 3. Matricea de corelație"
])

# ===================== TAB 1: ENCODING =====================
with tab_encode:
    st.subheader("Codificarea variabilelor categorice")
    st.markdown(
        """
        **Metode disponibile:**
        - **One-Hot Encoding** — fiecare categorie devine o coloană separată cu valori 0/1.  
          *Recomandat pentru:* variabile cu puține categorii (≤10), fără ordine naturală (ex: `position`, `foot`)
        - **Label Encoding** — atribuie un număr întreg fiecărei categorii (`0, 1, 2, ...`).  
          *Recomandat pentru:* variabile ordinale sau pentru a economisi memorie
        - **Target Encoding** — înlocuiește categoria cu media variabilei țintă (ex: prețul mediu).  
          *Recomandat pentru:* variabile cu multe categorii (>20), unde One-Hot ar exploda dimensionalitatea
        - **Frequency Encoding** — înlocuiește categoria cu frecvența ei relativă în set
        """
    )

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if not categorical_cols:
        st.warning("Nu există coloane categorice (de tip object) în setul curent.")
    else:
        st.markdown("**Coloane categorice disponibile:**")
        cat_info = pd.DataFrame({
            "Coloană": categorical_cols,
            "Nr. valori unice": [df[c].nunique() for c in categorical_cols],
            "Valori lipsă": [df[c].isnull().sum() for c in categorical_cols]
        })
        st.dataframe(cat_info, use_container_width=True, hide_index=True)

        st.markdown("---")
        col_to_encode = st.selectbox(
            "Alege coloana de codificat:",
            options=categorical_cols
        )

        n_unique = df[col_to_encode].nunique()
        st.info(f"Coloana `{col_to_encode}` are **{n_unique}** valori unice.")

        # Sugestie automată
        if n_unique <= 10:
            suggestion = "One-Hot Encoding (recomandat pentru cardinalitate mică)"
        elif n_unique <= 50:
            suggestion = "Label Encoding sau Target Encoding"
        else:
            suggestion = "Frequency Encoding sau Target Encoding (cardinalitate mare)"
        st.markdown(f"💡 **Sugestie:** {suggestion}")

        encode_method = st.radio(
            "Metoda de codificare:",
            options=["One-Hot Encoding", "Label Encoding", "Target Encoding", "Frequency Encoding"],
            horizontal=True
        )

        # Pentru target encoding, alegem coloana țintă
        target_col = None
        if encode_method == "Target Encoding":
            numeric_cols_for_target = df.select_dtypes(include=[np.number]).columns.tolist()
            target_col = st.selectbox(
                "Variabila țintă (numerică) pentru target encoding:",
                options=numeric_cols_for_target,
                index=numeric_cols_for_target.index("market_value_in_eur")
                if "market_value_in_eur" in numeric_cols_for_target else 0
            )

        # Preview
        st.markdown("**Preview înainte de codificare:**")
        st.write(df[col_to_encode].value_counts().head(10))

        if st.button("🚀 Aplică codificarea", type="primary"):
            df_new = st.session_state["main_df"].copy()

            if encode_method == "One-Hot Encoding":
                # drop_first=True pentru a evita coliniaritatea dummy
                encoded = pd.get_dummies(df_new[col_to_encode], prefix=col_to_encode, drop_first=True)
                # Convertim bool la int (0/1)
                encoded = encoded.astype(int)
                df_new = pd.concat([df_new.drop(columns=[col_to_encode]), encoded], axis=1)
                st.success(
                    f"✅ One-Hot Encoding aplicat. Au fost create **{encoded.shape[1]}** coloane noi: "
                    f"{', '.join(encoded.columns.tolist()[:5])}{'...' if len(encoded.columns) > 5 else ''}"
                )

            elif encode_method == "Label Encoding":
                le = LabelEncoder()
                # Tratăm NaN ca o categorie separată
                df_new[col_to_encode] = df_new[col_to_encode].fillna("Unknown").astype(str)
                df_new[f"{col_to_encode}_encoded"] = le.fit_transform(df_new[col_to_encode])
                mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                st.success(f"✅ Label Encoding aplicat. Coloană nouă: `{col_to_encode}_encoded`")
                with st.expander("Vezi maparea (primele 15)"):
                    st.write({k: int(v) for k, v in list(mapping.items())[:15]})

            elif encode_method == "Target Encoding":
                target_means = df_new.groupby(col_to_encode)[target_col].mean()
                new_col = f"{col_to_encode}_target_enc"
                df_new[new_col] = df_new[col_to_encode].map(target_means)
                # Pentru NaN, folosim media globală
                df_new[new_col] = df_new[new_col].fillna(df_new[target_col].mean())
                st.success(
                    f"✅ Target Encoding aplicat (target = `{target_col}`). "
                    f"Coloană nouă: `{new_col}`"
                )

            elif encode_method == "Frequency Encoding":
                freq_map = df_new[col_to_encode].value_counts(normalize=True)
                new_col = f"{col_to_encode}_freq_enc"
                df_new[new_col] = df_new[col_to_encode].map(freq_map)
                df_new[new_col] = df_new[new_col].fillna(0)
                st.success(f"✅ Frequency Encoding aplicat. Coloană nouă: `{new_col}`")

            st.session_state["main_df"] = df_new
            st.rerun()

# ===================== TAB 2: SCALING =====================
with tab_scale:
    st.subheader("Scalarea variabilelor numerice")
    st.markdown(
        r"""
        Modelele precum **regresia liniară**, **K-Means**, **SVM** și **rețelele neuronale**
        sunt sensibile la scala datelor. Dacă o variabilă are valori în milioane (ex: `market_value`)
        și alta în zeci (ex: `age`), prima va domina modelul.

        **Metode:**
        - **StandardScaler** — transformă datele să aibă **media 0** și **deviația standard 1**.  
          Formula: $z = (x - \mu) / \sigma$
        - **MinMaxScaler** — transformă datele în intervalul **[0, 1]**.  
          Formula: $x' = (x - x_{min}) / (x_{max} - x_{min})$
        """
    )

    df = st.session_state["main_df"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.warning("Nu există coloane numerice de scalat.")
    else:
        # Lista coloanelor potențial scalabile (excludem ID-uri evidente și coloane deja în [0,1])
        excluded = [c for c in numeric_cols
                    if c.endswith("_id") or c.endswith("_encoded")
                    or c.startswith("position_") or c.startswith("foot_")
                    or c.startswith("sub_position_")]
        scale_candidates = [c for c in numeric_cols if c not in excluded]

        cols_to_scale = st.multiselect(
            "Alege coloanele de scalat (numerice continue):",
            options=scale_candidates,
            default=[c for c in [
                "age", "height_in_cm", "market_value_in_eur",
                "highest_market_value_in_eur", "total_goals",
                "total_assists", "total_minutes", "goals_per_90"
            ] if c in scale_candidates]
        )

        scale_method = st.radio(
            "Metoda de scalare:",
            options=["StandardScaler", "MinMaxScaler"],
            horizontal=True
        )

        if cols_to_scale and st.button("🚀 Aplică scalarea", type="primary"):
            df_new = st.session_state["main_df"].copy()

            if scale_method == "StandardScaler":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()

            # Tratăm NaN cu mediana
            for c in cols_to_scale:
                if df_new[c].isnull().any():
                    df_new[c] = df_new[c].fillna(df_new[c].median())

            scaled_cols = [f"{c}_scaled" for c in cols_to_scale]
            df_new[scaled_cols] = scaler.fit_transform(df_new[cols_to_scale])

            st.success(
                f"✅ Scalare aplicată ({scale_method}). Au fost create coloane noi cu sufixul `_scaled`."
            )

            # Afișăm comparație
            comp_df = pd.DataFrame({
                "Coloană": cols_to_scale,
                "Min original": [df_new[c].min() for c in cols_to_scale],
                "Max original": [df_new[c].max() for c in cols_to_scale],
                "Min scalat": [df_new[f"{c}_scaled"].min() for c in cols_to_scale],
                "Max scalat": [df_new[f"{c}_scaled"].max() for c in cols_to_scale],
                "Medie scalat": [df_new[f"{c}_scaled"].mean() for c in cols_to_scale],
            }).round(3)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Boxplot comparativ
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            df_new[cols_to_scale].boxplot(ax=axes[0])
            axes[0].set_title("Înainte de scalare")
            axes[0].tick_params(axis="x", rotation=45)
            axes[0].grid(alpha=0.3)

            df_new[scaled_cols].boxplot(ax=axes[1])
            axes[1].set_title(f"După scalare ({scale_method})")
            axes[1].tick_params(axis="x", rotation=45)
            axes[1].grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.session_state["main_df"] = df_new

# ===================== TAB 3: CORELAȚII =====================
with tab_corr:
    st.subheader("Matricea de corelație")
    st.markdown(
        """
        Corelația măsoară cât de strâns sunt legate două variabile (între -1 și +1).
        - **+1** = corelație pozitivă perfectă (când una crește, cealaltă crește)
        - **-1** = corelație negativă perfectă (când una crește, cealaltă scade)
        - **0** = nicio relație liniară

        **Interpretare:** corelații peste **|0.7|** indică redundanță (poate fi nevoie să eliminăm o coloană).
        """
    )

    df = st.session_state["main_df"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Excludem coloane de tip ID
    interesting_cols = [c for c in numeric_cols
                        if not c.endswith("_id") and not c.endswith("encoded")]

    selected = st.multiselect(
        "Alege coloanele pentru matricea de corelație:",
        options=interesting_cols,
        default=[c for c in [
            "age", "height_in_cm", "market_value_in_eur",
            "highest_market_value_in_eur", "total_goals", "total_assists",
            "total_minutes", "goals_per_90", "assists_per_90",
            "club_total_market_value", "club_squad_size"
        ] if c in interesting_cols][:8]
    )

    if len(selected) >= 2:
        corr = df[selected].corr()

        fig, ax = plt.subplots(figsize=(min(12, max(6, len(selected) * 0.9)),
                                          min(10, max(5, len(selected) * 0.7))))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, vmin=-1, vmax=1, square=True,
                    linewidths=0.5, ax=ax)
        ax.set_title("Matricea de corelație Pearson")
        st.pyplot(fig)
        plt.close(fig)

        # Top 5 corelații puternice (excludem diagonala)
        st.markdown("**Cele mai puternice corelații:**")
        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                corr_pairs.append({
                    "Variabila A": corr.columns[i],
                    "Variabila B": corr.columns[j],
                    "Corelație": corr.iloc[i, j]
                })
        corr_df = pd.DataFrame(corr_pairs)
        corr_df["Abs"] = corr_df["Corelație"].abs()
        corr_df = corr_df.sort_values("Abs", ascending=False).drop(columns="Abs").head(10)
        corr_df["Corelație"] = corr_df["Corelație"].round(3)
        st.dataframe(corr_df, use_container_width=True, hide_index=True)

st.divider()
st.info("👉 **Următorul pas:** mergi la pagina **🗺️ Geospatial** pentru analize cu hărți europene.")
