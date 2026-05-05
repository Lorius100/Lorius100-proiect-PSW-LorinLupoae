"""
Pagina 5: Modeling
- K-Means clustering pentru segmentarea jucătorilor
- Regresie Logistică pentru predicția "top-tier" (Da/Nu)
- Regresie Multiplă (statsmodels) pentru predicția valorii de piață
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_curve, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression

import statsmodels.api as sm

from utils.data_loader import build_main_dataset

st.set_page_config(page_title="Modeling", page_icon="🤖", layout="wide")
st.title("🤖 Modeling – Modele de Machine Learning")

st.markdown(
    """
    În această pagină aplicăm **trei tipuri de modele**:
    1. **K-Means** — segmentarea jucătorilor pe profile (atacanți rapizi, fundași defensivi etc.)
    2. **Regresie Logistică** — clasificare binară: e jucătorul "top-tier" sau nu?
    3. **Regresie Multiplă** (statsmodels) — predicția valorii de piață în funcție de mai multe variabile
    """
)

# ============= Date =============
if "main_df" not in st.session_state:
    st.session_state["main_df"] = build_main_dataset()

df = st.session_state["main_df"].copy()

# Pregătim datele - eliminăm NaN-uri pe coloanele esențiale pentru modelare
required_cols = ["age", "height_in_cm", "market_value_in_eur",
                 "total_goals", "total_assists", "total_minutes", "goals_per_90"]
available_cols = [c for c in required_cols if c in df.columns]
df_clean = df.dropna(subset=available_cols).copy()

# Filtrăm doar jucători cu cel puțin un meci jucat
if "total_minutes" in df_clean.columns:
    df_clean = df_clean[df_clean["total_minutes"] > 0].copy()

st.success(
    f"📊 Date pregătite pentru modelare: **{len(df_clean):,} jucători** cu informații complete."
    .replace(",", ".")
)

st.divider()

tab_kmeans, tab_logistic, tab_multiple = st.tabs([
    "🎯 1. K-Means Clustering",
    "🔍 2. Regresie Logistică",
    "📈 3. Regresie Multiplă"
])

# ===================== TAB 1: K-MEANS =====================
with tab_kmeans:
    st.subheader("K-Means Clustering – Segmentarea jucătorilor")
    st.markdown(
        """
        **K-Means** este un algoritm de clustering nesupervizat care grupează jucătorii în
        **K segmente** pe baza similitudinii lor. Util pentru a identifica:
        - Profile de jucători (ex: "atacanți prolifici", "fundași defensivi", "tineri promițători")
        - Jucători similari cu cei pe care îi căutăm
        """
    )

    # Selectarea features-urilor
    st.markdown("### Pasul 1: Alege variabilele pentru clustering")
    cluster_features_options = [c for c in [
        "age", "height_in_cm", "market_value_in_eur",
        "total_goals", "total_assists", "total_minutes",
        "goals_per_90", "assists_per_90", "games_played",
        "total_yellow_cards", "total_red_cards"
    ] if c in df_clean.columns]

    cluster_features = st.multiselect(
        "Variabile (recomandat: 2-5 pentru claritate):",
        options=cluster_features_options,
        default=[c for c in ["age", "market_value_in_eur", "goals_per_90", "total_minutes"]
                 if c in cluster_features_options]
    )

    if len(cluster_features) >= 2:
        # Pregătim datele
        X_cluster = df_clean[cluster_features].copy()
        scaler_km = StandardScaler()
        X_scaled = scaler_km.fit_transform(X_cluster)

        st.markdown("### Pasul 2: Determină numărul optim de clustere (Metoda Cotului)")

        # Elbow Method
        max_k = st.slider("Test până la K=", min_value=4, max_value=12, value=8)

        with st.spinner("Calculez WCSS pentru diferite valori K..."):
            wcss = []
            silhouettes = []
            for k in range(2, max_k + 1):
                km_temp = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
                km_temp.fit(X_scaled)
                wcss.append(km_temp.inertia_)
                if k >= 2:
                    sil = silhouette_score(X_scaled, km_temp.labels_, sample_size=min(5000, len(X_scaled)), random_state=42)
                    silhouettes.append(sil)

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(range(2, max_k + 1), wcss, marker="o", color="#e41a1c", linewidth=2)
            ax.set_xlabel("Număr de clustere (K)")
            ax.set_ylabel("WCSS (Within-Cluster Sum of Squares)")
            ax.set_title("Metoda Cotului (Elbow Method)")
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(range(2, max_k + 1), silhouettes, marker="s", color="#377eb8", linewidth=2)
            ax.set_xlabel("Număr de clustere (K)")
            ax.set_ylabel("Silhouette Score")
            ax.set_title("Silhouette Score (mai mare = mai bine)")
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("### Pasul 3: Aplică K-Means cu K-ul ales")

        n_clusters = st.slider("Numărul de clustere (K):", min_value=2, max_value=max_k, value=4)

        if st.button("🚀 Antrenează K-Means", type="primary"):
            kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            df_clean["cluster"] = cluster_labels

            sil_score = silhouette_score(X_scaled, cluster_labels, sample_size=min(5000, len(X_scaled)), random_state=42)
            st.success(f"✅ Model antrenat. **Silhouette Score = {sil_score:.4f}**")

            if sil_score > 0.5:
                st.info("✨ Score excelent — clustere bine definite!")
            elif sil_score > 0.3:
                st.info("👍 Score bun — clustere acceptabile.")
            else:
                st.warning("⚠️ Score slab — încearcă altă combinație de features sau alt K.")

            # Profilul fiecărui cluster
            st.markdown("### 📊 Profilul fiecărui cluster (medii)")
            cluster_profile = df_clean.groupby("cluster")[cluster_features].mean().round(2)
            cluster_profile["nr_jucători"] = df_clean.groupby("cluster").size()
            st.dataframe(cluster_profile, use_container_width=True)

            # Vizualizare scatter (primele 2 features)
            if len(cluster_features) >= 2:
                st.markdown("### 🎨 Vizualizare clustere (primele 2 features)")
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
                for i in range(n_clusters):
                    cluster_data = df_clean[df_clean["cluster"] == i]
                    ax.scatter(
                        cluster_data[cluster_features[0]],
                        cluster_data[cluster_features[1]],
                        c=[colors[i]], label=f"Cluster {i} (n={len(cluster_data)})",
                        alpha=0.5, s=20
                    )
                ax.set_xlabel(cluster_features[0])
                ax.set_ylabel(cluster_features[1])
                ax.set_title(f"K-Means: {cluster_features[0]} vs {cluster_features[1]}")
                ax.legend(loc="best")
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)

            # Salvăm cluster labels pentru pagina recomandări
            st.session_state["clustered_df"] = df_clean
            st.session_state["cluster_features"] = cluster_features

            # Sample din fiecare cluster
            st.markdown("### 👤 Exemple de jucători din fiecare cluster")
            for c in range(n_clusters):
                with st.expander(f"Cluster {c} – {len(df_clean[df_clean['cluster']==c])} jucători"):
                    sample = df_clean[df_clean["cluster"] == c].nlargest(5, "total_minutes")
                    show_cols = ["name", "age", "position", "current_club_name",
                                 "market_value_in_eur"] + cluster_features
                    show_cols = list(dict.fromkeys([c2 for c2 in show_cols if c2 in sample.columns]))
                    st.dataframe(sample[show_cols].reset_index(drop=True), use_container_width=True)
    else:
        st.warning("⚠️ Selectează cel puțin 2 features pentru clustering.")

# ===================== TAB 2: REGRESIE LOGISTICĂ =====================
with tab_logistic:
    st.subheader("Regresie Logistică – Predicția jucătorilor 'Top-Tier'")
    st.markdown(
        r"""
        **Întrebarea de business:** Putem prezice dacă un jucător este "top-tier" (de top), 
        adică are o valoare de piață **peste mediană**, pe baza statisticilor de performanță?

        Dacă da, putem identifica **jucători subevaluați** — performanță bună, dar valoare scăzută.
        Aceștia sunt ținte ideale pentru Oțelul.

        **Funcția sigmoid** transformă scorul brut într-o probabilitate între 0 și 1:
        $\sigma(z) = \frac{1}{1 + e^{-z}}$
        """
    )

    # Construim variabila țintă
    median_value = df_clean["market_value_in_eur"].median()
    df_log = df_clean.copy()
    df_log["is_top_tier"] = (df_log["market_value_in_eur"] > median_value).astype(int)

    st.info(
        f"💡 **Variabila țintă:** `is_top_tier = 1` dacă valoarea de piață > "
        f"**{median_value:,.0f} €** (mediana), altfel 0.".replace(",", ".")
    )

    # Distribuția claselor
    class_counts = df_log["is_top_tier"].value_counts()
    c1, c2 = st.columns(2)
    c1.metric("Top-Tier (1)", f"{class_counts.get(1, 0):,}".replace(",", "."))
    c2.metric("Non Top-Tier (0)", f"{class_counts.get(0, 0):,}".replace(",", "."))

    st.markdown("### Pasul 1: Alege features-urile")
    log_feature_options = [c for c in [
        "age", "height_in_cm", "total_goals", "total_assists",
        "total_minutes", "goals_per_90", "assists_per_90",
        "games_played", "total_yellow_cards", "total_red_cards",
        "club_total_market_value", "club_squad_size", "club_average_age"
    ] if c in df_log.columns]

    log_features = st.multiselect(
        "Features (variabile predictoare):",
        options=log_feature_options,
        default=[c for c in [
            "age", "height_in_cm", "total_goals", "total_assists",
            "goals_per_90", "total_minutes", "club_total_market_value"
        ] if c in log_feature_options]
    )

    test_size = st.slider("Procent date pentru test (%):", 10, 40, 20) / 100

    if log_features and st.button("🚀 Antrenează Regresia Logistică", type="primary"):
        # Pregătire date
        X = df_log[log_features].fillna(df_log[log_features].median())
        y = df_log["is_top_tier"]

        # Scalare
        scaler_lr = StandardScaler()
        X_scaled = scaler_lr.fit_transform(X)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )

        # Training
        log_model = LogisticRegression(max_iter=1000, penalty="l2", solver="lbfgs", random_state=42)
        log_model.fit(X_train, y_train)

        y_pred = log_model.predict(X_test)
        y_proba = log_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        st.success(f"✅ Model antrenat!")

        c1, c2, c3 = st.columns(3)
        c1.metric("Acuratețe", f"{accuracy:.4f}")
        c2.metric("F1-Score", f"{f1:.4f}")
        c3.metric("ROC AUC", f"{auc:.4f}")

        # Confusion Matrix
        st.markdown("### 📊 Matricea de confuzie")
        cm = confusion_matrix(y_test, y_pred)
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Non Top-Tier", "Top-Tier"],
                        yticklabels=["Non Top-Tier", "Top-Tier"])
            ax.set_xlabel("Predicție")
            ax.set_ylabel("Real")
            ax.set_title("Confusion Matrix – Regresie Logistică")
            st.pyplot(fig)
            plt.close(fig)

        with c2:
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(fpr, tpr, color="#e41a1c", linewidth=2, label=f"Model (AUC = {auc:.3f})")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="No Skill")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Curba ROC")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        # Classification Report
        st.markdown("### 📋 Classification Report")
        report = classification_report(y_test, y_pred,
                                        target_names=["Non Top-Tier", "Top-Tier"],
                                        output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

        # Coefficients (importanța features)
        st.markdown("### 🎯 Importanța variabilelor (coeficienți)")
        coef_df = pd.DataFrame({
            "Variabilă": log_features,
            "Coeficient": log_model.coef_[0],
            "Abs": np.abs(log_model.coef_[0])
        }).sort_values("Abs", ascending=False).drop(columns="Abs")

        fig, ax = plt.subplots(figsize=(9, max(3, len(log_features) * 0.4)))
        colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in coef_df["Coeficient"]]
        ax.barh(coef_df["Variabilă"], coef_df["Coeficient"], color=colors, edgecolor="black")
        ax.set_xlabel("Coeficient")
        ax.set_title("Coeficienții Regresiei Logistice (verde = crește P, roșu = scade P)")
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.8)
        ax.grid(axis="x", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        # Salvăm pentru pagina de recomandări — jucători cu probabilitate mare
        # de a fi top-tier dar care în realitate sunt sub mediană (subevaluați)
        df_log["proba_top_tier"] = log_model.predict_proba(scaler_lr.transform(X))[:, 1]
        st.session_state["logistic_df"] = df_log
        st.session_state["logistic_features"] = log_features

# ===================== TAB 3: REGRESIE MULTIPLĂ =====================
with tab_multiple:
    st.subheader("Regresie Multiplă – Predicția valorii de piață")
    st.markdown(
        """
        Folosim **statsmodels** pentru a construi un model de regresie multiplă care prezice
        **valoarea de piață** a unui jucător pe baza mai multor variabile (vârstă, performanță, club etc.).

        Avantajul `statsmodels` față de `sklearn`: oferă **statistici detaliate** (p-values,
        intervale de încredere, R² ajustat) — esențial pentru analiza științifică.
        """
    )

    st.markdown("### Pasul 1: Alege features-urile")
    multi_options = [c for c in [
        "age", "height_in_cm", "total_goals", "total_assists",
        "total_minutes", "goals_per_90", "assists_per_90",
        "games_played", "total_yellow_cards",
        "club_total_market_value", "club_squad_size", "club_average_age"
    ] if c in df_clean.columns]

    multi_features = st.multiselect(
        "Features (X):",
        options=multi_options,
        default=[c for c in [
            "age", "height_in_cm", "total_goals", "total_assists",
            "total_minutes", "club_total_market_value"
        ] if c in multi_options]
    )

    use_log_target = st.checkbox(
        "📐 Aplică `log(market_value_in_eur)` ca țintă (recomandat — distribuția e foarte asimetrică)",
        value=True
    )

    if multi_features and st.button("🚀 Antrenează Regresia Multiplă", type="primary"):
        # Pregătim datele
        X = df_clean[multi_features].fillna(df_clean[multi_features].median())
        if use_log_target:
            y = np.log1p(df_clean["market_value_in_eur"])
            target_name = "log(market_value)"
        else:
            y = df_clean["market_value_in_eur"]
            target_name = "market_value (€)"

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # === statsmodels OLS (cu detalii) ===
        X_train_sm = sm.add_constant(X_train)
        X_test_sm = sm.add_constant(X_test)
        ols_model = sm.OLS(y_train, X_train_sm).fit()

        st.success("✅ Model antrenat (statsmodels OLS)!")

        # === Metrici ===
        y_pred = ols_model.predict(X_test_sm)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R² (test)", f"{r2:.4f}")
        c2.metric("MAE", f"{mae:.3f}" if use_log_target else f"{mae:,.0f}")
        c3.metric("RMSE", f"{rmse:.3f}" if use_log_target else f"{rmse:,.0f}")
        c4.metric("R² ajustat", f"{ols_model.rsquared_adj:.4f}")

        # === Sumar statistic ===
        st.markdown("### 📋 Sumar statistic (statsmodels)")
        with st.expander("Vezi sumarul complet OLS", expanded=False):
            st.text(ols_model.summary())

        # Tabel cu coeficienți și p-values
        coef_table = pd.DataFrame({
            "Variabilă": ols_model.params.index,
            "Coeficient": ols_model.params.values,
            "Std Error": ols_model.bse.values,
            "t-statistic": ols_model.tvalues.values,
            "P-value": ols_model.pvalues.values
        })
        coef_table["Semnificativ?"] = coef_table["P-value"].apply(
            lambda p: "✅ Da (p<0.05)" if p < 0.05 else "❌ Nu"
        )
        st.markdown("**Coeficienți și semnificație statistică:**")
        st.dataframe(coef_table.round(4), use_container_width=True, hide_index=True)

        # === Vizualizări ===
        st.markdown("### 🎨 Vizualizare predicții vs. valori reale")
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(y_test, y_pred, alpha=0.4, color="#3498db", s=20)
            min_v, max_v = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            ax.plot([min_v, max_v], [min_v, max_v], "--", color="red", linewidth=2, label="Linia ideală")
            ax.set_xlabel(f"Real: {target_name}")
            ax.set_ylabel(f"Predicție: {target_name}")
            ax.set_title(f"Predicție vs. Real (R² = {r2:.3f})")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        with c2:
            # Reziduuri
            residuals = y_test - y_pred
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(y_pred, residuals, alpha=0.4, color="#9b59b6", s=20)
            ax.axhline(0, color="red", linestyle="--", linewidth=2)
            ax.set_xlabel(f"Predicție: {target_name}")
            ax.set_ylabel("Reziduuri (Real - Predicție)")
            ax.set_title("Plot Reziduuri (ar trebui aleator în jurul 0)")
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        # === Predicție pentru un jucător ipotetic ===
        st.markdown("### 🎮 Predicție interactivă pentru un jucător ipotetic")
        st.caption("Setează valorile features-urilor și vezi predicția modelului.")

        input_values = {}
        cols = st.columns(min(3, len(multi_features)))
        for i, feat in enumerate(multi_features):
            with cols[i % len(cols)]:
                val = st.number_input(
                    feat,
                    value=float(df_clean[feat].median()),
                    step=1.0,
                    key=f"pred_{feat}"
                )
                input_values[feat] = val

        if st.button("🔮 Calculează predicția"):
            input_arr = pd.DataFrame([input_values])[multi_features]
            input_with_const = sm.add_constant(input_arr, has_constant="add")
            pred = ols_model.predict(input_with_const)[0]

            if use_log_target:
                actual_value = np.expm1(pred)
                st.success(
                    f"💰 Valoare estimată: **{actual_value:,.0f} €** "
                    f"(log = {pred:.3f})".replace(",", ".")
                )
            else:
                st.success(f"💰 Valoare estimată: **{pred:,.0f} €**".replace(",", "."))

st.divider()
st.info("👉 **Pasul final:** mergi la pagina **🎯 Recommendations** pentru recomandările concrete pentru Oțelul!")
