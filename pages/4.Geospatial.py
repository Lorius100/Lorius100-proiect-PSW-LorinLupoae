"""
Pagina 4: Geospatial Analysis
- Folosirea geopandas pentru hărți europene
- Folium pentru hărți interactive
- Vizualizare cluburi pe hartă, originea jucătorilor
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from utils.data_loader import build_main_dataset, load_clubs, load_competitions, get_european_country_coords

st.set_page_config(page_title="Geospatial Analysis", page_icon="🗺️", layout="wide")
st.title("🗺️ Geospatial Analysis – Analiză spațială")

st.markdown(
    """
    În această pagină folosim **GeoPandas** și **Folium** pentru a vizualiza geografic datele
    despre cluburi și jucători. Aceasta este una dintre cele mai puternice metode de a identifica
    **piețe de scouting** profitabile pentru SC Oțelul Galați.
    """
)

# ============= Date =============
if "main_df" not in st.session_state:
    df = build_main_dataset()
    st.session_state["main_df"] = df

df = st.session_state["main_df"].copy()
clubs = load_clubs()
comps = load_competitions()
country_coords = get_european_country_coords()

st.divider()

tab1, tab2, tab3 = st.tabs([
    "🌍 Distribuția jucătorilor pe țări",
    "🏟️ Cluburi pe hartă",
    "💎 Originea jucătorilor valoroși"
])

# ===================== TAB 1: DISTRIBUȚIA JUCĂTORILOR =====================
with tab1:
    st.subheader("Distribuția jucătorilor după țara de cetățenie")
    st.markdown(
        "Vedem **din ce țări provin jucătorii** activi în Europa. Țările cu mulți jucători "
        "exportați sunt piețe de scouting interesante (ex: Brazilia, Argentina, fosta Iugoslavie)."
    )

    # Top țări după număr de jucători
    top_countries = (
        df["country_of_citizenship"]
        .value_counts()
        .head(20)
        .reset_index()
    )
    top_countries.columns = ["Țară", "Nr. jucători"]

    # Adăugăm valoarea medie de piață
    avg_value = (
        df.groupby("country_of_citizenship")["market_value_in_eur"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    avg_value.columns = ["Țară", "Valoare medie €", "Valoare mediană €", "Nr. jucători"]
    avg_value = avg_value.sort_values("Nr. jucători", ascending=False).head(20)
    avg_value["Valoare medie €"] = avg_value["Valoare medie €"].round(0).astype(int)
    avg_value["Valoare mediană €"] = avg_value["Valoare mediană €"].round(0).astype(int)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Top 20 țări după număr de jucători activi:**")
        st.dataframe(avg_value, use_container_width=True, hide_index=True)

    with c2:
        fig, ax = plt.subplots(figsize=(7, 9))
        ax.barh(top_countries["Țară"], top_countries["Nr. jucători"],
                color="#2c7fb8", edgecolor="black")
        ax.set_xlabel("Nr. jucători")
        ax.set_title("Top 20 țări (cetățenie)")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # Hartă Folium cu numărul de jucători din țări europene
    st.markdown("---")
    st.markdown("### 🗺️ Hartă interactivă — jucători pe țări europene")
    st.markdown("Mărimea cercurilor reflectă numărul de jucători din fiecare țară.")

    m = folium.Map(location=[50, 15], zoom_start=4, tiles="cartodbpositron")

    european_countries_count = (
        df[df["country_of_citizenship"].isin(country_coords.keys())]
        .groupby("country_of_citizenship")
        .agg(
            n_players=("player_id", "count"),
            avg_value=("market_value_in_eur", "mean")
        )
        .reset_index()
    )

    max_count = european_countries_count["n_players"].max()
    for _, row in european_countries_count.iterrows():
        country = row["country_of_citizenship"]
        if country not in country_coords:
            continue
        lat, lon = country_coords[country]
        # Mărimea cercului: proporțional cu nr jucători
        radius = 5 + (row["n_players"] / max_count) * 30

        avg_val_mil = (row["avg_value"] / 1e6) if pd.notna(row["avg_value"]) else 0

        popup_html = f"""
        <b>{country}</b><br>
        Jucători: <b>{int(row['n_players'])}</b><br>
        Val. medie: <b>{avg_val_mil:.2f} mil €</b>
        """
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"{country}: {int(row['n_players'])} jucători",
            color="#1f78b4",
            fill=True,
            fill_color="#1f78b4",
            fill_opacity=0.6,
            weight=2
        ).add_to(m)

    # Marker special pentru Galați (Otelul)
    folium.Marker(
        location=[45.4353, 28.0080],
        popup="<b>SC Oțelul Galați</b><br>Stadionul Oțelul",
        tooltip="Oțelul Galați (Casa)",
        icon=folium.Icon(color="red", icon="star", prefix="fa")
    ).add_to(m)

    st_folium(m, width=None, height=550, returned_objects=[])

# ===================== TAB 2: CLUBURI =====================
with tab2:
    st.subheader("Distribuția cluburilor pe ligi naționale")

    # Calculăm valoarea totală pentru fiecare club din valorile jucătorilor
    # (coloana total_market_value din clubs.csv este goală în setul de date)
    club_values = (
        df.groupby("current_club_id")["market_value_in_eur"]
        .sum()
        .reset_index()
        .rename(columns={"current_club_id": "club_id", "market_value_in_eur": "total_market_value"})
    )

    # Eliminăm coloana goală din clubs ca să evităm coliziunea de nume la merge
    clubs_clean = clubs.drop(columns=["total_market_value"], errors="ignore")

    # Filtrăm doar cluburi din ligi domestice europene
    clubs_with_comp = clubs_clean.merge(club_values, on="club_id", how="left")
    clubs_with_comp = clubs_with_comp.merge(
        comps[["competition_id", "country_name", "type"]],
        left_on="domestic_competition_id",
        right_on="competition_id",
        how="left"
    )
    clubs_with_comp = clubs_with_comp[
        clubs_with_comp["country_name"].isin(country_coords.keys())
    ]
    # Completăm cluburile fără jucători activi cu 0
    clubs_with_comp["total_market_value"] = clubs_with_comp["total_market_value"].fillna(0)

    # Agregare pe țară
    clubs_by_country = (
        clubs_with_comp.groupby("country_name")
        .agg(
            n_clubs=("club_id", "count"),
            total_value=("total_market_value", "sum"),
            avg_value=("total_market_value", "mean")
        )
        .reset_index()
        .sort_values("total_value", ascending=False)
    )
    clubs_by_country["total_value_mil"] = (clubs_by_country["total_value"] / 1e6).round(1)
    clubs_by_country["avg_value_mil"] = (clubs_by_country["avg_value"] / 1e6).round(1)

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("**Cluburi pe țară:**")
        display_df = clubs_by_country[["country_name", "n_clubs", "total_value_mil", "avg_value_mil"]].copy()
        display_df.columns = ["Țară", "Nr. cluburi", "Valoare totală (mil €)", "Valoare medie (mil €)"]
        st.dataframe(display_df.head(20), use_container_width=True, hide_index=True)

    with c2:
        # Hartă cu cluburile pe țări
        m2 = folium.Map(location=[50, 15], zoom_start=4, tiles="cartodbpositron")

        max_total = clubs_by_country["total_value"].max()
        for _, row in clubs_by_country.iterrows():
            country = row["country_name"]
            if country not in country_coords:
                continue
            lat, lon = country_coords[country]
            radius = 5 + (row["total_value"] / max_total) * 35

            popup_html = f"""
            <b>{country}</b><br>
            Cluburi: <b>{int(row['n_clubs'])}</b><br>
            Val. totală: <b>{row['total_value_mil']:.0f} mil €</b><br>
            Val. medie: <b>{row['avg_value_mil']:.1f} mil €</b>
            """

            color = "#e31a1c" if row["total_value_mil"] > 1000 else \
                    "#fd8d3c" if row["total_value_mil"] > 200 else "#fecc5c"

            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                popup=folium.Popup(popup_html, max_width=200),
                tooltip=f"{country}: {row['total_value_mil']:.0f} mil €",
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                weight=2
            ).add_to(m2)

        folium.Marker(
            location=[45.4353, 28.0080],
            popup="<b>SC Oțelul Galați</b>",
            tooltip="Oțelul Galați",
            icon=folium.Icon(color="red", icon="star", prefix="fa")
        ).add_to(m2)

        st_folium(m2, width=None, height=550, returned_objects=[])

# ===================== TAB 3: JUCĂTORI VALOROȘI ACCESIBILI =====================
with tab3:
    st.subheader("Originea jucătorilor accesibili pentru Oțelul (val. < 2 mil €)")
    st.markdown(
        """
        Aici filtrăm jucători cu **valoare de piață sub 2 milioane €** (buget realist pentru Oțelul)
        și vizualizăm țările lor de origine. Țările cu **densitate mare** sunt piețe de scouting
        cost-eficiente.
        """
    )

    # Filtru de buget
    budget_max = st.slider(
        "Buget maxim per jucător (€):",
        min_value=100_000, max_value=5_000_000,
        value=2_000_000, step=100_000,
        format="%d"
    )

    age_range = st.slider(
        "Interval de vârstă:",
        min_value=16, max_value=40,
        value=(20, 30)
    )

    affordable = df[
        (df["market_value_in_eur"] <= budget_max) &
        (df["market_value_in_eur"] > 0) &
        (df["age"] >= age_range[0]) &
        (df["age"] <= age_range[1])
    ].copy()

    st.metric(
        "Jucători accesibili în acest filtru",
        f"{len(affordable):,}".replace(",", ".")
    )

    # Top țări de unde provin acești jucători
    top_origin = (
        affordable["country_of_citizenship"]
        .value_counts()
        .head(15)
        .reset_index()
    )
    top_origin.columns = ["Țară", "Nr. jucători accesibili"]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top 15 țări de origine a jucătorilor accesibili:**")
        st.dataframe(top_origin, use_container_width=True, hide_index=True)

    with c2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top_origin["Țară"], top_origin["Nr. jucători accesibili"],
                color="#33a02c", edgecolor="black")
        ax.set_xlabel("Nr. jucători")
        ax.set_title(f"Țările cu cei mai mulți jucători accesibili\n(val. ≤ {budget_max/1e6:.1f} mil €)")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # Hartă cu jucători accesibili
    st.markdown("### 🗺️ Hartă: țări cu jucători accesibili (Europa)")
    m3 = folium.Map(location=[50, 15], zoom_start=4, tiles="cartodbpositron")

    affordable_eu = affordable[
        affordable["country_of_citizenship"].isin(country_coords.keys())
    ].groupby("country_of_citizenship").size().reset_index(name="n")

    max_n = affordable_eu["n"].max() if len(affordable_eu) > 0 else 1
    for _, row in affordable_eu.iterrows():
        country = row["country_of_citizenship"]
        lat, lon = country_coords[country]
        radius = 5 + (row["n"] / max_n) * 25
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            popup=f"<b>{country}</b>: {int(row['n'])} jucători accesibili",
            tooltip=f"{country}: {int(row['n'])}",
            color="#33a02c",
            fill=True,
            fill_color="#33a02c",
            fill_opacity=0.7
        ).add_to(m3)

    folium.Marker(
        location=[45.4353, 28.0080],
        popup="<b>SC Oțelul Galați</b>",
        icon=folium.Icon(color="red", icon="star", prefix="fa")
    ).add_to(m3)

    st_folium(m3, width=None, height=500, returned_objects=[])

st.divider()
st.info("👉 **Următorul pas:** mergi la pagina **🤖 Modeling** pentru K-Means, Regresie Logistică și Multiplă.")
