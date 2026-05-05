"""
Pachete Software - Proiect
SC Oțelul Galați - Analiză și recomandări de transferuri
Pagina principală: Home
"""
import streamlit as st
import pandas as pd
from utils.data_loader import (
    load_players, load_clubs, load_competitions, file_exists
)

# ===================== Configurare pagină =====================
st.set_page_config(
    page_title="Oțelul Galați – Analiză Transferuri",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================== Header =====================
st.title("⚽ SC Oțelul Galați — Analiză și recomandări de transferuri")
st.markdown(
    "**Proiect Pachete Software** — Facultatea CSIE, anul III  \n"
    "Aplicație Streamlit pentru analiza performanței jucătorilor europeni "
    "și identificarea de ținte de transfer pentru SC Oțelul Galați."
)

st.divider()

# ===================== Verificare date =====================
required_files = [
    "players.csv", "clubs.csv", "competitions.csv",
    "player_valuations.csv", "appearances.csv", "transfers.csv"
]
missing = [f for f in required_files if not file_exists(f)]

if missing:
    st.error(
        "⚠️ **Fișiere lipsă în folderul `data/`:**\n\n"
        + "\n".join([f"- `{f}`" for f in missing])
        + "\n\n**Cum repari:** copiază fișierele CSV în folderul `data/` din rădăcina proiectului. "
        "Vezi README.md pentru detalii."
    )
    st.stop()

# ===================== Introducere =====================
col_intro, col_logo = st.columns([3, 1])
with col_intro:
    st.subheader("📖 Introducere")
    st.markdown(
        """
        **SC Oțelul Galați** este un club profesionist de fotbal fondat în **1964**, cu sediul în Galați,
        în partea de est a României. Clubul este unul dintre cele mai emblematice nume din fotbalul
        românesc, câștigând titlul în Liga I în sezonul **2011–2012** — o performanță remarcabilă care
        i-a consolidat locul în istoria fotbalului național.

        După ani petrecuți în diviziile inferioare, Oțelul Galați a revenit în primul eșalon al
        fotbalului românesc, unde se confruntă cu provocarea de a-și consolida poziția și de a
        concura la un nivel mai ridicat.

        Clubul activează cu un buget moderat față de marile puteri ale fotbalului românesc
        (FCSB, CFR Cluj, Rapid București), ceea ce face ca **deciziile bazate pe date** să
        reprezinte un avantaj esențial în recrutare.
        """
    )

with col_logo:
    st.markdown("### 🎯 Obiectiv")
    st.info(
        "**Identificarea a 2-3 jucători** în poziții cheie pentru a califica clubul "
        "în play-off-ul sezonului următor din Liga I, cu un buget rezonabil."
    )

st.divider()

# ===================== Scopul proiectului =====================
st.subheader("🎯 Scopul proiectului")
st.markdown(
    """
    Acest proiect analizează datele de performanță ale jucătorilor din **ligile europene**
    pentru a susține strategia de scouting și recrutare a clubului SC Oțelul Galați.

    Prin combinarea **modelelor statistice**, a tehnicilor de **clusterizare** și a **analizei
    geospațiale** a originii jucătorilor și locațiilor cluburilor, aplicația oferă recomandări
    fundamentate pe date pentru transferuri țintite și eficiente din punct de vedere financiar.
    """
)

# ===================== KPIs =====================
st.subheader("📊 Indicatori cheie (KPI)")

players = load_players()
clubs = load_clubs()
comps = load_competitions()

# Filtrăm la jucători activi (cei cu last_season recent)
active_players = players[players["last_season"] >= 2023]

c1, c2, c3, c4 = st.columns(4)
c1.metric("👥 Jucători activi", f"{len(active_players):,}".replace(",", "."))
c2.metric("🏟️ Cluburi", f"{len(clubs):,}".replace(",", "."))
c3.metric("🏆 Competiții", f"{len(comps):,}".replace(",", "."))
c4.metric(
    "🌍 Țări reprezentate",
    f"{active_players['country_of_citizenship'].nunique():,}".replace(",", ".")
)

# A doua linie de KPIs
c5, c6, c7, c8 = st.columns(4)
total_market = active_players["market_value_in_eur"].sum() / 1e9
avg_market = active_players["market_value_in_eur"].mean() / 1e6
median_age = active_players["age"].median()
domestic_leagues = comps[comps["type"] == "domestic_league"]

c5.metric("💰 Valoare totală piață", f"{total_market:,.1f} mld €")
c6.metric("💵 Valoare medie / jucător", f"{avg_market:,.2f} mil €")
c7.metric("📅 Vârstă mediană", f"{median_age:,.1f} ani")
c8.metric("🇪🇺 Ligi naționale", f"{len(domestic_leagues)}")

st.divider()

# ===================== Structura aplicației =====================
st.subheader("🗺️ Structura aplicației")
st.markdown(
    """
    Aplicația este organizată pe **6 pagini** care urmează fluxul natural al unui proiect de
    analiză de date — de la explorare la recomandări finale:

    | Pagină | Descriere | Funcționalități acoperite |
    |--------|-----------|---------------------------|
    | **📂 Data Overview** | Inspectarea datelor: dimensiuni, tipuri, valori lipsă, statistici descriptive | `streamlit display`, `pandas grouping` |
    | **🧹 Data Cleaning** | Tratarea valorilor lipsă și a outlier-ilor (alegere interactivă) | tratarea valorilor lipsă, tratarea valorilor extreme |
    | **🔢 Encoding & Scaling** | Codificare variabile categorice + scalare numerică | One-Hot, Label, Target Encoding, StandardScaler, MinMaxScaler |
    | **🗺️ Geospatial** | Hărți europene cu cluburi și jucători (densitate, valori) | `geopandas`, `folium` |
    | **🤖 Modeling** | K-Means, Regresie Logistică, Regresie Multiplă | `scikit-learn`, `statsmodels` |
    | **🎯 Recommendations** | Top 2-3 jucători recomandați pentru Oțelul, cu argumentare | sinteză + filtrare avansată |

    👉 **Folosește meniul din stânga (sidebar) pentru a naviga între pagini.**
    """
)

st.divider()

# ===================== Liga 1 România =====================
st.subheader("🇷🇴 Liga 1 România — context")

ro_players = active_players[
    active_players["current_club_domestic_competition_id"] == "RO1"
]
ro_clubs = clubs[clubs["domestic_competition_id"] == "RO1"].copy()
# Eliminăm coloana goală pentru a evita coliziunea de nume la merge
ro_clubs = ro_clubs.drop(columns=["total_market_value"], errors="ignore")

# Calculăm valoarea totală a fiecărui club RO din valorile jucătorilor
# (coloana total_market_value din clubs.csv este goală)
ro_club_values = (
    ro_players.groupby("current_club_id")["market_value_in_eur"]
    .sum()
    .reset_index()
    .rename(columns={"current_club_id": "club_id", "market_value_in_eur": "total_market_value"})
)
ro_clubs = ro_clubs.merge(ro_club_values, on="club_id", how="left")
ro_clubs["total_market_value"] = ro_clubs["total_market_value"].fillna(0)

cR1, cR2, cR3, cR4 = st.columns(4)
cR1.metric("Cluburi în Liga 1", f"{len(ro_clubs)}")
cR2.metric("Jucători activi", f"{len(ro_players):,}".replace(",", "."))
cR3.metric(
    "Valoare medie jucător RO",
    f"{(ro_players['market_value_in_eur'].mean() / 1e6):.2f} mil €"
)
cR4.metric(
    "Valoare totală Liga 1",
    f"{(ro_clubs['total_market_value'].sum() / 1e6):.0f} mil €"
)

# Top cluburi din Liga 1 după valoarea lotului
if len(ro_clubs) > 0:
    st.markdown("**Top cluburi Liga 1 după valoarea totală a lotului:**")
    top_ro = (
        ro_clubs.sort_values("total_market_value", ascending=False)
        .head(8)[["name", "total_market_value", "squad_size", "average_age", "stadium_name"]]
        .reset_index(drop=True)
    )
    top_ro["total_market_value"] = (top_ro["total_market_value"] / 1e6).round(2)
    top_ro = top_ro.rename(columns={
        "name": "Club",
        "total_market_value": "Valoare lot (mil €)",
        "squad_size": "Mărime lot",
        "average_age": "Vârstă medie",
        "stadium_name": "Stadion"
    })
    st.dataframe(top_ro, use_container_width=True, hide_index=True)

st.divider()

# ===================== Footer =====================
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 0.85em;">
    Proiect realizat pentru disciplina <b>Pachete Software</b> · Facultatea CSIE, anul III · 2024-2025<br>
    Sursă date: <a href="https://www.kaggle.com/datasets/davidcariboo/player-scores">
    Kaggle - Football Data from Transfermarkt</a>
    </div>
    """,
    unsafe_allow_html=True
)
