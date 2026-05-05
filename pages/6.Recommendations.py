"""
Pagina 6: Recommendations
- Filtrare jucători potriviți pentru Oțelul Galați
- Top 2-3 recomandări per poziție cheie
- Argumentare bazată pe date
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium

from utils.data_loader import build_main_dataset, load_clubs, get_european_country_coords

st.set_page_config(page_title="Recommendations", page_icon="🎯", layout="wide")
st.title("🎯 Recommendations – Recomandări pentru SC Oțelul Galați")

st.markdown(
    """
    Aceasta este **pagina finală** care răspunde la întrebarea de business:

    > _"Ce 2-3 jucători ar trebui SC Oțelul Galați să cumpere pentru a-și maximiza șansele de
    > a se califica în play-off-ul sezonului următor din Liga I?"_

    Ne bazăm pe datele și modelele construite în paginile anterioare. Aplicăm un set de **filtre
    realiste** (buget, vârstă, performanță) și rankuim jucătorii.
    """
)

# ============= Date =============
if "main_df" not in st.session_state:
    st.session_state["main_df"] = build_main_dataset()

df = st.session_state["main_df"].copy()
clubs = load_clubs()
country_coords = get_european_country_coords()

# Eliminăm jucători din Oțelul (nu vrem să-i recomandăm pe cei pe care îi avem deja)
# Folosim un pattern care prinde toate variantele de scriere ale "Oțelul" / "Galați"
import re
otelul_pattern = r"o[țţt]elul.*gala[țţt]i|o[țţt]elul"
otelul = df[df["current_club_name"].fillna("").str.contains(otelul_pattern, case=False, regex=True, na=False)]
otelul_id = otelul["current_club_id"].iloc[0] if len(otelul) > 0 else None

if otelul_id is not None:
    df_candidates = df[df["current_club_id"] != otelul_id].copy()
else:
    df_candidates = df.copy()

st.divider()

# ===================== Context Oțelul =====================
st.subheader("📌 Contextul Oțelului Galați")

c1, c2, c3, c4 = st.columns(4)
if otelul_id is not None and len(otelul) > 0:
    c1.metric("Jucători în lot", len(otelul))
    c2.metric(
        "Valoare lot",
        f"{(otelul['market_value_in_eur'].sum() / 1e6):.2f} mil €"
    )
    c3.metric(
        "Vârstă medie lot",
        f"{otelul['age'].mean():.1f} ani"
    )
    c4.metric(
        "Goluri totale (2 sez.)",
        f"{otelul['total_goals'].sum():.0f}" if 'total_goals' in otelul.columns else "N/A"
    )
else:
    st.warning("Nu am găsit Oțelul în date — folosim toate cluburile ca țintă pentru recomandări.")

st.divider()

# ===================== Configurare filtre =====================
st.subheader("⚙️ Configurare filtre pentru recomandări")

col_buget, col_varsta, col_minute = st.columns(3)

with col_buget:
    st.markdown("**💰 Buget**")
    budget = st.slider(
        "Valoare maximă jucător (€):",
        min_value=100_000, max_value=3_000_000,
        value=800_000, step=50_000, format="%d"
    )

with col_varsta:
    st.markdown("**📅 Vârstă**")
    age_range = st.slider(
        "Interval de vârstă:",
        min_value=18, max_value=35,
        value=(21, 28)
    )

with col_minute:
    st.markdown("**⚽ Activitate**")
    min_minutes = st.slider(
        "Minimum minute jucate (2 sezoane):",
        min_value=500, max_value=5000,
        value=1500, step=100
    )

# Filtre suplimentare
with st.expander("🔧 Filtre avansate"):
    col_a, col_b = st.columns(2)
    with col_a:
        exclude_top_leagues = st.checkbox(
            "Exclude jucători din top-5 ligi (Premier League, La Liga, Serie A, Bundesliga, Ligue 1) — mai realist pentru bugetul Oțelului",
            value=True
        )
    with col_b:
        prefer_younger = st.checkbox(
            "Penalizează jucătorii peste 28 ani (potențial de revânzare scăzut)",
            value=True
        )

st.divider()

# ===================== Aplicare filtre =====================
filtered = df_candidates.copy()

# Filtru valoare
filtered = filtered[
    (filtered["market_value_in_eur"] > 0) &
    (filtered["market_value_in_eur"] <= budget)
]

# Filtru vârstă
filtered = filtered[
    (filtered["age"] >= age_range[0]) &
    (filtered["age"] <= age_range[1])
]

# Filtru minute
if "total_minutes" in filtered.columns:
    filtered = filtered[filtered["total_minutes"] >= min_minutes]

# Excludere top-5 ligi
if exclude_top_leagues:
    top5 = ["GB1", "ES1", "IT1", "L1", "FR1"]
    filtered = filtered[~filtered["current_club_domestic_competition_id"].isin(top5)]

st.success(
    f"✅ **{len(filtered):,}** jucători îndeplinesc criteriile.".replace(",", ".")
)

st.divider()

# ===================== Score & ranking =====================
st.subheader("🏆 Top recomandări per poziție")

# Construim un scor compozit
df_score = filtered.copy()

# Scor de performanță (normalizat 0-1)
def normalize(s):
    s = s.fillna(0)
    if s.max() == s.min():
        return s * 0
    return (s - s.min()) / (s.max() - s.min())

# Scoruri ponderate
df_score["score_goals"] = normalize(df_score.get("goals_per_90", pd.Series(0, index=df_score.index)))
df_score["score_assists"] = normalize(df_score.get("assists_per_90", pd.Series(0, index=df_score.index)))
df_score["score_minutes"] = normalize(df_score["total_minutes"]) if "total_minutes" in df_score.columns else 0
# Scor "value-for-money" — performanță / preț
df_score["value_for_money"] = (df_score.get("total_goals", 0) + df_score.get("total_assists", 0) * 0.7) / \
                               (df_score["market_value_in_eur"] / 1e6 + 1)
df_score["score_vfm"] = normalize(df_score["value_for_money"])

# Penalizare vârstă > 28
if prefer_younger:
    df_score["age_penalty"] = df_score["age"].apply(lambda x: max(0, (x - 28)) * 0.05)
else:
    df_score["age_penalty"] = 0

# Calculăm scoruri specifice pe poziție
def attacker_score(row):
    return (0.40 * row["score_goals"] +
            0.20 * row["score_assists"] +
            0.20 * row["score_vfm"] +
            0.20 * row["score_minutes"]) - row["age_penalty"]

def midfielder_score(row):
    return (0.25 * row["score_goals"] +
            0.35 * row["score_assists"] +
            0.20 * row["score_vfm"] +
            0.20 * row["score_minutes"]) - row["age_penalty"]

def defender_score(row):
    return (0.10 * row["score_goals"] +
            0.10 * row["score_assists"] +
            0.30 * row["score_vfm"] +
            0.50 * row["score_minutes"]) - row["age_penalty"]

def goalkeeper_score(row):
    return (0.30 * row["score_vfm"] +
            0.70 * row["score_minutes"]) - row["age_penalty"]

position_scoring = {
    "Attack": ("⚔️ Atacanți", attacker_score),
    "Midfield": ("🎯 Mijlocași", midfielder_score),
    "Defender": ("🛡️ Fundași", defender_score),
    "Goalkeeper": ("🥅 Portari", goalkeeper_score),
}

# Tab-uri pentru fiecare poziție
tabs = st.tabs([info[0] for info in position_scoring.values()] + ["🌟 Final Top 3"])

position_keys = list(position_scoring.keys())
all_recommendations = {}

for idx, pos_key in enumerate(position_keys):
    label, scorer = position_scoring[pos_key]
    with tabs[idx]:
        st.markdown(f"### {label}")

        pos_df = df_score[df_score["position"] == pos_key].copy()

        if len(pos_df) == 0:
            st.warning(f"Nu există jucători pe poziția **{pos_key}** care să îndeplinească criteriile.")
            continue

        pos_df["recommendation_score"] = pos_df.apply(scorer, axis=1)
        top_pos = pos_df.nlargest(10, "recommendation_score")

        # Salvăm pentru tab-ul final
        all_recommendations[pos_key] = top_pos.head(3)

        # Afișare detaliată
        for rank, (_, player) in enumerate(top_pos.head(5).iterrows(), 1):
            with st.container():
                cA, cB, cC = st.columns([1, 3, 2])
                with cA:
                    medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "🎯"
                    st.markdown(f"### {medal} #{rank}")
                with cB:
                    st.markdown(f"### {player['name']}")
                    st.markdown(
                        f"**Club:** {player.get('current_club_name', 'N/A')} • "
                        f"**Țară:** {player.get('country_of_citizenship', 'N/A')}  \n"
                        f"**Vârstă:** {player['age']:.1f} ani • "
                        f"**Sub-poziție:** {player.get('sub_position', 'N/A')}  \n"
                        f"**Liga:** {player.get('competition_name', 'N/A')}"
                    )
                with cC:
                    st.metric(
                        "💰 Valoare",
                        f"{player['market_value_in_eur']:,.0f} €".replace(",", ".")
                    )
                    st.metric(
                        "📊 Scor recomandare",
                        f"{player['recommendation_score']:.3f}"
                    )

                # Statistici detaliate
                stats_cols = st.columns(5)
                stats_cols[0].metric("⚽ Goluri", f"{int(player.get('total_goals', 0))}")
                stats_cols[1].metric("🅰️ Asisturi", f"{int(player.get('total_assists', 0))}")
                stats_cols[2].metric("⏱️ Minute", f"{int(player.get('total_minutes', 0)):,}".replace(",", "."))
                stats_cols[3].metric("📈 G/90min", f"{player.get('goals_per_90', 0):.2f}")
                stats_cols[4].metric("💎 Value/€", f"{player.get('value_for_money', 0):.2f}")

                st.divider()

# ===================== Tab final: Top 3 global =====================
with tabs[-1]:
    st.markdown("### 🌟 Top 3 recomandări finale pentru SC Oțelul Galați")
    st.markdown(
        """
        Sintetizând analiza din toate paginile, iată **cei 3 jucători** pe care îi recomand
        clubului SC Oțelul Galați pentru a-și consolida lotul și a urmări calificarea în play-off:
        """
    )

    # Combinăm top 1 din fiecare poziție prioritară
    priority_order = ["Attack", "Midfield", "Defender", "Goalkeeper"]
    final_top = []
    for pos in priority_order:
        if pos in all_recommendations and len(all_recommendations[pos]) > 0:
            top1 = all_recommendations[pos].iloc[0].copy()
            top1["target_position"] = pos
            final_top.append(top1)
        if len(final_top) >= 3:
            break

    if len(final_top) > 0:
        for rank, player in enumerate(final_top, 1):
            st.markdown(f"## {'🥇' if rank == 1 else '🥈' if rank == 2 else '🥉'} Recomandare #{rank}: **{player['name']}**")

            c1, c2 = st.columns([2, 1])
            with c1:
                minutes_str = f"{int(player.get('total_minutes', 0)):,}".replace(",", ".")
                st.markdown(
                    f"""
                    - **Poziție:** {player.get('position', 'N/A')} ({player.get('sub_position', 'N/A')})
                    - **Club actual:** {player.get('current_club_name', 'N/A')} ({player.get('competition_name', 'N/A')})
                    - **Naționalitate:** {player.get('country_of_citizenship', 'N/A')}
                    - **Vârstă:** {player['age']:.1f} ani
                    - **Performanță (2 sezoane):** {int(player.get('total_goals', 0))} goluri, {int(player.get('total_assists', 0))} asisturi, în {minutes_str} minute
                    """
                )

                # Argumentare
                st.info(
                    f"💡 **Motivul recomandării:** Profilul jucătorului combină **performanță bună** "
                    f"(scor {player['recommendation_score']:.3f}/1.0) cu o **valoare accesibilă** "
                    f"({player['market_value_in_eur']:,.0f} €). Vârsta ({player['age']:.1f} ani) îl face un "
                    f"investiție echilibrată între aport imediat și potențial de revânzare.".replace(",", ".")
                )

            with c2:
                st.metric(
                    "💰 Valoare de transfer",
                    f"{player['market_value_in_eur']:,.0f} €".replace(",", ".")
                )
                st.metric(
                    "📊 Scor compozit",
                    f"{player['recommendation_score']:.3f}",
                    delta=f"din 1.000"
                )

            st.divider()

        # Sumarul investiției
        st.markdown("### 💼 Sumar investiție totală")
        total_cost = sum(p["market_value_in_eur"] for p in final_top)
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "💰 Cost total transferuri",
            f"{total_cost:,.0f} €".replace(",", ".")
        )
        c2.metric("👥 Jucători recomandați", len(final_top))
        c3.metric(
            "📊 Scor mediu compozit",
            f"{np.mean([p['recommendation_score'] for p in final_top]):.3f}"
        )

        # Hartă cu locațiile jucătorilor recomandați
        st.markdown("### 🗺️ De unde vin jucătorii recomandați")

        m = folium.Map(location=[50, 15], zoom_start=4, tiles="cartodbpositron")

        # Marker Oțelul
        folium.Marker(
            location=[45.4353, 28.0080],
            popup="<b>SC Oțelul Galați</b> (destinație)",
            tooltip="Oțelul Galați",
            icon=folium.Icon(color="red", icon="star", prefix="fa")
        ).add_to(m)

        # Markeri jucători
        for rank, player in enumerate(final_top, 1):
            country = player.get("country_of_citizenship")
            if country in country_coords:
                lat, lon = country_coords[country]
                # Adăugăm un offset mic pentru a evita suprapunerile
                lat += rank * 0.5
                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(
                        f"<b>#{rank} {player['name']}</b><br>"
                        f"Poziție: {player['position']}<br>"
                        f"Club: {player.get('current_club_name', 'N/A')}<br>"
                        f"Valoare: {player['market_value_in_eur']:,.0f} €",
                        max_width=300
                    ),
                    tooltip=f"#{rank} {player['name']}",
                    icon=folium.Icon(color="green", icon="user", prefix="fa")
                ).add_to(m)

                # Linie de la jucător la Oțelul
                folium.PolyLine(
                    locations=[[lat, lon], [45.4353, 28.0080]],
                    color="green",
                    weight=2,
                    opacity=0.5,
                    dash_array="5"
                ).add_to(m)

        st_folium(m, width=None, height=500, returned_objects=[])

    else:
        st.warning("Nu am putut genera recomandări — încearcă să relaxezi filtrele.")

st.divider()

# ===================== Concluzii =====================
st.subheader("📝 Concluzii și pași următori")
st.markdown(
    """
    ### Ce am descoperit prin această analiză:

    1. **Piețe accesibile pentru Oțelul:** ligile din Europa Centrală și de Est (Polonia,
       Cehia, Slovacia, Croația, Serbia) oferă cei mai mulți jucători cu **raport bun
       performanță/preț**, exact unde Oțelul își poate permite să cumpere.

    2. **Profilul ideal:** jucători de **22-26 ani**, cu **>1500 minute jucate** în ultimele 2
       sezoane și **valoare sub 800k €**. La acest segment ai un mix optim între aport imediat
       și potențial de revânzare.

    3. **Modelul de regresie multiplă** confirmă că variabilele cu cel mai mare impact asupra
       valorii unui jucător sunt: **vârsta**, **golurile/asisturile** și **valoarea totală a
       clubului** unde activează.

    4. **Regresia logistică** identifică jucători "subevaluați" — cu probabilitate mare de a fi
       top-tier dar cu valoare actuală scăzută. Aceștia sunt **ținte ideale** pentru Oțelul.

    ### Pași următori:
    - 🔎 Verificare manuală a fiecărui jucător recomandat (scouting fizic, atitudine, contract).
    - 💬 Negociere directă cu cluburile actuale.
    - 📋 Analiza compatibilității tactice cu sistemul de joc al antrenorului.
    """
)

st.success("🎉 **Proiect finalizat!** Mulțumesc pentru atenție. Disponibil pentru întrebări.")
