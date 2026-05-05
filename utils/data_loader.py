"""
Funcții pentru încărcarea și pregătirea datelor.
Toate funcțiile folosesc @st.cache_data pentru performanță.
"""
import os
import pandas as pd
import numpy as np
import streamlit as st


# Calea către folderul cu date - relativ la rădăcina proiectului
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def get_data_path(filename: str) -> str:
    """Returnează calea completă către un fișier de date."""
    return os.path.join(DATA_DIR, filename)


def file_exists(filename: str) -> bool:
    """Verifică dacă un fișier există în folderul data/."""
    return os.path.exists(get_data_path(filename))


@st.cache_data(show_spinner="Se încarcă jucătorii...")
def load_players() -> pd.DataFrame:
    """Încarcă tabela cu jucători."""
    df = pd.read_csv(get_data_path("players.csv"), low_memory=False)
    # Conversie dată naștere
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
    # Calculăm vârsta (la 1 iulie 2024 - început sezon)
    reference_date = pd.Timestamp("2024-07-01")
    df["age"] = ((reference_date - df["date_of_birth"]).dt.days / 365.25).round(1)
    return df


@st.cache_data(show_spinner="Se încarcă cluburile...")
def load_clubs() -> pd.DataFrame:
    """Încarcă tabela cu cluburi."""
    df = pd.read_csv(get_data_path("clubs.csv"), low_memory=False)
    return df


@st.cache_data(show_spinner="Se încarcă competițiile...")
def load_competitions() -> pd.DataFrame:
    """Încarcă tabela cu competiții."""
    df = pd.read_csv(get_data_path("competitions.csv"), low_memory=False)
    return df


@st.cache_data(show_spinner="Se încarcă transferurile...")
def load_transfers() -> pd.DataFrame:
    """Încarcă tabela cu transferuri."""
    df = pd.read_csv(get_data_path("transfers.csv"), low_memory=False)
    df["transfer_date"] = pd.to_datetime(df["transfer_date"], errors="coerce")
    return df


@st.cache_data(show_spinner="Se încarcă evaluările de piață...")
def load_valuations() -> pd.DataFrame:
    """Încarcă istoricul evaluărilor de piață."""
    df = pd.read_csv(get_data_path("player_valuations.csv"), low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(show_spinner="Se calculează statistici de performanță...")
def load_player_stats() -> pd.DataFrame:
    """
    Agregă apparițiile pe jucător -> total goluri, asisturi, minute, cartonașe.
    Filtrăm doar pe ultimele 2 sezoane (2023-2024 și 2024-2025) pentru relevanță.
    """
    apps = pd.read_csv(
        get_data_path("appearances.csv"),
        low_memory=False,
        usecols=[
            "player_id", "date", "goals", "assists",
            "yellow_cards", "red_cards", "minutes_played",
            "competition_id"
        ],
    )
    apps["date"] = pd.to_datetime(apps["date"], errors="coerce")
    # Filtrare ultimele 2 sezoane
    apps = apps[apps["date"] >= "2023-07-01"].copy()

    stats = (
        apps.groupby("player_id")
        .agg(
            total_goals=("goals", "sum"),
            total_assists=("assists", "sum"),
            total_minutes=("minutes_played", "sum"),
            total_yellow_cards=("yellow_cards", "sum"),
            total_red_cards=("red_cards", "sum"),
            games_played=("date", "count"),
        )
        .reset_index()
    )
    # Goluri per 90 de minute (metric standard în fotbal)
    stats["goals_per_90"] = (stats["total_goals"] / (stats["total_minutes"] / 90)).round(3)
    stats["goals_per_90"] = stats["goals_per_90"].replace([np.inf, -np.inf], 0).fillna(0)
    stats["assists_per_90"] = (stats["total_assists"] / (stats["total_minutes"] / 90)).round(3)
    stats["assists_per_90"] = stats["assists_per_90"].replace([np.inf, -np.inf], 0).fillna(0)
    return stats


@st.cache_data(show_spinner="Se construiește setul de date principal...")
def build_main_dataset() -> pd.DataFrame:
    """
    Construiește setul de date PRINCIPAL pentru analize: players + stats + clubs + competitions.
    Filtrăm doar jucători activi (last_season >= 2023) și cu date suficiente.
    """
    players = load_players()
    stats = load_player_stats()
    clubs = load_clubs()
    competitions = load_competitions()

    # Filtrare jucători activi
    df = players[players["last_season"] >= 2023].copy()

    # Calculăm valoarea totală a clubului pe baza jucătorilor activi
    # (coloana total_market_value din clubs.csv e complet goală)
    club_total_value = (
        df.groupby("current_club_id")["market_value_in_eur"]
        .sum()
        .reset_index()
        .rename(columns={
            "current_club_id": "club_id",
            "market_value_in_eur": "club_total_market_value"
        })
    )

    # Join cu statistici de performanță
    df = df.merge(stats, on="player_id", how="left")

    # Completăm cu 0 pentru jucătorii fără apariții
    fill_zero = ["total_goals", "total_assists", "total_minutes",
                 "total_yellow_cards", "total_red_cards", "games_played",
                 "goals_per_90", "assists_per_90"]
    for col in fill_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Join cu cluburi - folosim coloanele oficiale pentru squad_size și average_age
    clubs_subset = clubs[["club_id", "domestic_competition_id",
                          "squad_size", "average_age", "stadium_seats"]].copy()
    clubs_subset = clubs_subset.rename(columns={
        "squad_size": "club_squad_size",
        "average_age": "club_average_age",
        "stadium_seats": "club_stadium_seats",
    })
    df = df.merge(clubs_subset, left_on="current_club_id", right_on="club_id", how="left")

    # Join cu valoarea calculată
    df = df.merge(club_total_value, on="club_id", how="left")

    # Completăm NaN-urile rămase la nivelul clubului cu mediana globală
    for col in ["club_total_market_value", "club_squad_size", "club_average_age"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Join cu competiții
    comp_subset = competitions[["competition_id", "name", "country_name", "type"]].copy()
    comp_subset = comp_subset.rename(columns={
        "name": "competition_name",
        "country_name": "competition_country"
    })
    df = df.merge(
        comp_subset,
        left_on="current_club_domestic_competition_id",
        right_on="competition_id",
        how="left"
    )

    # Eliminăm coloane redundante / temporare
    drop_cols = ["club_id", "competition_id", "url", "image_url",
                 "player_code", "filename"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


def get_european_country_coords() -> dict:
    """
    Coordonate (lat, lon) pentru capitalele țărilor europene principale + state.
    Folosit pentru hărțile geospațiale din pagina 4.
    """
    return {
        "Germany": (51.1657, 10.4515),
        "France": (46.2276, 2.2137),
        "Spain": (40.4637, -3.7492),
        "Italy": (41.8719, 12.5674),
        "England": (52.3555, -1.1743),
        "Portugal": (39.3999, -8.2245),
        "Netherlands": (52.1326, 5.2913),
        "Belgium": (50.5039, 4.4699),
        "Austria": (47.5162, 14.5501),
        "Switzerland": (46.8182, 8.2275),
        "Greece": (39.0742, 21.8243),
        "Türkiye": (38.9637, 35.2433),
        "Turkey": (38.9637, 35.2433),
        "Russia": (61.5240, 105.3188),
        "Ukraine": (48.3794, 31.1656),
        "Poland": (51.9194, 19.1451),
        "Czech Republic": (49.8175, 15.4730),
        "Croatia": (45.1000, 15.2000),
        "Serbia": (44.0165, 21.0059),
        "Romania": (45.9432, 24.9668),
        "Denmark": (56.2639, 9.5018),
        "Sweden": (60.1282, 18.6435),
        "Norway": (60.4720, 8.4689),
        "Scotland": (56.4907, -4.2026),
        "Bulgaria": (42.7339, 25.4858),
        "Hungary": (47.1625, 19.5033),
        "Slovakia": (48.6690, 19.6990),
        "Slovenia": (46.1512, 14.9955),
    }
