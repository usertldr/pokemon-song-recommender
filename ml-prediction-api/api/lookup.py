import pandas as pd

STAT_COLUMNS = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]


def find_pokemon(df: pd.DataFrame, name: str) -> dict | None:
    match = df[df["name"].str.lower() == name.strip().lower()]
    if match.empty:
        return None
    row = match.iloc[0]
    type2 = row["type2"] if pd.notna(row["type2"]) else None
    return {
        "name": row["name"],
        "height": float(row["height"]),
        "weight": float(row["weight"]),
        "base_experience": float(row["base_experience"]),
        "type1": row["type1"],
        "type2": type2,
        "actual_total_stats": float(sum(row[c] for c in STAT_COLUMNS)),
    }
