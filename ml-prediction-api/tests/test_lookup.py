import pandas as pd
from api.lookup import find_pokemon


def _sample_df():
    return pd.DataFrame([
        {
            "name": "bulbasaur", "height": 7, "weight": 69, "base_experience": 64,
            "type1": "grass", "type2": "poison",
            "hp": 45, "attack": 49, "defense": 49, "sp_attack": 65, "sp_defense": 65, "speed": 45,
        },
        {
            "name": "rattata", "height": 3, "weight": 35, "base_experience": 51,
            "type1": "normal", "type2": None,
            "hp": 30, "attack": 56, "defense": 35, "sp_attack": 25, "sp_defense": 35, "speed": 72,
        },
    ])


def test_find_pokemon_case_insensitive_match():
    result = find_pokemon(_sample_df(), "BulbaSaur")

    assert result["name"] == "bulbasaur"
    assert result["type1"] == "grass"
    assert result["type2"] == "poison"
    assert result["actual_total_stats"] == 45 + 49 + 49 + 65 + 65 + 45


def test_find_pokemon_missing_type2_is_none():
    result = find_pokemon(_sample_df(), "rattata")
    assert result["type2"] is None


def test_find_pokemon_returns_none_when_not_found():
    assert find_pokemon(_sample_df(), "missingno") is None
