import pandas as pd
from api.encoding import build_type_encoders, encode_types


def test_build_type_encoders_matches_pandas_category_codes(tmp_path):
    csv_path = tmp_path / "p.csv"
    pd.DataFrame([
        {"type1": "fire", "type2": "flying"},
        {"type1": "water", "type2": None},
        {"type1": "grass", "type2": "poison"},
    ]).to_csv(csv_path, index=False)

    type1_map, type2_map = build_type_encoders(str(csv_path))

    # type1: no missing values, plain alphabetical
    assert type1_map == {"fire": 0, "grass": 1, "water": 2}
    # type2: missing -> "None" (capital), which sorts BEFORE lowercase names
    assert type2_map == {"None": 0, "flying": 1, "poison": 2}


def test_encode_types_handles_missing_type2_and_case():
    type1_map = {"fire": 0, "water": 1}
    type2_map = {"None": 0, "flying": 1}

    assert encode_types("Fire", None, type1_map, type2_map) == (0, 0)
    assert encode_types("water", "Flying", type1_map, type2_map) == (1, 1)


def test_encode_types_raises_on_unknown_type():
    type1_map = {"fire": 0}
    type2_map = {"None": 0}
    try:
        encode_types("unknown", None, type1_map, type2_map)
        assert False, "expected ValueError"
    except ValueError:
        pass
