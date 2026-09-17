import pandas as pd


def build_type_encoders(csv_path: str) -> tuple[dict[str, int], dict[str, int]]:
    df = pd.read_csv(csv_path)
    type1_categories = df["type1"].astype("category").cat.categories
    type2_categories = df["type2"].fillna("None").astype("category").cat.categories
    type1_map = {v: i for i, v in enumerate(type1_categories)}
    type2_map = {v: i for i, v in enumerate(type2_categories)}
    return type1_map, type2_map


def encode_types(
    type1: str,
    type2: str | None,
    type1_map: dict[str, int],
    type2_map: dict[str, int],
) -> tuple[int, int]:
    t1 = type1.strip().lower()
    if t1 not in type1_map:
        raise ValueError(f"unknown type1 {type1!r}; valid: {sorted(type1_map)}")

    t2 = "None" if type2 is None or not type2.strip() else type2.strip().lower()
    if t2 not in type2_map:
        raise ValueError(f"unknown type2 {type2!r}; valid: {sorted(type2_map)}")

    return type1_map[t1], type2_map[t2]
