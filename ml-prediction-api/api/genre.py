TYPE_GENRE_MAP = {
    "normal": "pop", "fire": "rock", "water": "chill", "electric": "edm",
    "grass": "folk", "ice": "ambient", "fighting": "hip-hop", "poison": "punk",
    "ground": "country", "flying": "indie", "psychic": "trance", "bug": "techno",
    "rock": "metal", "ghost": "darkwave", "dragon": "orchestral", "dark": "industrial",
    "steel": "synthwave", "fairy": "dream-pop",
}


def genre_for_type(type_name: str) -> str:
    key = type_name.strip().lower()
    if key not in TYPE_GENRE_MAP:
        raise ValueError(f"unknown type {type_name!r}; valid: {sorted(TYPE_GENRE_MAP)}")
    return TYPE_GENRE_MAP[key]


def build_search_query(type1: str, type2: str | None) -> str:
    query = f'genre:"{genre_for_type(type1)}"'
    if type2:
        query = f"{query} {genre_for_type(type2)}"
    return query


def fallback_query(type1: str) -> str:
    return f'genre:"{genre_for_type(type1)}"'
