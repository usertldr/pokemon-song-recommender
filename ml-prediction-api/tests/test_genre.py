from api.genre import TYPE_GENRE_MAP, genre_for_type, build_search_query, fallback_query


def test_type_genre_map_has_all_eighteen_types():
    assert len(TYPE_GENRE_MAP) == 18
    assert TYPE_GENRE_MAP["fire"] == "rock"
    assert TYPE_GENRE_MAP["water"] == "chill"


def test_genre_for_type_case_insensitive():
    assert genre_for_type("Fire") == "rock"


def test_genre_for_type_raises_on_unknown():
    try:
        genre_for_type("plastic")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_build_search_query_single_type():
    assert build_search_query("fire", None) == 'genre:"rock"'


def test_build_search_query_dual_type_appends_second_genre():
    assert build_search_query("fire", "flying") == 'genre:"rock" indie'


def test_fallback_query_uses_only_first_type():
    assert fallback_query("fire") == 'genre:"rock"'
