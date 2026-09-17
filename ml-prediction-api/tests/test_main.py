from fastapi.testclient import TestClient
from api.main import app, _state
from api.spotify_client import SpotifyAuthError


class FakeSpotify:
    def __init__(self, track):
        self._track = track
        self.queries = []

    def search_track(self, query):
        self.queries.append(query)
        return self._track


class FailingSpotify:
    def search_track(self, query):
        raise SpotifyAuthError("upstream failure")


def test_predict_returns_prediction():
    with TestClient(app) as client:
        resp = client.post("/predict", json={
            "height": 7, "weight": 69, "base_experience": 64,
            "type1": "grass", "type2": "poison",
        })
    assert resp.status_code == 200
    assert isinstance(resp.json()["predicted_total_stats"], float)


def test_predict_unknown_type_returns_400():
    with TestClient(app) as client:
        resp = client.post("/predict", json={
            "height": 7, "weight": 69, "base_experience": 64,
            "type1": "plastic", "type2": None,
        })
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "UNKNOWN_TYPE"


def test_pokemon_lookup_found():
    with TestClient(app) as client:
        resp = client.get("/pokemon/bulbasaur")
    assert resp.status_code == 200
    assert resp.json()["type1"] == "grass"


def test_pokemon_lookup_not_found():
    with TestClient(app) as client:
        resp = client.get("/pokemon/missingno")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "NOT_FOUND"


def test_music_without_spotify_configured_returns_503():
    with TestClient(app) as client:
        _state["spotify"] = None
        resp = client.get("/pokemon/bulbasaur/music")
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "SPOTIFY_NOT_CONFIGURED"


def test_music_returns_track_from_spotify():
    with TestClient(app) as client:
        fake = FakeSpotify({
            "name": "Song A", "artist": "Artist A",
            "spotify_url": "https://open.spotify.com/track/abc", "preview_url": None,
        })
        _state["spotify"] = fake
        resp = client.get("/pokemon/bulbasaur/music")
    assert resp.status_code == 200
    body = resp.json()
    assert body["genre"] == "folk"  # bulbasaur is grass -> folk
    assert body["track"]["name"] == "Song A"
    # bulbasaur is grass/poison -> one combined query, found on first try
    assert fake.queries == ['genre:"folk" punk']


def test_music_pokemon_not_found_returns_404():
    with TestClient(app) as client:
        _state["spotify"] = FakeSpotify(None)
        resp = client.get("/pokemon/missingno/music")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "NOT_FOUND"


def test_music_no_track_found_returns_404():
    with TestClient(app) as client:
        fake = FakeSpotify(None)
        _state["spotify"] = fake
        resp = client.get("/pokemon/bulbasaur/music")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "NO_TRACK_FOUND"
    # dual-type: combined query, then fallback to genre1 alone
    assert fake.queries == ['genre:"folk" punk', 'genre:"folk"']


def test_music_single_type_pokemon_does_not_retry_identical_query():
    with TestClient(app) as client:
        fake = FakeSpotify(None)
        _state["spotify"] = fake
        resp = client.get("/pokemon/pikachu/music")
    assert resp.status_code == 404
    # pikachu is electric-only (no type2) -> fallback would be an identical query, so skip it
    assert fake.queries == ['genre:"edm"']


def test_music_spotify_error_returns_502():
    with TestClient(app) as client:
        _state["spotify"] = FailingSpotify()
        resp = client.get("/pokemon/bulbasaur/music")
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "SPOTIFY_ERROR"
