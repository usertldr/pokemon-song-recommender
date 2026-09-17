# Pokemon Stats + Music Recommendation API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish `ml-prediction-api` as a working local FastAPI service with three
endpoints — Pokemon total-stats prediction, a real-Pokemon data lookup, and a
Spotify-backed per-Pokemon music genre/track recommendation — wrapped in a Dockerfile.

**Architecture:** A small `api/` package with one pure-logic module per concern
(type encoding, CSV lookup, genre mapping, Spotify HTTP client), wired together in
`api/main.py`'s FastAPI routes. Each module is unit-testable without a model, CSV, or
network call except where that's literally its job (the Spotify client), so tests stay
fast and the endpoint tests only need to fake the Spotify layer.

**Tech Stack:** Python 3.11, FastAPI + uvicorn, lightgbm/scikit-learn (existing
pickled model), pandas, requests (Spotify HTTP calls), pytest + FastAPI's `TestClient`.

## Global Constraints

- Repo root for all new/modified files in this plan: `C:\Users\naira\ml-stage1\ml-prediction-api\`.
- The trained model currently lives at `C:\Users\naira\ml-stage1\models\pokemon_lgbm.pkl`
  (one level up, outside this sub-project). Task 1 copies it into
  `ml-prediction-api\models\pokemon_lgbm.pkl` so the project and its Docker build are
  self-contained. The original file is left in place, not moved.
- Type encoding for `type1`/`type2` must exactly replicate the real original
  preprocessing found in `ml-prediction-api/src/preprocess.py`:
  `pandas.astype("category").cat.codes` after `type2.fillna("None")` (capital-N
  sentinel — sorts before all lowercase type names under plain string sort, which
  changes the resulting codes versus a naive lowercase sentinel). Verified against
  the live pickle: MAE 6.85 on the full CSV, vs. 8.42 for an earlier lowercase-
  sentinel guess — use the exact logic, not a reconstruction.
- Model feature order is fixed and must not change:
  `[height, weight, base_experience, type1_encoded, type2_encoded]` (confirmed via
  `model.feature_name_` on the existing pickle).
- Spotify integration uses only the **Client Credentials** (app-only) flow — no user
  OAuth login/consent screen, no user-specific data.
- Spotify's `/v1/recommendations` endpoint must **not** be used (deprecated for apps
  created after Nov 2024, may not work for a freshly-registered app). Use `/v1/search`
  with a `genre:` filter instead.
- `GET /pokemon/{name}/music` must degrade gracefully (HTTP 503) when
  `SPOTIFY_CLIENT_ID`/`SPOTIFY_CLIENT_SECRET` aren't set — `/predict` and
  `/pokemon/{name}` must keep working regardless.
- All error responses use the shape `{"status": "error", "error": {"code": str,
  "message": str}}`, matching the existing `satquery_vqa/service.py` convention in
  this repo.

---

### Task 1: Project scaffolding + type-encoding reconstruction

**Files:**
- Create: `ml-prediction-api/requirements.txt`
- Create: `ml-prediction-api/api/__init__.py` (empty)
- Create: `ml-prediction-api/api/encoding.py`
- Create: `ml-prediction-api/models/pokemon_lgbm.pkl` (copied, not authored)
- Test: `ml-prediction-api/tests/test_encoding.py`

**Interfaces:**
- Produces: `build_type_encoders(csv_path: str) -> tuple[dict[str, int], dict[str,
  int]]` — returns `(type1_map, type2_map)`, replicating the exact encoding
  `ml-prediction-api/src/preprocess.py` used at training time
  (`pandas.astype("category").cat.codes`, `type2.fillna("None")` before encoding —
  the capital-N sentinel sorts before all lowercase type names). Produces:
  `encode_types(type1: str, type2: str | None, type1_map: dict[str, int],
  type2_map: dict[str, int]) -> tuple[int, int]` — case-insensitive, raises
  `ValueError` on an unrecognized type name. Used by Task 5 (`POST /predict`).

- [ ] **Step 1: Create the requirements file**

```
# ml-prediction-api/requirements.txt
fastapi
uvicorn
lightgbm
scikit-learn
pandas
requests
pytest
httpx
```

- [ ] **Step 2: Copy the model into the project**

```bash
mkdir -p ml-prediction-api/models
cp models/pokemon_lgbm.pkl ml-prediction-api/models/pokemon_lgbm.pkl
```

- [ ] **Step 3: Create the `api` package marker**

```python
# ml-prediction-api/api/__init__.py
```
(empty file)

- [ ] **Step 4: Write the failing test**

```python
# ml-prediction-api/tests/test_encoding.py
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

    assert type1_map == {"fire": 0, "grass": 1, "water": 2}
    # missing type2 -> "None" (capital), which sorts BEFORE lowercase names
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
```

- [ ] **Step 5: Run it to verify it fails**

Run (from `ml-prediction-api/`): `python -m pytest tests/test_encoding.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api'`.

- [ ] **Step 6: Implement `api/encoding.py`**

```python
# ml-prediction-api/api/encoding.py
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
```

(This replicates `ml-prediction-api/src/preprocess.py`'s actual training-time encoding —
verified against the live pickle at MAE 6.85 on the full CSV, vs. 8.42 for an earlier
lowercase-sentinel guess.)

- [ ] **Step 7: Run the test and verify it passes**

Run: `python -m pytest tests/test_encoding.py -v`
Expected: 3 passed.

- [ ] **Step 8: Commit**

```bash
git add ml-prediction-api/requirements.txt ml-prediction-api/api/__init__.py ml-prediction-api/api/encoding.py ml-prediction-api/tests/test_encoding.py ml-prediction-api/models/pokemon_lgbm.pkl
git commit -m "Add project scaffolding and type-encoding reconstruction"
```

---

### Task 2: Pokemon CSV lookup

**Files:**
- Create: `ml-prediction-api/api/lookup.py`
- Test: `ml-prediction-api/tests/test_lookup.py`

**Interfaces:**
- Consumes: a pandas DataFrame loaded from `data/pokemon_gen1-gen9.csv` (I/O stays in
  `main.py`, this module is pure).
- Produces: `find_pokemon(df: pd.DataFrame, name: str) -> dict | None` — case-
  insensitive match on `name`; returns `{"name", "height", "weight",
  "base_experience", "type1", "type2", "actual_total_stats"}` (`type2` is `None` for
  single-type Pokemon; `actual_total_stats` is the sum of
  `hp+attack+defense+sp_attack+sp_defense+speed`), or `None` if not found. Used by
  Task 5's `GET /pokemon/{name}` and `GET /pokemon/{name}/music`.

- [ ] **Step 1: Write the failing test**

```python
# ml-prediction-api/tests/test_lookup.py
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_lookup.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.lookup'`.

- [ ] **Step 3: Implement `api/lookup.py`**

```python
# ml-prediction-api/api/lookup.py
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
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `python -m pytest tests/test_lookup.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add ml-prediction-api/api/lookup.py ml-prediction-api/tests/test_lookup.py
git commit -m "Add case-insensitive Pokemon CSV lookup"
```

---

### Task 3: Type-to-genre mapping and Spotify query building

**Files:**
- Create: `ml-prediction-api/api/genre.py`
- Test: `ml-prediction-api/tests/test_genre.py`

**Interfaces:**
- Produces: `TYPE_GENRE_MAP: dict[str, str]` (18 entries, canonical Pokemon types
  lowercase → genre string). Produces: `genre_for_type(type_name: str) -> str` —
  case-insensitive, raises `ValueError` on an unknown type. Produces:
  `build_search_query(type1: str, type2: str | None) -> str` — e.g. `genre:"rock"
  indie`. Produces: `fallback_query(type1: str) -> str` — e.g. `genre:"rock"` alone.
  Used by Task 5's `GET /pokemon/{name}/music`.

- [ ] **Step 1: Write the failing test**

```python
# ml-prediction-api/tests/test_genre.py
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_genre.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.genre'`.

- [ ] **Step 3: Implement `api/genre.py`**

```python
# ml-prediction-api/api/genre.py
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
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `python -m pytest tests/test_genre.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add ml-prediction-api/api/genre.py ml-prediction-api/tests/test_genre.py
git commit -m "Add type-to-genre mapping and Spotify search query builder"
```

---

### Task 4: Spotify Client Credentials HTTP client

**Files:**
- Create: `ml-prediction-api/api/spotify_client.py`
- Test: `ml-prediction-api/tests/test_spotify_client.py`

**Interfaces:**
- Consumes: `requests` (module-level `requests.post`/`requests.get`, monkeypatched
  in tests — no real network calls in this test file).
- Produces: `class SpotifyClient(client_id: str, client_secret: str)` with
  `.get_access_token() -> str` (fetches via Client Credentials flow, caches until
  expiry) and `.search_track(query: str) -> dict | None` (returns `{"name",
  "artist", "spotify_url", "preview_url"}` for the first result, or `None` if no
  tracks matched). Produces: `class SpotifyAuthError(Exception)`, raised on any
  non-200 response from either Spotify call. Used by Task 5's
  `GET /pokemon/{name}/music`.

- [ ] **Step 1: Write the failing test**

```python
# ml-prediction-api/tests/test_spotify_client.py
import time
import pytest
from api import spotify_client as sc
from api.spotify_client import SpotifyClient, SpotifyAuthError


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def test_get_access_token_caches_until_expiry(monkeypatch):
    calls = []

    def fake_post(url, data=None, auth=None, timeout=None):
        calls.append(url)
        return FakeResponse(200, {"access_token": "tok123", "expires_in": 3600})

    monkeypatch.setattr(sc.requests, "post", fake_post)

    client = SpotifyClient("id", "secret")
    assert client.get_access_token() == "tok123"
    assert client.get_access_token() == "tok123"
    assert len(calls) == 1  # second call served from cache


def test_get_access_token_raises_on_non_200(monkeypatch):
    monkeypatch.setattr(
        sc.requests, "post",
        lambda url, data=None, auth=None, timeout=None: FakeResponse(401, {"error": "bad creds"}),
    )

    client = SpotifyClient("id", "secret")
    with pytest.raises(SpotifyAuthError):
        client.get_access_token()


def test_search_track_returns_first_track(monkeypatch):
    monkeypatch.setattr(
        sc.requests, "post",
        lambda url, data=None, auth=None, timeout=None: FakeResponse(200, {"access_token": "tok", "expires_in": 3600}),
    )
    monkeypatch.setattr(
        sc.requests, "get",
        lambda url, params=None, headers=None, timeout=None: FakeResponse(200, {
            "tracks": {"items": [{
                "name": "Song A",
                "artists": [{"name": "Artist A"}, {"name": "Artist B"}],
                "external_urls": {"spotify": "https://open.spotify.com/track/abc"},
                "preview_url": "https://p.scdn.co/mp3/abc",
            }]},
        }),
    )

    client = SpotifyClient("id", "secret")
    track = client.search_track('genre:"rock"')

    assert track == {
        "name": "Song A",
        "artist": "Artist A, Artist B",
        "spotify_url": "https://open.spotify.com/track/abc",
        "preview_url": "https://p.scdn.co/mp3/abc",
    }


def test_search_track_returns_none_when_no_items(monkeypatch):
    monkeypatch.setattr(
        sc.requests, "post",
        lambda url, data=None, auth=None, timeout=None: FakeResponse(200, {"access_token": "tok", "expires_in": 3600}),
    )
    monkeypatch.setattr(
        sc.requests, "get",
        lambda url, params=None, headers=None, timeout=None: FakeResponse(200, {"tracks": {"items": []}}),
    )

    client = SpotifyClient("id", "secret")
    assert client.search_track('genre:"rock"') is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_spotify_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.spotify_client'`.

- [ ] **Step 3: Implement `api/spotify_client.py`**

```python
# ml-prediction-api/api/spotify_client.py
import time
import requests

TOKEN_URL = "https://accounts.spotify.com/api/token"
SEARCH_URL = "https://api.spotify.com/v1/search"


class SpotifyAuthError(Exception):
    pass


class SpotifyClient:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self._token: str | None = None
        self._token_expires_at = 0.0

    def get_access_token(self) -> str:
        if self._token and time.time() < self._token_expires_at:
            return self._token
        resp = requests.post(
            TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(self.client_id, self.client_secret),
            timeout=10,
        )
        if resp.status_code != 200:
            raise SpotifyAuthError(f"token request failed: {resp.status_code} {resp.text}")
        data = resp.json()
        self._token = data["access_token"]
        self._token_expires_at = time.time() + data["expires_in"] - 30
        return self._token

    def search_track(self, query: str) -> dict | None:
        token = self.get_access_token()
        resp = requests.get(
            SEARCH_URL,
            params={"q": query, "type": "track", "limit": 1},
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if resp.status_code != 200:
            raise SpotifyAuthError(f"search failed: {resp.status_code} {resp.text}")
        items = resp.json().get("tracks", {}).get("items", [])
        if not items:
            return None
        track = items[0]
        return {
            "name": track["name"],
            "artist": ", ".join(a["name"] for a in track["artists"]),
            "spotify_url": track["external_urls"]["spotify"],
            "preview_url": track.get("preview_url"),
        }
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `python -m pytest tests/test_spotify_client.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add ml-prediction-api/api/spotify_client.py ml-prediction-api/tests/test_spotify_client.py
git commit -m "Add Spotify Client Credentials HTTP client"
```

---

### Task 5: FastAPI app — wire up all three endpoints

**Files:**
- Create: `ml-prediction-api/api/main.py`
- Test: `ml-prediction-api/tests/test_main.py`

**Interfaces:**
- Consumes: `build_type_encoders`, `encode_types` (Task 1); `find_pokemon` (Task 2);
  `genre_for_type`, `build_search_query`, `fallback_query` (Task 3); `SpotifyClient`,
  `SpotifyAuthError` (Task 4); the pickled model at `models/pokemon_lgbm.pkl`; the
  CSV at `data/pokemon_gen1-gen9.csv`.
- Produces: FastAPI `app` object with `POST /predict`, `GET /pokemon/{name}`,
  `GET /pokemon/{name}/music`. Module-level `_state: dict` holding `"model"`,
  `"df"`, `"type1_map"`, `"type2_map"`, `"spotify"` (populated by the `lifespan`
  context manager) — tests reach into `_state["spotify"]` directly to inject a fake
  Spotify client after the app's `TestClient` context has entered, avoiding a second
  layer of HTTP mocking on top of Task 4's.

- [ ] **Step 1: Write the failing test**

```python
# ml-prediction-api/tests/test_main.py
from fastapi.testclient import TestClient
from api.main import app, _state


class FakeSpotify:
    def __init__(self, track):
        self._track = track
        self.queries = []

    def search_track(self, query):
        self.queries.append(query)
        return self._track


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
        _state["spotify"] = FakeSpotify({
            "name": "Song A", "artist": "Artist A",
            "spotify_url": "https://open.spotify.com/track/abc", "preview_url": None,
        })
        resp = client.get("/pokemon/bulbasaur/music")
    assert resp.status_code == 200
    body = resp.json()
    assert body["genre"] == "folk"  # bulbasaur is grass -> folk
    assert body["track"]["name"] == "Song A"


def test_music_pokemon_not_found_returns_404():
    with TestClient(app) as client:
        _state["spotify"] = FakeSpotify(None)
        resp = client.get("/pokemon/missingno/music")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "NOT_FOUND"


def test_music_no_track_found_returns_404():
    with TestClient(app) as client:
        _state["spotify"] = FakeSpotify(None)
        resp = client.get("/pokemon/bulbasaur/music")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "NO_TRACK_FOUND"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.main'`.

- [ ] **Step 3: Implement `api/main.py`**

```python
# ml-prediction-api/api/main.py
import os
from pathlib import Path
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api.encoding import build_type_encoders, encode_types
from api.lookup import find_pokemon
from api.genre import build_search_query, fallback_query, genre_for_type
from api.spotify_client import SpotifyClient, SpotifyAuthError

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "pokemon_lgbm.pkl"
CSV_PATH = BASE_DIR / "data" / "pokemon_gen1-gen9.csv"

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["model"] = joblib.load(MODEL_PATH)
    _state["df"] = pd.read_csv(CSV_PATH)
    _state["type1_map"], _state["type2_map"] = build_type_encoders(str(CSV_PATH))
    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    _state["spotify"] = SpotifyClient(client_id, client_secret) if client_id and client_secret else None
    yield


app = FastAPI(title="Pokemon stats + music API", lifespan=lifespan)


class PredictRequest(BaseModel):
    height: float
    weight: float
    base_experience: float
    type1: str
    type2: str | None = None


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"status": "error", "error": {"code": code, "message": message}},
    )


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        t1, t2 = encode_types(request.type1, request.type2, _state["type1_map"], _state["type2_map"])
    except ValueError as exc:
        return _error(400, "UNKNOWN_TYPE", str(exc))
    features = pd.DataFrame(
        [[request.height, request.weight, request.base_experience, t1, t2]],
        columns=["height", "weight", "base_experience", "type1_encoded", "type2_encoded"],
    )
    prediction = _state["model"].predict(features)[0]
    return {"predicted_total_stats": float(prediction)}


@app.get("/pokemon/{name}")
def pokemon(name: str):
    result = find_pokemon(_state["df"], name)
    if result is None:
        return _error(404, "NOT_FOUND", f"no pokemon named {name!r}")
    return result


@app.get("/pokemon/{name}/music")
def pokemon_music(name: str):
    result = find_pokemon(_state["df"], name)
    if result is None:
        return _error(404, "NOT_FOUND", f"no pokemon named {name!r}")

    spotify = _state.get("spotify")
    if spotify is None:
        return _error(503, "SPOTIFY_NOT_CONFIGURED", "SPOTIFY_CLIENT_ID/SPOTIFY_CLIENT_SECRET not set")

    type1, type2 = result["type1"], result["type2"]
    genre = genre_for_type(type1)
    try:
        track = spotify.search_track(build_search_query(type1, type2))
        if track is None:
            track = spotify.search_track(fallback_query(type1))
    except SpotifyAuthError as exc:
        return _error(502, "SPOTIFY_ERROR", str(exc))

    if track is None:
        return _error(404, "NO_TRACK_FOUND", f"no track found for genre {genre!r}")

    return {"pokemon": result["name"], "genre": genre, "track": track}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8012)
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `python -m pytest tests/test_main.py -v`
Expected: 8 passed.

- [ ] **Step 5: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: all tests across Tasks 1-5 pass (24 total).

- [ ] **Step 6: Commit**

```bash
git add ml-prediction-api/api/main.py ml-prediction-api/tests/test_main.py
git commit -m "Wire up FastAPI app: predict, pokemon lookup, and music endpoints"
```

---

### Task 6: Dockerfile, README, and manual smoke test

**Files:**
- Modify: `ml-prediction-api/Dockerfile`
- Modify: `ml-prediction-api/README.md`
- Test: manual run (no pytest — this verifies the app actually serves over HTTP)

**Interfaces:**
- Consumes: `api/main.py`'s `app` (Task 5), `requirements.txt` (Task 1).
- Produces: a Docker image that runs the service on port 8012.

- [ ] **Step 1: Write the Dockerfile**

```dockerfile
# ml-prediction-api/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY data/ data/
COPY models/ models/

EXPOSE 8012
CMD ["python", "-m", "api.main"]
```

- [ ] **Step 2: Write the README**

```markdown
# ml-prediction-api

Predicts a Pokemon's total base stats from `height, weight, base_experience, type1,
type2` (LightGBM, tracked in `mlruns/`), looks up real Pokemon data, and recommends a
Spotify track by mapping the Pokemon's type(s) to a music genre.

## Type encoding

`api/encoding.py` replicates the exact type-to-integer encoding
`src/preprocess.py` used at training time (`pandas.astype("category").cat.codes`,
with `type2` missing values filled as `"None"` before encoding — that capital-N
sentinel sorts before all lowercase type names). Verified against the live model:
MAE 6.85 against actual total stats across the full CSV.

## Endpoints

- `POST /predict` — body: `{"height": float, "weight": float, "base_experience":
  float, "type1": str, "type2": str | null}` → `{"predicted_total_stats": float}`.
- `GET /pokemon/{name}` — a real Pokemon's known data + actual total stats.
- `GET /pokemon/{name}/music` — a genre + real Spotify track recommendation, derived
  from the Pokemon's type(s). Returns `503` if Spotify isn't configured (see below).

## Running locally

```bash
pip install -r requirements.txt
python -m api.main
```

Serves on `http://127.0.0.1:8012`.

## Spotify setup (optional — only needed for `/pokemon/{name}/music`)

1. Create a free app at https://developer.spotify.com/dashboard.
2. Set environment variables before starting the service:

```bash
export SPOTIFY_CLIENT_ID=your_client_id
export SPOTIFY_CLIENT_SECRET=your_client_secret
```

Without these set, `/predict` and `/pokemon/{name}` still work; `/pokemon/{name}/music`
returns a `503`.

## Docker

```bash
docker build -t pokemon-stats-music-api .
docker run --rm -p 8012:8012 \
  -e SPOTIFY_CLIENT_ID=your_client_id \
  -e SPOTIFY_CLIENT_SECRET=your_client_secret \
  pokemon-stats-music-api
```
```

- [ ] **Step 3: Run the service locally and smoke-test the endpoints that don't need Spotify**

```bash
cd ml-prediction-api
python -m api.main
```
In another terminal:
```bash
curl -X POST http://127.0.0.1:8012/predict -H "Content-Type: application/json" -d "{\"height\":7,\"weight\":69,\"base_experience\":64,\"type1\":\"grass\",\"type2\":\"poison\"}"
curl http://127.0.0.1:8012/pokemon/bulbasaur
```
Expected: both return `200` with the shapes documented in the README. Stop the server
(Ctrl+C) after confirming.

- [ ] **Step 4: (Only if you have Spotify credentials) smoke-test the music endpoint**

```bash
export SPOTIFY_CLIENT_ID=...
export SPOTIFY_CLIENT_SECRET=...
python -m api.main
```
```bash
curl http://127.0.0.1:8012/pokemon/bulbasaur/music
```
Expected: `200` with a real track. This step is flagged as conditional, not blocking —
Spotify credentials are yours to provision, not something to fake or skip silently if
missing; if it returns `503`, that's the documented behavior, not a bug.

- [ ] **Step 5: Commit**

```bash
git add ml-prediction-api/Dockerfile ml-prediction-api/README.md
git commit -m "Add Dockerfile and document the API, its endpoints, and Spotify setup"
```

---

## Self-review notes

- **Spec coverage**: Component 1 (predict, Task 1 + Task 5), Component 2 (lookup,
  Task 2 + Task 5), Component 3 (music, Tasks 3-5), error-shape consistency (Global
  Constraints + every task), Dockerfile/README/env-var docs (Task 6) — all covered.
- **Known unknowns flagged, not hidden**: type-encoding reconstruction accuracy
  (Task 1, restated in Task 6's README), Spotify `genre:` search quality for unusual
  genre strings (inherent to Task 3/5, only observable once queried live) — both
  called out rather than assumed to work perfectly.
- **Type consistency**: `find_pokemon`'s return dict shape (Task 2) is used
  identically in Task 5's `/pokemon/{name}` and `/pokemon/{name}/music` handlers;
  `SpotifyClient.search_track`'s return shape (Task 4) matches exactly what Task 5's
  `FakeSpotify` test double and the real `/music` response embed under `"track"`;
  `build_search_query`/`fallback_query`/`genre_for_type` signatures (Task 3) match
  their call sites in Task 5.
