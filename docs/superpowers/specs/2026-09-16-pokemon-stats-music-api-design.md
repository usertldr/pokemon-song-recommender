# Pokemon Stats + Music Recommendation API (Design Spec)

## Context

`ml-prediction-api/` already contains a trained LightGBM model
(`models/pokemon_lgbm.pkl`) predicting a Pokemon's total base stats
(HP+Attack+Defense+SpAttack+SpDefense+Speed) from `height, weight,
base_experience, type1, type2`, tracked via MLflow (`mlruns/`), plus the source
data (`data/pokemon_gen1-gen9.csv`) and EDA/feature-importance images. The
serving code was never built: `api/main.py` and `Dockerfile` are both empty, and
the original training notebook that produced the model is gone (only stray
Jupyter checkpoint files for unrelated artifacts remain — no notebook).

This spec covers finishing that project as a small FastAPI service, and adding a
new, unrelated-to-the-model feature on top: a per-Pokemon music genre/song
recommendation, driven by its type(s) via Spotify's public catalog.

## Goal

A locally runnable FastAPI service with three endpoints:
1. Predict a hypothetical Pokemon's total base stats from raw features.
2. Look up a real Pokemon's known data (and actual total stats) from the CSV.
3. Recommend a genre + a real Spotify track for a given (real) Pokemon, derived
   from its type(s).

Non-goals: retraining or improving the stats model, a frontend/UI, deployment
beyond a local Docker image, user-specific Spotify data (playlists, playback) —
only the app-only Client Credentials flow against Spotify's public catalog.

## Component 1: Stats prediction (`POST /predict`)

**Type encoding problem (update: not actually lost):** the pickled model expects
`type1_encoded`/`type2_encoded` integer columns. This spec originally assumed the
training notebook was lost and planned a guessed reconstruction — during
implementation, the real preprocessing code turned out to still exist at
`ml-prediction-api/src/preprocess.py` (pre-existing on disk, just not noticed during
initial exploration). It encodes via `pandas.astype("category").cat.codes` after
`type2.fillna("None")` (capital-N sentinel, which sorts before all lowercase type
names under plain string sort — this changes the resulting integer codes compared to
a naive lowercase-sentinel guess). Verified against the live pickle: the exact
original encoding gives MAE 6.85 against actual total stats across the full CSV,
versus 8.42 for the initial guess — confirming this is the real encoding, not another
approximation.

**Request:** `height: float, weight: float, base_experience: float, type1:
str, type2: str | None`. `type1`/`type2` are case-insensitive; matched against
the known encoder vocabulary (18 canonical Pokemon types + `"None"` for a
missing `type2`).

**Response:** `{"predicted_total_stats": float}`.

**Errors:** unknown type string → `400` naming the valid type list.

## Component 2: Pokemon lookup (`GET /pokemon/{name}`)

Case-insensitive lookup of `name` against the CSV's `name` column. Returns
`{"name", "height", "weight", "base_experience", "type1", "type2",
"actual_total_stats"}` (the last being the real sum of the six battle stats,
for comparing against a `/predict` call on the same inputs). `404` if the name
isn't found.

## Component 3: Music recommendation (`GET /pokemon/{name}/music`)

Reuses Component 2's lookup to get `type1`/`type2` for a real Pokemon, then:

1. **Type → genre mapping** (one fixed dict, 18 entries, applies to both
   `type1` and `type2` when looked up):
   `normal→pop, fire→rock, water→chill, electric→edm, grass→folk, ice→ambient,
   fighting→hip-hop, poison→punk, ground→country, flying→indie,
   psychic→trance, bug→techno, rock→metal, ghost→darkwave, dragon→orchestral,
   dark→industrial, steel→synthwave, fairy→dream-pop`.
2. `genre1 = map[type1]` (strict), `genre2 = map[type2]` if `type2` is present
   (used as a loose bias term, not a strict filter).
3. Query Spotify's `GET /v1/search` endpoint (**not** `/v1/recommendations`,
   which Spotify deprecated for apps created after Nov 2024 and may be
   unavailable to a freshly-registered app) with `q=genre:"{genre1}"
   {genre2}&type=track&limit=1`. If that returns nothing, retry with just
   `q=genre:"{genre1}"&type=track&limit=1` before giving up.
4. Auth: Client Credentials flow (app-only, no user login/consent screen) —
   `SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET` read from environment
   variables the user sets themselves after registering a free app at
   developer.spotify.com. Access token cached in memory, refreshed on expiry
   (Spotify tokens last ~1hr).

**Response:** `{"pokemon", "genre", "track": {"name", "artist", "spotify_url",
"preview_url"}}`.

**Errors:**
- Spotify credentials not configured → `503`, clear message (the other two
  endpoints remain fully functional — this is a soft dependency).
- Pokemon not found → `404` (same as Component 2).
- Both the strict and loosened Spotify queries return zero tracks → `404`
  ("no track found for genre ...").
- Spotify auth call itself fails (bad credentials, rate limited) → `502`.

## Deliverables

- `api/main.py` — FastAPI app: startup-time model + encoder + CSV loading,
  the three routes above, a small internal Spotify client module
  (`api/spotify_client.py`) wrapping token fetch/cache + search, so it can be
  mocked cleanly in tests.
- `Dockerfile` (standard capitalization — an earlier draft of this spec said
  `DockerFile`, which breaks `docker build .` on case-sensitive filesystems) —
  slim Python base image, installs `requirements.txt`
  (`fastapi`, `uvicorn`, `lightgbm`, `scikit-learn`, `pandas`, `requests`),
  copies `api/`, `data/`, `models/`, runs `python -m api.main`.
- `requirements.txt`.
- Tests (pytest, `TestClient`) for: the type-encoding reconstruction, the
  type→genre mapping and query-building logic (pure, no network), and each
  endpoint with the model/CSV loaded but Spotify calls mocked.
- README update: documents the three endpoints, the type-encoding caveat
  above, and the required `SPOTIFY_CLIENT_ID`/`SPOTIFY_CLIENT_SECRET` env
  vars.

## Risks / open questions

- The type→genre mapping is a fixed, opinionated dict with no empirical
  basis — it's a fun/cosmetic feature, not a data-driven recommendation, and
  is trivially editable if the mappings feel off once tried.
- Spotify's `genre:` search filter is not a strictly documented enum; results
  quality for less-common genre strings (e.g. `darkwave`, `dream-pop`) is
  unverified until actually queried against the live API.
- ~~Reconstructed type-encoding is a best-effort match to lost training
  preprocessing, not a verified exact match~~ — resolved: the original
  preprocessing code was found (`src/preprocess.py`) and the exact logic is now
  used (see Component 1's update).
