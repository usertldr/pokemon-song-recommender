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
