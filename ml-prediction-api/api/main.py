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
    try:
        genre = genre_for_type(type1)
        track = spotify.search_track(build_search_query(type1, type2))
        if track is None and type2:
            track = spotify.search_track(fallback_query(type1))
    except SpotifyAuthError as exc:
        return _error(502, "SPOTIFY_ERROR", str(exc))
    except ValueError as exc:
        return _error(400, "UNKNOWN_TYPE", str(exc))

    if track is None:
        return _error(404, "NO_TRACK_FOUND", f"no track found for genre {genre!r}")

    return {"pokemon": result["name"], "genre": genre, "track": track}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8012)
