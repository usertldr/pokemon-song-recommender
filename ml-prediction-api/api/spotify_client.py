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
        try:
            resp = requests.post(
                TOKEN_URL,
                data={"grant_type": "client_credentials"},
                auth=(self.client_id, self.client_secret),
                timeout=10,
            )
        except requests.RequestException as exc:
            raise SpotifyAuthError(f"token request failed: {exc}") from exc
        if resp.status_code != 200:
            raise SpotifyAuthError(f"token request failed: {resp.status_code} {resp.text}")
        try:
            data = resp.json()
            self._token = data["access_token"]
            self._token_expires_at = time.time() + data["expires_in"] - 30
        except (ValueError, KeyError) as exc:
            raise SpotifyAuthError(f"malformed token response: {exc}") from exc
        return self._token

    def search_track(self, query: str) -> dict | None:
        token = self.get_access_token()
        try:
            resp = requests.get(
                SEARCH_URL,
                params={"q": query, "type": "track", "limit": 1},
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
        except requests.RequestException as exc:
            raise SpotifyAuthError(f"search failed: {exc}") from exc
        if resp.status_code != 200:
            raise SpotifyAuthError(f"search failed: {resp.status_code} {resp.text}")
        try:
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
        except (ValueError, KeyError) as exc:
            raise SpotifyAuthError(f"malformed search response: {exc}") from exc
