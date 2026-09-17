import time
import pytest
import requests
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


def test_get_access_token_wraps_request_exception(monkeypatch):
    def raise_connection_error(url, data=None, auth=None, timeout=None):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(sc.requests, "post", raise_connection_error)

    client = SpotifyClient("id", "secret")
    with pytest.raises(SpotifyAuthError):
        client.get_access_token()


def test_search_track_wraps_request_exception(monkeypatch):
    monkeypatch.setattr(
        sc.requests, "post",
        lambda url, data=None, auth=None, timeout=None: FakeResponse(200, {"access_token": "tok", "expires_in": 3600}),
    )

    def raise_connection_error(url, params=None, headers=None, timeout=None):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(sc.requests, "get", raise_connection_error)

    client = SpotifyClient("id", "secret")
    with pytest.raises(SpotifyAuthError):
        client.search_track('genre:"rock"')
