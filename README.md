# Pokemon Song Recommender

A small FastAPI service with two things bolted together:

- **Stats prediction** — a LightGBM model predicts a Pokemon's total base stats
  (HP+Attack+Defense+SpAttack+SpDefense+Speed) from its height, weight, base
  experience, and type(s).
- **Music recommendation** — maps a Pokemon's type(s) to a music genre and looks up
  a real matching track via the Spotify API.

Full documentation, endpoints, and setup instructions: [ml-prediction-api/README.md](ml-prediction-api/README.md)

Design spec: [docs/superpowers/specs/2026-09-16-pokemon-stats-music-api-design.md](docs/superpowers/specs/2026-09-16-pokemon-stats-music-api-design.md)
Implementation plan: [docs/superpowers/plans/2026-09-16-pokemon-stats-music-api.md](docs/superpowers/plans/2026-09-16-pokemon-stats-music-api.md)
