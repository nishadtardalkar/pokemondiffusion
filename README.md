## Pokémon Sprite Scraper

This is a small helper script to download Pokémon sprites using the public [PokeAPI](https://pokeapi.co/).

Sprites are saved into subfolders per Pokémon, with many variants (front/back, shiny, official artwork, versioned sprites, etc.) when available.

### Install

From this folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Usage

Basic usage (first 151 Pokémon, saved to `./sprites`):

```bash
python scrape_pokemon_sprites.py --start-id 1 --end-id 151
```

Custom output directory and slower rate (to be extra gentle to the API):

```bash
python scrape_pokemon_sprites.py --start-id 1 --end-id 898 --out-dir data\sprites --delay 1.0
```

Arguments:

- **`--start-id`**: First Pokémon ID (inclusive). Default: `1`.
- **`--end-id`**: Last Pokémon ID (inclusive). Default: `1025`.
- **`--out-dir`**: Output directory. Default: `./sprites`.
- **`--delay`**: Seconds to wait between API calls. Default: `0.7` (recommended \>= 0.5 to respect PokeAPI).

### Notes

- This uses PokeAPI’s JSON to find **official, already-hosted** sprite URLs; it does not scrape any fan sites.
- Be respectful of PokeAPI: avoid extremely low delays and heavy parallelization.

