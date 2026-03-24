from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


POKEAPI_BASE_URL = "https://pokeapi.co/api/v2"

# Basic sprite fields to download from the "sprites" object
BASE_SPRITE_FIELDS = [
    "front_default",
]


def create_session() -> requests.Session:
    """Create a requests session with retry and a custom User-Agent."""
    session = requests.Session()

    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers["User-Agent"] = "PokemonSpriteScraper/1.0 (https://pokeapi.co/)"
    return session


def fetch_pokemon_data(
    session: requests.Session, identifier: int | str
) -> Optional[Dict[str, Any]]:
    """Fetch Pokémon data by numeric ID or name."""
    url = f"{POKEAPI_BASE_URL}/pokemon/{identifier}"
    resp = session.get(url, timeout=15)

    if resp.status_code == 404:
        print(f"[WARN] Pokémon not found: {identifier}")
        return None

    resp.raise_for_status()
    return resp.json()


def download_image(session: requests.Session, url: str, dest_path: Path) -> None:
    """Download a single image if it does not already exist."""
    if dest_path.exists():
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    resp = session.get(url, stream=True, timeout=30)
    resp.raise_for_status()

    with dest_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def save_sprites_for_pokemon(
    session: requests.Session,
    pokemon: Dict[str, Any],
    out_dir: Path,
) -> None:
    """Extract and save only the base sprites for a single Pokémon."""
    pid = pokemon.get("id")
    name = pokemon.get("name", f"pokemon-{pid}")
    sprites = pokemon.get("sprites") or {}

    # Basic sprites
    for field in BASE_SPRITE_FIELDS:
        url = sprites.get(field)
        if not url:
            continue
        # Single file per Pokémon, named by zero-padded ID
        dest = out_dir / f"{pid:04d}.png"
        try:
            download_image(session, url, dest)
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] Failed to download {name} {field}: {e}")


def scrape_pokemon_range(
    start_id: int,
    end_id: int,
    out_dir: Path,
    delay: float,
) -> None:
    """Scrape sprites for a range of Pokémon IDs (inclusive)."""
    session = create_session()
    out_dir.mkdir(parents=True, exist_ok=True)

    for pid in range(start_id, end_id + 1):
        try:
            data = fetch_pokemon_data(session, pid)
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] Failed to fetch data for ID {pid}: {e}")
            continue

        if not data:
            continue

        name = data.get("name", f"pokemon-{pid}")
        print(f"[INFO] {pid}: {name}")

        save_sprites_for_pokemon(session, data, out_dir)

        if delay > 0:
            time.sleep(delay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Pokémon sprites from PokeAPI.\n\n"
            "Examples:\n"
            "  python scrape_pokemon_sprites.py --start-id 1 --end-id 151\n"
            "  python scrape_pokemon_sprites.py --out-dir sprites --delay 0.8"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="First Pokémon ID to download (inclusive). Default: 1",
    )
    parser.add_argument(
        "--end-id",
        type=int,
        default=1025,
        help="Last Pokémon ID to download (inclusive). Default: 1025",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("sprites"),
        help="Output directory where sprites will be saved. Default: ./sprites",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.7,
        help=(
            "Delay in seconds between API calls to be gentle to PokeAPI.\n"
            "Recommended >= 0.5. Default: 0.7"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.end_id < args.start_id:
        raise SystemExit("--end-id must be greater than or equal to --start-id")

    scrape_pokemon_range(
        start_id=args.start_id,
        end_id=args.end_id,
        out_dir=args.out_dir,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()

