#!/usr/bin/env python3
"""Download the interview data files."""
import urllib.request
from pathlib import Path

# Some hosts block requests without a User-Agent header
_opener = urllib.request.build_opener()
_opener.addheaders = [("User-Agent", "Mozilla/5.0")]
urllib.request.install_opener(_opener)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

MANUALS = [
    {
        "name": "carrier-30xa-iom.pdf",
        "url": "https://www.shareddocs.com/hvac/docs/1013/Public/05/13452-76-07022015.pdf",
        "description": "Carrier 30XA Air-Cooled Screw Chillers — Installation, Operation & Maintenance (56 pages)",
    },
    {
        "name": "carrier-30xa-controls.pdf",
        "url": "https://chillers.com/wp-content/uploads/2019/04/Carrier-Chiller-30XA-Operations-Manual.pdf",
        "description": "Carrier 30XA — Controls, Configuration & Diagnostics (206 pages)",
    },
    {
        "name": "carrier-30xa-installation.pdf",
        "url": "https://brandportal.carrier.com/m/b61e47b5462d0ccd/original/13452_IOM_01_2013_30XA_2C_LR.pdf",
        "description": "Carrier 30XA — Installation Manual (52 pages)",
    },
]


def main():
    for manual in MANUALS:
        path = DATA_DIR / manual["name"]
        if path.exists():
            print(f"Already downloaded: {manual['name']}")
        else:
            print(f"Downloading: {manual['description']}")
            urllib.request.urlretrieve(manual["url"], path)
            print(f"  Saved: {path}")
        print(f"  Size: {path.stat().st_size / 1024:.0f} KB")
        print()

    print("Done. All manuals downloaded to data/")


if __name__ == "__main__":
    main()
