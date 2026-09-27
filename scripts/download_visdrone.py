"""Download and extract the VisDrone-DET splits into datasets/VisDrone/.

Ultralytics hosts mirror zips of the VisDrone-DET splits, which are far more
reliable to script than the original Google Drive / BaiduYun links. This uses
those mirrors. If they are ever unavailable, download manually from
https://github.com/VisDrone/VisDrone-Dataset and extract the split folders
into datasets/VisDrone/.

Usage:
    python scripts/download_visdrone.py                 # train + val + test-dev
    python scripts/download_visdrone.py --challenge     # also test-challenge
    python scripts/download_visdrone.py --root datasets/VisDrone
"""
import argparse
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

BASE = "https://github.com/ultralytics/assets/releases/download/v0.0.0"
ZIPS = {
    "train": f"{BASE}/VisDrone2019-DET-train.zip",
    "val": f"{BASE}/VisDrone2019-DET-val.zip",
    "test-dev": f"{BASE}/VisDrone2019-DET-test-dev.zip",
    "test-challenge": f"{BASE}/VisDrone2019-DET-test-challenge.zip",
}


def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            read += len(chunk)
            if total:
                pct = 100 * read / total
                print(f"\r    {read/1e6:7.1f} / {total/1e6:7.1f} MB ({pct:5.1f}%)",
                      end="", flush=True)
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/VisDrone")
    ap.add_argument("--challenge", action="store_true",
                    help="Also download the (unlabeled) test-challenge split.")
    args = ap.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    keys = ["train", "val", "test-dev"] + (["test-challenge"] if args.challenge else [])
    for key in keys:
        url = ZIPS[key]
        zpath = root / Path(url).name
        folder = root / Path(url).stem
        if folder.is_dir():
            print(f"[have] {folder.name} already extracted, skipping.")
            continue
        if not zpath.exists():
            download(url, zpath)
        print(f"  extracting {zpath.name}")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(root)
        zpath.unlink(missing_ok=True)
        print(f"[done] {folder.name}")

    print("\nAll requested splits ready under:", root.resolve())
    print("Next: python scripts/convert_visdrone.py --root", args.root)


if __name__ == "__main__":
    main()
