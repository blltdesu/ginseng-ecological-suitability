from __future__ import annotations

import time
from pathlib import Path

import requests
from tqdm import tqdm

BASE_URL = "https://geodata.ucdavis.edu/cmip6/2.5m"

GCMS = [
    "ACCESS-CM2",
    "BCC-CSM2-MR",
    "MIROC6",
    "MPI-ESM1-2-HR",
    "MRI-ESM2-0",
]

SSPS = [
    "ssp126",
    "ssp585",
]

PERIODS = [
    "2041-2060",
    "2061-2080",
]

OUTPUT_ROOT = Path.home() / "ginseng_sdm" / "04_climate_future"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers.update(
    {"User-Agent": "ginseng-sdm-research/1.0 contact-your-email@example.com"}
)


def download_file(url: str, destination: Path) -> None:
    """支持断点续传的文件下载。"""
    destination.parent.mkdir(parents=True, exist_ok=True)

    downloaded = destination.stat().st_size if destination.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}

    with session.get(
        url,
        headers=headers,
        stream=True,
        timeout=300,
    ) as response:
        if response.status_code not in (200, 206):
            raise RuntimeError(
                f"下载失败：HTTP {response.status_code}\n{url}"
            )

        total = int(response.headers.get("content-length", 0))
        mode = "ab" if response.status_code == 206 and downloaded else "wb"

        with destination.open(mode) as file_handle:
            with tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=destination.name,
            ) as progress:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    file_handle.write(chunk)
                    progress.update(len(chunk))


def main() -> None:
    for gcm in GCMS:
        for ssp in SSPS:
            for period in PERIODS:
                filename = f"wc2.1_2.5m_bioc_{gcm}_{ssp}_{period}.tif"
                url = f"{BASE_URL}/{gcm}/{ssp}/{filename}"
                destination = OUTPUT_ROOT / gcm / ssp / filename

                if destination.exists():
                    print(f"\n已存在，跳过：{destination}")
                    continue

                print(f"\n下载：{url}")
                try:
                    download_file(url, destination)
                except Exception as exc:
                    print(f"失败：{exc}")
                    continue

                time.sleep(1)

    print("未来气候数据下载完成。")


if __name__ == "__main__":
    main()
