"""Download public source datasets, with resume, progress, and safe extraction."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import shutil
import stat
import time
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / "data/downloads"
DATASETS = {
    "sh17": ("mugheesahmad/sh17-dataset-for-ppe-detection", ROOT / "data/sh17"),
    "dfire": ("sayedgamal99/smoke-fire-detection-yolo", ROOT / "data/dfire/dataset"),
}


def download(name, dataset, destination):
    status_path = DOWNLOADS / f"{name}-status.json"
    archive = DOWNLOADS / f"{name}.zip"
    partial = DOWNLOADS / f"{name}.zip.part"
    url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset}"
    status = {"dataset": dataset, "url": url, "destination": str(destination)}

    def update(**values):
        status.update(values, updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        temp = status_path.with_suffix(".tmp")
        for retry in range(10):
            try:
                temp.write_text(json.dumps(status, indent=2), encoding="utf-8")
                temp.replace(status_path)
                return
            except PermissionError:
                # A status reader or Windows scanner can briefly hold the file.
                # Progress reporting must not interrupt the dataset transfer.
                time.sleep(0.1)
        print(f"{name}: status file temporarily locked; transfer continues", flush=True)

    try:
        if not archive.exists():
            for attempt in range(1, 31):
                downloaded = partial.stat().st_size if partial.exists() else 0
                update(state="downloading", downloaded_bytes=downloaded, attempt=attempt)
                headers = {"User-Agent": "jetson-safety-dataset-setup/1.0"}
                if downloaded:
                    headers["Range"] = f"bytes={downloaded}-"
                try:
                    with urlopen(Request(url, headers=headers), timeout=60) as response:
                        if "zip" not in response.headers.get("Content-Type", ""):
                            raise RuntimeError("Download returned something other than a ZIP archive")
                        if downloaded and response.status != 206:
                            raise RuntimeError("Server did not honor resume; existing partial file preserved")
                        content_range = response.headers.get("Content-Range", "")
                        if downloaded and not content_range.startswith(f"bytes {downloaded}-"):
                            raise RuntimeError("Resume response starts at the wrong byte")
                        size = int(response.headers["Content-Length"]) + downloaded
                        if shutil.disk_usage(DOWNLOADS).free < size - downloaded + 1024**3:
                            raise RuntimeError("Insufficient free disk space for remaining download")
                        update(total_bytes=size)
                        last_update = time.monotonic()
                        with partial.open("ab" if downloaded else "wb") as target:
                            while chunk := response.read(1024 * 1024):
                                target.write(chunk)
                                downloaded += len(chunk)
                                if time.monotonic() - last_update >= 2:
                                    update(downloaded_bytes=downloaded)
                                    last_update = time.monotonic()
                        if downloaded != size:
                            raise RuntimeError(f"Incomplete download: {downloaded} / {size}")
                    partial.replace(archive)
                    update(downloaded_bytes=downloaded)
                    break
                except Exception as error:
                    update(last_error=str(error))
                    if attempt == 30:
                        raise
                    time.sleep(min(2 * attempt, 30))
        update(state="verifying")
        with zipfile.ZipFile(archive) as source:
            entries = source.infolist()
            expanded = sum(entry.file_size for entry in entries)
            if shutil.disk_usage(DOWNLOADS).free < expanded + 1024**3:
                raise RuntimeError("Insufficient free disk space to extract dataset")
            base = destination.resolve()
            for entry in entries:
                target = (base / entry.filename).resolve()
                if not target.is_relative_to(base) or "\\" in entry.filename:
                    raise RuntimeError(f"Unsafe ZIP path: {entry.filename}")
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise RuntimeError(f"Symlink not allowed in ZIP: {entry.filename}")
            update(state="extracting", total_files=len(entries), extracted_files=0,
                   expanded_bytes=expanded)
            destination.mkdir(parents=True, exist_ok=True)
            for index, entry in enumerate(entries, 1):
                # ZipFile.extract checks CRC while reading every entry.
                source.extract(entry, destination)
                if index % 100 == 0:
                    update(extracted_files=index)
        update(state="complete", extracted_files=len(entries), archive_bytes=archive.stat().st_size)
        print(f"{name}: complete -> {destination}", flush=True)
        return True
    except Exception as error:
        update(state="failed", error=str(error))
        print(f"{name}: failed: {error}", flush=True)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, help="Download only this dataset")
    args = parser.parse_args()
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    selected = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(download, name, *spec) for name, spec in selected.items()]
        succeeded = [job.result() for job in jobs]
    raise SystemExit(0 if all(succeeded) else 1)
