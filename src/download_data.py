import requests
import json
from pathlib import Path

METADATA_URL = ("https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items/xubh-q36u?show-reference-ids=false")

OUTPUT_DIR = Path("data/raw")

def get_download_url():
    """Retrieve the current CMS dataset download URL."""

    response = requests.get(METADATA_URL, timeout=30)
    response.raise_for_status()

    metadata = response.json()

    download_url = metadata["distribution"][0]["data"]["downloadURL"]

    return download_url

def download_dataset(download_url):
    """Download the CMS hospital dataset to the raw data directory."""
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "Hospital_General_Information.csv"

    response = requests.get(download_url, timeout=60)
    response.raise_for_status()

    output_file.write_bytes(response.content)

    print(f"Dataset downloaded to: {output_file}")
    print(f"File size: {output_file.stat().st_size / (1024 * 1024):.2f} MB")

def main():
    print("Retrieving CMS dataset metadata...")

    download_url = get_download_url()

    print("Current CMS download URL found.")
    print("Downloading dataset...")

    download_dataset(download_url)

    print("Data ingestion complete.")

if __name__ == "__main__":
    main() 