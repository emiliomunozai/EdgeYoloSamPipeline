import os
import urllib.request
import zipfile

ROOT = "data/raw"
BASE_URL = "http://images.cocodataset.org"

FILES = {
    "train2017.zip": f"{BASE_URL}/zips/train2017.zip",
    "val2017.zip": f"{BASE_URL}/zips/val2017.zip",
    "annotations_trainval2017.zip": f"{BASE_URL}/annotations/annotations_trainval2017.zip"
}

os.makedirs(ROOT, exist_ok=True)

def download(url, path):
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, path)

def unzip(path, target_dir):
    print(f"Extracting {path}")
    with zipfile.ZipFile(path, 'r') as z:
        z.extractall(target_dir)

for name, url in FILES.items():
    zip_path = os.path.join(ROOT, name)
    if not os.path.exists(zip_path):
        download(url, zip_path)
        unzip(zip_path, ROOT)
