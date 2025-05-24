"""
Contains functionality for creating data folders, downloading data from any specified
data path, and returning the data path used.
"""
import os
import zipfile
import requests

from pathlib import Path

def from_path(from_path: str,              # e.g. "pizza_steak_sushi_20_percent.zip"
              image_dir: str) -> Path:     # e.g. "pizza_steak_sushi_20_percent"
  # Set up path to data folder
  data_path = Path("data/")
  image_path = data_path / image_dir       # data/pizza_steak_sushi_20_percent

  # If the image folder doesn't exist, download it and prepare it...
  if image_path.is_dir():
    print(f"{image_path} directory exists.")
  else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

    # Download images
    with open(data_path / from_path, "wb") as f:     # data/pizza_steak_sushi_20_percent.zip
      url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/" + from_path
      request = requests.get(url)
      print(f"Downloading {image_dir} data...")      # pizza_steak_sushi_20_percent
      f.write(request.content)

    # Unzip image data
    with zipfile.ZipFile(data_path / from_path, "r") as zip_ref:
      print(f"Unzipping {image_dir} data...")
      zip_ref.extractall(image_path)

    # Remove zip file
    os.remove(data_path / from_path)

  # For some reason os.system('any linux cmd') commands aren't working in this function, have to use popen or subprocess instead
  print(f">!ls {image_path}")
  cmd = f"ls {image_path}"
  output = os.popen(cmd).read()
  print(output)

  return image_path
