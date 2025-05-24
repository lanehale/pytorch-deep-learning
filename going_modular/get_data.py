"""
Contains functionality for creating data folders and downloading pizza_steak_sushi data.
"""
import os
import requests
import zipfile
from pathlib import Path

# Set up path to data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# If the image folder doesn't exist, download it and prepare it...
if image_path.is_dir():
  print(f"{image_path} directory exists.")
else:
  print(f"Did not find {image_path} directory, creating one...")
  image_path.mkdir(parents=True, exist_ok=True)

  # Download images
  with open(data_path / "pizza_steak_sushi_20_percent.zip", "wb") as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip")
    print("Downloading pizza, steak, sushi data...")
    f.write(request.content)

  # Unzip image data
  with zipfile.ZipFile(data_path / "pizza_steak_sushi_20_percent.zip", "r") as zip_ref:
    print("Unzipping pizza, steak, sushi data...")
    zip_ref.extractall(image_path)

  # Remove zip file
  os.remove(data_path / "pizza_steak_sushi_20_percent.zip")


"""
Contains functionality for creating data folders, downloading data from any specified
data path, and returning the data path used.
"""
def from_path(from_path: str,              # e.g. "pizza_steak_sushi_20_percent.zip"
              image_dir: str) -> Path:     # e.g. "pizza_steak_sushi_20_percent"
  # Set up path to data folder
  data_path = Path("data/")
  image_path = data_path / image_dir  # "pizza_steak_sushi"

  # If the image folder doesn't exist, download it and prepare it...
  if image_path.is_dir():
    print(f"{image_path} directory exists.")
  else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

    # Download images
    with open(data_path / from_path, "wb") as f:  # "pizza_steak_sushi_20_percent.zip"
      #url = Path("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/") / from_path  # Path removes extra slash
      url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/" + from_path
      request = requests.get(url)
      print("Downloading {image_dir} data...")    # pizza, steak, sushi
      f.write(request.content)

    # Unzip image data
    with zipfile.ZipFile(data_path / from_path, "r") as zip_ref:  # "pizza_steak_sushi_20_percent.zip"
      print("Unzipping {image_dir} data...")      # pizza, steak, sushi
      zip_ref.extractall(image_path)

    # Remove zip file
    os.remove(data_path / from_path)  # "pizza_steak_sushi_20_percent.zip"

  # For some reason os.system('any linux cmd') commands aren't working in this function, have to use popen or subprocess instead
  print(f">!ls {image_path}")
  cmd = f"ls {image_path}"  # "ls data/pizza_steak_sushi"
  output = os.popen(cmd).read()
  print(output)

  return image_path
