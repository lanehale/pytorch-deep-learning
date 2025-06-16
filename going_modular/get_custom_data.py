"""
Contains functionality for downloading custom images from GitHub.
"""
import requests
from pathlib import Path

data_path = Path("data")

# Get multiple custom images
"""
To get raw address:  right click jpeg file name in main repository view, select copy link address, paste into browser and enter. Then
right click on image, select copy link address = https://github.com/lanehale/pytorch-deep-learning/blob/main/cheese-pizza.jpeg?raw=true
paste into browser and enter to get url format below
"""
urls = [
    "https://raw.githubusercontent.com/lanehale/pytorch-deep-learning/refs/heads/main/custom_images/cheese-pizza.jpeg",
    "https://raw.githubusercontent.com/lanehale/pytorch-deep-learning/refs/heads/main/custom_images/pizza-slice.jpeg",
    "https://raw.githubusercontent.com/lanehale/pytorch-deep-learning/refs/heads/main/custom_images/pizza-slice2.jpeg",
    "https://raw.githubusercontent.com/lanehale/pytorch-deep-learning/refs/heads/main/custom_images/pizza-sliced.jpeg",
    "https://raw.githubusercontent.com/lanehale/pytorch-deep-learning/refs/heads/main/custom_images/pizza-sliced2.jpeg",
    "https://raw.githubusercontent.com/lanehale/pytorch-deep-learning/refs/heads/main/custom_images/pizza-partial-view.jpeg",
    "https://raw.githubusercontent.com/lanehale/pytorch-deep-learning/refs/heads/main/custom_images/pizza-partial-view2.jpeg",
    "https://raw.githubusercontent.com/lanehale/pytorch-deep-learning/refs/heads/main/custom_images/pizza-side-view.jpeg"
]

filenames = [
    "cheese-pizza.jpeg",
    "pizza-slice.jpeg",
    "pizza-slice2.jpeg",
    "pizza-sliced.jpeg",
    "pizza-sliced2.jpeg",
    "pizza-partial-view.jpeg",
    "pizza-partial-view2.jpeg",
    "pizza-side-view.jpeg"
]

if len(urls) != len(filenames):
  raise ValueError("The number of URLs and filenames must be the same.")

# Download the images if they don't already exist
if (data_path / "cheese-pizza.jpeg").is_file():
  print(f"Custom images already exist, skipping download.")
else:
  for i, url in enumerate(urls):
    try:
      response = requests.get(url)
      response.raise_for_status()  # Raise an exception for HTTP errors

      custom_image_path = data_path / filenames[i]

      with open(custom_image_path, "wb") as f:
        f.write(response.content)

      print(f"DownLoading {custom_image_path}...")

    except requests.exceptions.RequestException as e:
      print(f"Error downloading {url}: {e}")
    except Exception as e:
      print(f"An unexpected error occurred: {e}")
