"""
Makes predictions with a trained PyTorch model and saves the results to file.
"""
import torch
import torchvision
import argparse
import matplotlib.pyplot as plt

from going_modular import model_builder
from torchvision import transforms
from typing import List, Tuple
from PIL import Image


# Create a parser
parser = argparse.ArgumentParser()

# Create an arg for model path
parser.add_argument("--model_path",
                    default="models/05_going_modular_script_mode_tinyvgg_model.pth",
                    type=str,
                    help="filepath of model to use for prediction")

# Create an arg for image path
parser.add_argument("--image_path",
                    default="data/pizza_steak_sushi/train/pizza/12301.jpg",
                    type=str,
                    help="filepath of image to predict on")

# Create an arg for transform type
parser.add_argument("--transform",
                    default="no",
                    type=str,
                    help="yes to transform using horizontal flip, no is default")

# Create an arg for hidden units
parser.add_argument("--hidden_units",
                    default=10,
                    type=int,
                    help="number of hidden units used by model")

args = parser.parse_args()

model_path = args.model_path
image_path = args.image_path
transform = args.transform
hidden_units = args.hidden_units

print(f"[INFO] Predicting on {image_path} with {model_path}")

# Set up class names
class_names = ["pizza", "steak", "sushi"]

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Need to use same hyperparameters as saved model
model = model_builder.TinyVGG(input_shape=3,
                              hidden_units=hidden_units,
                              output_shape=3).to(device)  # len(class_names) = 3

# Load in the saved model state dictionary from file
model.load_state_dict(torch.load(model_path))

# 1. Load in an image and convert tensor values to float32
target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

# 2. Divide the image pixel values by 255 to get them between 0 and 1
target_image /= 255

# 3. Transform if necessary
if transform == "yes":
  data_transform_flip = transforms.Compose([
    transforms.Resize((224,224)),#64,64)),
    # Flip images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5),  # p = probability of flip, 0.5 = 50% chance
    #transforms.ToTensor()
  ])
  target_image = data_transform_flip(target_image)
  print("Using data_transform_flip")
else:  # Resize the image to be the same size as the model
  data_transform = transforms.Compose([
    transforms.Resize((224,224)),#64,64)),
    #transforms.ToTensor()
  ])
  target_image = data_transform(target_image)

# 4. Make sure model is on target device
model.to(device)

# 5. Turn on model evaluation and inference modes
model.eval()
with torch.inference_mode():

  # Add an extra dimension to image
  target_image = target_image.unsqueeze(dim=0)
  # Make a prediction on image with an extra dimension and send it to the target device
  target_image_pred = model(target_image.to(device))

# 6. Convert logits to probabilities
target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

# 7. Convert probs to label
target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

# 8. Plot the image alongside the prediction and prediction probability
plt.imshow(target_image.squeeze().permute(1, 2, 0))  # make sure it's right size for matplotlib
if class_names:
  title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
else:
  title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
plt.title(title)
plt.axis(False);

print(f"[INFO] Prediction label: {class_names[target_image_pred_label]}, prediction probability: {target_image_pred_probs.max():.3f}")


"""
Makes predictions with a trained PyTorch model and plots the results above the image.
"""
# 1. Take in a trained model, class names, image path, image size, a transform and target device
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device=device):
  # 2. Open image
  img = Image.open(image_path)

  # 3. Create transformation for image (if one doesn't exist)
  if transform is not None:
    image_transform = transform
  else:
    image_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

  ### Predict on image ###

  # 4. Make sure the model is on the target device
  model.to(device)

  # 5. Turn on model evaluation mode and inference mode
  model.eval()
  with torch.inference_mode():
    # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
    transformed_image = image_transform(img).unsqueeze(dim=0)

    # 7. Make a prediction on image with an extra dimension and send it to the target device
    target_image_pred = model(transformed_image.to(device))

  # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
  target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

  # 9. Convert prediction probabilities -> prediction labels
  target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

  # 10. Plot image with predicted label and probability
  plt.figure()
  plt.imshow(img)
  plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
  plt.axis(False);
