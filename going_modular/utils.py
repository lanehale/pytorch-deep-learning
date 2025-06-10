"""
Contains various utility functions for PyTorch model training.
"""
import torch
import torchvision
import matplotlib.pyplot as plt

from torch import nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from typing import List, Tuple
from pathlib import Path
from PIL import Image

# Add import for os and datetime to fix potential NameErrors later
import os
from datetime import datetime

"""
Save a PyTorch model to a target directory.
"""
def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create a target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True, exist_ok=True)
  
  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"  # text to display if assert check fails
  model_save_path = target_dir + "/" + model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)


"""
Make predictions on images and plot them.
"""
# 1. Take in a trained model, class names, image path, image size, a transform and target device
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        extra_title: bool = False,
                        device: torch.device='cpu'):               # Removed the default device=device here
  """Makes a prediction on a target image and plots the image and prediction.

  Args:
      model: A PyTorch model to make a prediction on.
      image_path: The path to the target image to make a prediction on.
      class_names: A list of the class names for the model.
      image_size: The size the image should be resized to before being passed to
          the model.
      transform: A torchvision transforms object to transform the target image
          before being passed to the model. If None, a transforms.Resize +
          transforms.ToTensor() will be created.
      device: The device the model is on (e.g. "cuda" or "cpu").
  """
  # 2. Open image
  img = Image.open(image_path)

  # 3. Transform the image
  if transform is not None:
    # Use the provided transform
    image_transform = transform
  else:
    # Otherwise use a standard transform
    image_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
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
  if extra_title == True:
    plt.title(f"{image_path}\nPred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
  else:
    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
  plt.axis(False);


"""
Plot loss and accuracy curves of a model.
"""
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of results, e.g.
            {'train_loss': [...],
             'train_acc': [...],
             'test_loss': [...],
             'test_acc': [...]}
    """
    loss = results['train_loss']
    test_loss = results['test_loss']

    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    epochs = range(len(results['train_loss']))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()


"""
Add a helper function to create a SummaryWriter
"""
def create_writer(experiment_name: str,
                  model_name: str,
                  extra: str=None) -> torch.utils.tensorboard.SummaryWriter():
  """Creates a torch.utils.tensorboard.SummaryWriter() instance saving to a specific log_dir.

  Args:
      experiment_name (str): Name of experiment.
      model_name (str): Name of model.
      extra (str, optional): Anything extra to add to the log directory. Defaults to None.

  Returns:
      torch.utils.tensorboard.SummaryWriter(): Instance of a SummaryWriter saving to log_dir.

  Example usage:
    # Create a writer saving to "runs/2025-05-24/data_10_percent/effnetb2/5_epochs/"
    writer = create_writer(experiment_name="data_10_percent",
                           model_name="effnetb2",
                           extra="5_epochs")

    # The above is the same as:
    writer = SummaryWriter(log_dir="runs/2025-05-24/data_10_percent/effnetb2/5_epochs/")

  """
  # Get timestamp of current date (all experiments on certain day live in same folder)
  timestamp = datetime.now().strftime("%Y-%m-%d")  # returns current date in YYYY-MM-DD format

  if extra:
    # Create log directory path
    log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
  else:
    log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

  print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")

  return SummaryWriter(log_dir=log_dir)


"""
Create a function to display results
"""
def compare_results(pred_list, name, col1_width):
  false_count = 0
  for pred in pred_list:
    if pred['correct'] == False:
      false_count += 1
  false_percent = 100 * false_count / len(pred_list)
  formatted_string = ("%-"+str(col1_width)+"s") % name   # e.g. formatted_string = "%-45s" % name
  print(
      f"{formatted_string} | False predictions: {false_count :<2} out of {len(pred_list) :<3}, "
      f"or {false_percent:5.2f}% wrong, "
      f"{(100.0 - false_percent):.2f}% right"
  )


"""
Create a function to set seeds
"""
def set_seeds(seed: int=42):
  """Sets random sets for torch operations.

  Args:
    seed (int, optional): Random seed to set. Defaults to 42.
  """
  # Set the seed for general torch operations
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
