"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import os
import torch
import argparse

from going_modular import data_setup, engine, model_builder, utils
from torchvision import transforms
from pathlib import Path

# Create a parser
parser = argparse.ArgumentParser(description="Get some hyperparameters.")

# Create an arg for number of epochs
parser.add_argument("--num_epochs",
                    default=10,
                    type=int,
                    help="the number of epochs to train for")

# Create an arg for batch size
parser.add_argument("--batch_size",
                    default=32,
                    type=int,
                    help="number of samples per batch")

# Create an arg for hidden units
parser.add_argument("--hidden_units",
                    default=10,
                    type=int,
                    help="number of hidden units in hidden layers")

# Create an arg for learning rate
parser.add_argument("--learning_rate",
                    default=0.001,
                    type=float,
                    help="learning rate to use for model")

# Create an arg for training directory
parser.add_argument("--train_dir",
                    default="data/pizza_steak_sushi/train",
                    type=str,
                    help="directory file path to training data in standard image classification format")

# Create an arg for test directory
parser.add_argument("--test_dir",
                    default="data/pizza_steak_sushi/test",
                    type=str,
                    help="directory file path to testing data in standard image classification format")

# Create an arg for model name
parser.add_argument("--model_name",
                    default="model_0.pth",
                    type=str,
                    help="model name to save")

# Get our arguments from the parser
args = parser.parse_args()

# Set up hyperparameters
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate
print(f"[INFO] Training a model for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE}, hidden units {HIDDEN_UNITS} and learning rate {LEARNING_RATE}...")

# Set up directories
train_dir = args.train_dir
test_dir = args.test_dir
print(f"[INFO] Training data file: {train_dir}")
print(f"[INFO] Testing data file: {test_dir}")

model_name = args.model_name
print(f"[INFO] Model will be saved as: {model_name}")

# Set up target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr=LEARNING_RATE)
#optimizer = torch.optim.Adam(model.parameters(),
#                             lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model with help from utils.py
target_dir = Path("saved_models")

utils.save_model(
    model=model,
    target_dir=target_dir,
    model_name=model_name,
    device=device
)
