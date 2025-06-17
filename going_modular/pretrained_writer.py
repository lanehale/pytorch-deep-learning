"""
Contains functions for using pretrained models (transfer learning) to train and
predict on new datasets, and track results with SummaryWriter.
"""
import torch
import torchvision

from torch import nn
from torchvision import transforms
from going_modular import data_setup, engine
from tqdm.auto import tqdm
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


"""
Contains a function to make predictions on test data.
"""
# Create a function to return a list of dictionaries with sample, label, prediction, pred prob
def predict_and_store(model, test_paths, transform, class_names, device):
  pred_list = []
  test_preds = []
  for path in test_paths:
    # Create empty dict to store info for each sample
    pred_dict = {}

    # Save sample path
    pred_dict["image_path"] = path

    # Save class name
    class_name = path.parent.stem
    pred_dict["class_name"] = class_name

    # Save prediction and pred prob
    img = Image.open(path).convert("RGB")  # ensure images are converted to RGB format before applying the default transformations
    transformed_image = transform(img).unsqueeze(dim=0).to(device)  # transform image and add batch dimension
    model.eval()
    with torch.inference_mode():
      pred_logit = model(transformed_image.to(device))
      pred_prob = torch.softmax(pred_logit, dim=1)
      pred_label = torch.argmax(pred_prob, dim=1)
      pred_class = class_names[pred_label.cpu()]       # or can replace .cpu()] with .item

      pred_dict["pred_prob"] = pred_prob.max().item()  # get the highest pred_prob
      pred_dict["pred_class"] = pred_class             # predicted class name

      test_preds.append(pred_label.cpu())

    # Does the prediction match the true label?
    pred_dict["correct"] = class_name == pred_class

    # Add sample dict to list of preds
    pred_list.append(pred_dict)

    test_preds_tensor = torch.cat(test_preds)

  return pred_list, test_preds_tensor


"""
Contains a function to run data through torchvision models
and track data with SummaryWriter.
"""
def run_model_writer(model,
                     weights,
                     train_dir,
                     test_dir,
                     batch_size,
                     dropout,
                     in_features,
                     optimizer_type,
                     optimizer_lr,
                     num_epochs,
                     image_data,
                     device,
                     writer,
                     model_name,
                     transform=None):

  if transform == None:
    transform = weights.transforms()  # default to auto transforms

  # Create training and testing DataLoaders and get a list of class names
  train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
      train_dir=train_dir,
      test_dir=test_dir,
      transform=transform,   # perform the same data transforms on our training data as the pretrained model
      batch_size=batch_size
  )

  # Freeze all base layers in the "features" section of the model
  #for parm in model.features.parameters():
  for parm in model.parameters():
    parm.requires_grad = False

  """ Adjust the output layer or the classifier portion of our pretrained model to our needs (out_features=3). """
  # Set manual seeds
  torch.manual_seed(42)
  torch.cuda.manual_seed(42)

  # Get the length of class_names (one output unit for each class)
  output_shape = len(class_names)

  # Recreate the classifier layer and seed it to the target device
  model.classifier = torch.nn.Sequential(
      torch.nn.Dropout(p=dropout, inplace=True),
      torch.nn.Linear(in_features=in_features,
                      out_features=output_shape,  # same number of output units as number of classes
                      bias=True)).to(device)

  # Define loss and optimizer
  loss_fn = nn.CrossEntropyLoss()
  if optimizer_type == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_lr)
  else:
    optimizer = torch.optim.SGD(model.parameters(), lr=optimizer_lr)

  """ Train the model """
  # Start the timer
  from timeit import default_timer as timer
  start_time = timer()

  """ Note: We're only going to be training the parameters classifier here as all of the other parameters in our model have been frozen. """
  # Set up training and save the results
  print(f"Training with model {model_name}...")
  results = engine.train_writer(model=model,
                                train_dataloader=train_dataloader,
                                test_dataloader=test_dataloader,
                                optimizer=optimizer,
                                loss_fn=loss_fn,
                                epochs=num_epochs,
                                device=device,
                                writer=writer)

  # End the timer and print out how long it took
  end_time = timer()
  print(f"[INFO] Total running time: {end_time - start_time:.3f} seconds")

  auto_transforms = weights.transforms()  # transform the image to predict but don't augment it

  # Make predictions and store in a list of dictionaries
  print(f"Predicting with {test_dir} image_data...")
  pred_list, test_preds_tensor = predict_and_store(
      model=model,
      test_paths=image_data,
      transform=auto_transforms,
      class_names=class_names,
      device=device
  )

  print(f"Max test acc: {max(results['test_acc']):.3f} | Min test loss: {min(results['test_loss']):.3f}")

  return results, pred_list
