"""
Contains functions for using pretrained models (transfer learning) to train and
predict on new datasets, and display results in a confusion matrix.
"""
import torch
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
import torchmetrics, mlxtend

from torch import nn
from torchvision import transforms
from going_modular import data_setup, engine
from tqdm.auto import tqdm

# Import specific functionalities from torchmetrics and mlxtend
from torchmetrics.classification import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix


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
    from PIL import Image
    img = Image.open(path)
    transformed_image = transform(img).unsqueeze(dim=0)  # transform image and add batch dimension
    model.eval()
    with torch.inference_mode():
      pred_logit = model(transformed_image.to(device))
      pred_prob = torch.softmax(pred_logit, dim=1)
      pred_label = torch.argmax(pred_prob, dim=1)
      pred_class = class_names[pred_label.cpu()]  # or could replace .cpu() with .item() since pred_label is a scalar value (index to the class_name)

      # Make sure the highest pred_prob is back on the CPU for the dictionary
      pred_dict["pred_prob"] = pred_prob.unsqueeze(0).max().cpu().item()
      pred_dict["pred_class"] = pred_class  # predicted class name

      test_preds.append(pred_label.cpu())

    # Does the prediction match the true label?
    pred_dict["correct"] = class_name == pred_class

    # Add sample dict to list of preds
    pred_list.append(pred_dict)

    test_preds_tensor = torch.cat(test_preds)

  return pred_list, test_preds_tensor


"""
Contains a function to run data through torchvision models,
display a confusion matrix, predict, store and return results.
"""
def run_model(model,
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
              device):

  auto_transforms = weights.transforms()

  # Create training and testing DataLoaders and get a list of class names
  train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
      train_dir=train_dir,
      test_dir=test_dir,
      transform=auto_transforms,  # perform the same data transforms on our training data as the pretrained model
      batch_size=batch_size
  )

  # Freeze all base layers in the "features" section of the model
  for parm in model.features.parameters():
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
  print(f"Training the model...")
  results = engine.train(model=model,
                         train_dataloader=train_dataloader,
                         test_dataloader=test_dataloader,
                         optimizer=optimizer,
                         loss_fn=loss_fn,
                         epochs=num_epochs,
                         device=device)

  # End the timer and print out how long it took
  end_time = timer()
  print(f"[INFO] Total running time: {end_time - start_time:.3f} seconds")

  print(f"Max test acc: {max(results['test_acc']):.3f} | Min test loss: {min(results['test_loss']):.3f}")

  # Make predictions and store in a list of dictionaries
  print("Predicting with image_data...")
  pred_list, test_preds_tensor = predict_and_store(
      model=model,
      test_paths=image_data,
      transform=auto_transforms,
      class_names=class_names,
      device=device
  )

  false_count = 0
  for pred in pred_list:
    if pred['correct'] == False:
      false_count += 1
  false_percent = 100 * false_count / len(pred_list)
  print(
      f"{name :<10} | False predictions: {false_count :<2} out of {len(pred_list) :<3}, "
      f"or {false_percent:5.2f}% wrong, "
      f"{(100.0 - false_percent):.2f}% right"
  )

  # Setup confusion matrix instance and compare predictions to targets
  confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')

  # Get truth labels for test dataset
  """ We can't use these since the test_image_path_list order isn't the same as the dataset's sample order """
  #test_truth = torch.cat([y for X, y in test_dataloader])

  """
  Build a truths tensor from the class order in test_image_path_list since it isn't
  the same order as the dataset in the test_dataloader. Without a true labels tensor
  the confusion matrix won't be correct.
  """
  class_indices_list = []
  for item in pred_list:
    if item["class_name"] == "pizza":
      index = 0;
    elif item["class_name"] == "steak":
      index = 1;
    elif item["class_name"] == "sushi":
      index = 2;
    class_indices_list.append(index)

  true_labels_tensor = torch.tensor(class_indices_list)

  confmat_tensor = confmat(preds=test_preds_tensor,
                           true_labels_tensor)       # can't use this, target=test_truth)

  # Plot the confusion matrix
  fig, ax = plot_confusion_matrix(
      conf_mat=confmat_tensor.numpy(),
      class_names=class_names,
      figsize=(10, 7)
  );

  return results, pred_list


"""
Contains functionality to plot the N most wrong predictions.
"""
def plot_N_most_wrong(pred_list, n):

  # Sort the list of dictionaries, False elements first, pred_prob high to low
  sorted_pred_list = sorted(pred_list, key=lambda x: (x['correct']==False, x['pred_prob']), reverse=True)

  # Turn sorted list into a DataFrame of top 5 wrong
  test_pred_df = pd.DataFrame(sorted_pred_list[:n])

  # Plot the 5 most wrong images
  for row in test_pred_df.iterrows():
    row = row[1]
    image_path = row["image_path"]
    true_label = row["class_name"]
    pred_class = row["pred_class"]
    pred_prob = row["pred_prob"]

    img = torchvision.io.read_image(str(image_path)).permute(1, 2, 0)  # get image as tensor and permute to [height, width, color_channels]
    plt.imshow(img)
    plt.title(f"True: {true_label} | Pred: {pred_class} | Prob: {pred_prob:.4f}")
    plt.axis(False)
    plt.show()
