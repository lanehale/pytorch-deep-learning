"""
Contains functionality for creating data folders and downloading from any specified data path.
"""
import os

# Try to import the going_modular directory, download it from GitHub if it doesn't work
try:
  from going_modular import data_setup, engine
  print("going_modular scripts already downloaded.")
except:
  """
  This block attempts to download a GitHub repository,
  move a specific directory from the downloaded repository to the current working directory,
  and then remove the downloaded repository.
  """
  # Get the going_modular scripts
  print("[INFO] Couldn't find going_modular scripts... downloading them from GitHub.")

  # Clone the git repository
  os.system('git clone https://github.com/lanehale/pytorch-deep-learning')

  # When cloning a GitHub repository, the directory structure on your local machine doesn't include /tree/main/, so it shouldn't be included in the mv command.
  # The . at the end of the command tells mv to move the specified directory into the current working directory.
  os.system('mv pytorch-deep-learning/going_modular .')

  # remove the downloaded repository
  os.system('rm -rf pytorch-deep-learning')

  # move these two files out to parent directory
  os.system('mv going_modular/train.py .')
  os.system('mv going_modular/predict.py .')

  from going_modular import data_setup, engine

  print(">!ls")
  os.system('ls')
  print(">!ls going_modular/")
  os.system('ls going_modular/')
