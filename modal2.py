import modal
from training import train_model
from plot_results import plot_results
from download_results import download_results
import wandb  # Weights and Biases for tracking
import os
import yaml
from read_config import read_config

app = modal.App(name="cnn-dog-cat-classifier")
wandb.login(key=os.getenv("WANDB_API_KEY"))
# Define Modal image with PyTorch and dependencies
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "numpy",
    "matplotlib",
    "torch",
    "torchvision",
    "wandb",
    "pillow",
    "pyyaml"
)

volume = modal.Volume.from_name("cats_vs_dogs")
CONFIG_PATH = "/config/modal2_config.yaml"  # Path inside Modal
MODEL_PATH = "/data"
wandb.login(key=os.getenv("WANDB_API_KEY"))

#Since Modal runs in a remote environment, your YAML config file needs to be stored in a volume.
#modal volume create config_volume
#modal volume put config_volume config.yaml

read_config()

@app.function(
    volumes={MODEL_PATH: volume, CONFIG_PATH: volume},
    image=image,
    gpu="any",
    secrets=[modal.Secret.from_name("wandb_api_key")]# Set valid GPU configuration
)

def run_training(): #model_path, data_path, local_download_path, use_gpu):
    # Load YAML config
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    print("Loaded Config:", config)
    model_path, data_path, local_download_path, use_gpu = read_config()

    epochs, train_losses, val_losses, train_accuracies, val_accuracies = train_model(target_size=(150, 150), batch_size=32, epochs=10,
                                                                                     use_gpu=use_gpu,
                                                                                     data_path=data_path, model_path=model_path)
    plot_results(epochs, train_losses, val_losses, train_accuracies, val_accuracies, model_path=model_path)
    download_results(model_path=model_path, local_download_dir=local_download_path)



