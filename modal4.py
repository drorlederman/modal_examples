import modal
import wandb
import os
import yaml
import enum
from training import train_model
from plot_results import plot_results
from download_results import download_results
import matplotlib
matplotlib.use('Agg')  # Save plots instead of opening a window

# Define Modal App (modern usage)
app = modal.App(name="cnn-dog-cat-classifier")

# Define Image with required Python packages
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "numpy",
    "matplotlib",
    "torch",
    "torchvision",
    "wandb",
    "pillow",
    "pyyaml",
    "wandb"
)

# Add required local source modules explicitly
image_with_source = image.add_local_python_source(
    "architecture",
    "download_results",
    "plot_results",
    "training",
)


class DeploymentEnvironment(enum.Enum):
    LOCAL = 1
    MODAL = 2

# Set this flag to control environment
deployment_environment = DeploymentEnvironment.LOCAL  # or DeploymentEnvironment.MODAL
# Define volumes and mount paths
#CONFIG_VOLUME_PATH = "/config_volume"
#DATA_VOLUME_PATH = "/data"
config_volume = modal.Volume.from_name("config_volume")
data_volume = modal.Volume.from_name("cats_vs_dogs")
WANDB_SECRET = modal.Secret.from_name("wandb-secret_opmed")

if deployment_environment == DeploymentEnvironment.LOCAL:
    CONFIG_VOLUME_PATH = os.path.abspath("./")
    DATA_VOLUME_PATH = os.path.abspath("./")
    CONFIG_FILE_PATH = os.path.join(CONFIG_VOLUME_PATH, "modal2_config.yaml")


    def run_training_local():
        CONFIG_FILE_PATH = 'modal2_config.yaml'
        with open(CONFIG_FILE_PATH, "r") as f:
            config = yaml.safe_load(f)

        folders = config["folders"]
        settings = config["settings"]
        model_path = folders["model_path"]
        data_path = folders["data_path"]
        local_download_path = folders["local_download_path"]
        use_gpu = settings["use_gpu"]
        target_size = settings["target_size"]
        batch_size = settings["batch_size"]
        epochs = settings["epochs"]
        print(f"Loaded config: {folders} | {settings}")
        print(f"Training on data_path={data_path}, model_path={model_path}")

        wandb.login(key=os.getenv("WANDB_API_KEY"))

        # Run the training
        epochs, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
            target_size=target_size,
            batch_size=batch_size,
            epochs=epochs,
            use_gpu=use_gpu,
            data_path=data_path,
            model_path=model_path
        )

        # Plot the results
        plot_results(epochs, train_losses, val_losses, train_accuracies, val_accuracies, model_path=model_path)

        # Optionally download results locally
        # download_results(model_path=model_path, local_download_dir=local_download_path)
else:
    CONFIG_VOLUME_PATH = "/config_volume"
    DATA_VOLUME_PATH = "/data"
    CONFIG_FILE_PATH = os.path.join(CONFIG_VOLUME_PATH, "modal2_config.yaml")

    # Main training function
    @app.function(
        volumes={DATA_VOLUME_PATH: data_volume, CONFIG_VOLUME_PATH: config_volume},
        image=image_with_source,
        gpu="any",
        secrets=[WANDB_SECRET]
    )
    def run_training():
        print("Listing files in /config_volume:")
        print(os.listdir(CONFIG_VOLUME_PATH))

        if not os.path.isfile(CONFIG_FILE_PATH):
            raise FileNotFoundError(f"Missing config file: {CONFIG_FILE_PATH}")

        with open(CONFIG_FILE_PATH, "r") as f:
            config = yaml.safe_load(f)

        folders = config["folders"]
        settings = config["settings"]
        model_path = folders["model_path"]
        data_path = folders["data_path"]
        local_download_path = folders["local_download_path"]
        use_gpu = settings["use_gpu"]
        target_size = settings["target_size"]
        batch_size = settings["batch_size"]
        epochs = settings["epochs"]
        print(f"Loaded config: {folders} | {settings}")
        print(f"Training on data_path={data_path}, model_path={model_path}")

        wandb.login(key=os.getenv("WANDB_API_KEY"))

        # Run the training
        epochs, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
            target_size=target_size,
            batch_size=batch_size,
            epochs=epochs,
            use_gpu=use_gpu,
            data_path=data_path,
            model_path=model_path
        )

        # Plot the results
        plot_results(epochs, train_losses, val_losses, train_accuracies, val_accuracies, model_path=model_path)

        # Optionally download results locally
        # download_results(model_path=model_path, local_download_dir=local_download_path)



# Entrypoint
if __name__ == "__main__":
    if deployment_environment == DeploymentEnvironment.LOCAL:
        run_training_local()
    elif deployment_environment == DeploymentEnvironment.MODAL:
        with app.run():
            run_training.remote()
    else:
        raise ValueError(f"Unknown deployment environment: {deployment_environment}")
