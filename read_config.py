import modal
import wandb  # Weights and Biases for tracking
import os
import yaml

# Define Modal App at the beginning
app = modal.App(name="cnn-dog-cat-classifier")

# Ensure W&B API key is set
wandb.login(key=os.getenv("WANDB_API_KEY"))

# Define Modal Image with Dependencies
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "numpy",
    "matplotlib",
    "torch",
    "torchvision",
    "wandb",
    "pillow",
    "pyyaml"
)

# ✅ Define Modal Volume (MOUNT IT AT A DIRECTORY)
VOLUME_MOUNT_PATH = "/config_volume"  # Mount the directory
CONFIG_FILE_PATH = os.path.join(VOLUME_MOUNT_PATH, "modal2_config.yaml")  # Correct file path
volume = modal.Volume.from_name("config_volume")

# ✅ Define function to read the config file
@app.function(
    volumes={VOLUME_MOUNT_PATH: volume},  # ✅ Mount the volume at a directory
    secrets=[modal.Secret.from_name("wandb_api_key")],  # Set valid GPU configuration
    image=image
)
def read_config():
    """ Read and parse the config YAML file. """
    print("Checking files in /config_volume:")
    try:
        files = os.listdir("/config_volume")
        print(files)  # ✅ Print all files inside the directory
    except Exception as e:
        print(f"Error listing files: {e}")
    if not os.path.isfile(CONFIG_FILE_PATH):
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE_PATH}")

    with open(CONFIG_FILE_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Extract settings
    db = config['folders']
    data_path = db['data_path']
    model_path = db['model_path']
    local_download_path = db['local_download_path']
    settings = config['settings']
    use_gpu = settings['use_gpu']
    target_size = settings['target_size']
    batch_size = settings['batch_size']
    epochs = settings['epochs']

    print(f"Loaded config: model_path={model_path}, data_path={data_path}, local_download_path={local_download_path}, use_gpu={use_gpu}")
    return model_path, data_path, local_download_path, use_gpu, target_size, batch_size, epochs

