#
# import yaml
# import modal
#
# # CONFIG_PATH = "/config/modal2_config.yaml"  # Path inside Modal
# # volume = modal.Volume.from_name("cats_vs_dogs")
# #
# # app = modal.App(name="cnn-dog-cat-classifier")
# # # Define Modal image with PyTorch and dependencies
# # image = modal.Image.debian_slim(python_version="3.10").pip_install(
# #     "numpy",
# #     "matplotlib",
# #     "torch",
# #     "torchvision",
# #     "wandb",
# #     "pillow",
# #     "pyyaml"
# # )
# #
# # @app.function(
# #     volumes={CONFIG_PATH: volume},
# #     image=image,
# #     gpu="any",
# #     secrets=[modal.Secret.from_name("wandb_api_key")]# Set valid GPU configuration
# # )
# #
# # def read_config(config_path):
# #     # 1. Load the YAML file
# #     #file_name = os.path.join(data_path, './modal2_config.yaml')
# #     #file_name = 'modal2_config.yaml'
# #     with open(config_path, 'r') as f:
# #         config = yaml.safe_load(f)
# #
# #     # 2. Access values
# #     db = config['folders']
# #     data_path = db['data_path']
# #     model_path = db['model_path']
# #     local_download_path = db['local_download_path']
# #
# #     settings = config['settings']
# #     use_gpu = settings['use_gpu']
# #     print(model_path, data_path, local_download_path, use_gpu)
# #     return model_path, data_path, local_download_path, use_gpu
#
# import modal
# import wandb  # Weights and Biases for tracking
# import os
# import yaml
#
# # ✅ Define Modal App at the beginning
# app = modal.App(name="cnn-dog-cat-classifier")
#
# # ✅ Ensure W&B API key is set
# wandb.login(key=os.getenv("WANDB_API_KEY"))
#
# # Define Modal image with PyTorch and dependencies
# image = modal.Image.debian_slim(python_version="3.10").pip_install(
#     "numpy",
#     "matplotlib",
#     "torch",
#     "torchvision",
#     "wandb",
#     "pillow",
#     "pyyaml"
# )
#
# # Define Modal Volumes
# volume = modal.Volume.from_name("cats_vs_dogs")
# CONFIG_PATH = "/config/modal2_config.yaml"  # Path inside Modal
# MODEL_PATH = "/data"
#
# # Mount the volume at a single base directory
# VOLUME_MOUNT_PATH = "/data"  # Base path for the mounted volume
# CONFIG_FILE_PATH = os.path.join(VOLUME_MOUNT_PATH, "modal2_config.yaml")  # Path to config inside volume
#
#
# @app.function(
#     volumes={VOLUME_MOUNT_PATH: volume},  # ✅ Mount volume at a single location
#     image=image
# )
# def read_config():
#     """ Read and parse the config YAML file. """
#
#     # ✅ Use the correct path inside the mounted volume
#     with open(CONFIG_FILE_PATH, 'r') as f:
#         config = yaml.safe_load(f)
#
#     # Extract settings
#     db = config['folders']
#     data_path = db['data_path']
#     model_path = db['model_path']
#     local_download_path = db['local_download_path']
#     settings = config['settings']
#     use_gpu = settings['use_gpu']
#
#     print(model_path, data_path, local_download_path, use_gpu)
#     return model_path, data_path, local_download_path, use_gpu
