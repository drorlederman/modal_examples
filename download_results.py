import os
def download_results(model_path='', local_download_path=''):
    # Download the plot to the local folder
    plot_path = os.path.join(model_path, "training_plot.png")
    local_plot_path = os.path.join(local_download_path, "training_plot.png")
    os.system(f"modal volume get cats_vs_dogs {plot_path} {local_plot_path}")
    print(f"Plot downloaded to local folder: {os.path.abspath(local_plot_path)}")
