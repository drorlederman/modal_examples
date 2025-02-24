
def read_config():
    # 1. Load the YAML file
    #file_name = os.path.join(data_path, './modal2_config.yaml')
    #file_name = 'modal2_config.yaml'
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Access values
    db = config['folders']
    data_path = db['data_path']
    model_path = db['model_path']
    local_download_path = db['local_download_path']

    settings = config['settings']
    use_gpu = settings['use_gpu']
    print(model_path, data_path, local_download_path, use_gpu)
    return model_path, data_path, local_download_path, use_gpu