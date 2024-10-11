from pathlib import Path

def get_config():
    return{
        'batch_size': 8,
        'num_epochs': 20,
        'lr': 1e-4,
        'seq_len': 350,
        'lang_src': 'en',
        'lang_tgt': 'it',
        'model_folder': 'weights',
        'model_filename': 'tmodel_',
        'preload': None,
        'tokenizer_file' : 'tokenizer-{0}.json',
        'experiment_name': 'runs/tmodel',
    }
    
def get_weights_file_path(config, epoch):
    model_folder = config['model_folder']
    model_basename = config['model_basname']
    model_filename = f"{['model_filename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)