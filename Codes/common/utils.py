from attrdict import AttrDict
import yaml
import torch
import numpy as np
from trainer import basic_trainer, vae_trainer, cvae_trainer, segnet_trainer
from models import fastspeech_with_resnet, vae, cvae, segnet
import librosa
from pathlib import Path

def read_yaml(yamlFile):
    with open(yamlFile) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        cfg = AttrDict(config)
    return cfg

def t_(dataset):
    return torch.from_numpy(np.array(dataset))


def get_trainer(config):
    if config.select_model.name == 'fastspeech_with_resnet':
        return basic_trainer
    elif config.select_model.name == 'vae':
        return vae_trainer
    elif config.select_model.name == 'cvae':
        return cvae_trainer
    elif config.select_model.name == 'segnet':
        return segnet_trainer
    else: 
        raise NotImplementedError


def get_model(config):
    modelChoice = config.select_model.name
    subs = 1
    assert config.select_model.conv in ['2d', '3d']
    if modelChoice == 'fastspeech_with_resnet':
        
        model_config = read_yaml('config/fs.yaml')
        model = fastspeech_with_resnet.FastSpeech(
                                        use_conv_on_gen=config.select_model.use_conv_on_gen,
                                        use_gen=config.select_model.use_vae,
                                        decoder_conv=config.select_model.conv,
                                        concat_and_reduce=config.select_model.concat_and_reduce,
                                        use_spk_embed=False,
                                        n_symbols=42,
                                        padding_idx=0,
                                        image_size=config.common.image_size,
                                        **model_config).to(config.common.device)
        if config.select_model.use_vae:
            model_config = read_yaml('config/vae.yaml')
            model2 = cvae.VAE(**model_config).to(config.common.device)
            model = [model, model2]
    elif modelChoice == 'vae':
        model_config = read_yaml('config/vae.yaml')
        model = vae.VAE(**model_config).to(config.common.device)
    elif modelChoice == 'cvae':
        model_config = read_yaml('config/vae.yaml')
        model = cvae.VAE(**model_config).to(config.common.device)
    elif modelChoice == 'segnet':
        model = segnet.SegNet().to(config.common.device)
    else:
        raise Exception('model Not found')


    return model


def get_files(path, extension='.wav'):
    if isinstance(path, str): path = Path(path).expanduser().resolve()
    return list(path.rglob(f'*{extension}'))

