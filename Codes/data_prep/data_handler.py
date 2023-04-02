from data_prep.load_data import SeqDataset
from data_prep.load_data_gen import GenDataset
from data_prep.load_data_segnet import SegnetDataset
from torch.utils.data import DataLoader


def collect(cfg):
    loaders = []
    for mode in ['train', 'val'] + list(cfg.data.test_set):
        if cfg.select_model.name in ['vae', 'cvae']:
            dataset = GenDataset(cfg, mode)   #pytorch dataloader which returns individual frames
            batch_size = int(cfg.generative_config.batch_size)
        elif cfg.select_model.name in ['segnet']:
            if mode not in ['train', 'val']: continue
            dataset = SegnetDataset(cfg, mode) #pytorch dataloader for images + atb boundaries 
            print(mode, len(dataset))
            batch_size = int(cfg.generative_config.batch_size)
        else:
            dataset = SeqDataset(cfg, mode) #pytorch dataloader for full sequence of phonemes + rtMRI video frames
            
            batch_size = int(cfg.common.batch_size)
        loader_ = DataLoader(   
                            dataset, 
                            shuffle=True if mode == 'train' else False, 
                            batch_size=batch_size, 
                            )
        loaders.append(loader_)
       
    return loaders
