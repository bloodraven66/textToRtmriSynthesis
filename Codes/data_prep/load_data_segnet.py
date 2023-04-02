import os
import cv2
from common.utils import get_files
from common.logger import logger
import scipy.io
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import librosa
from multiprocessing import Pool
from scipy.io import loadmat
from tqdm import tqdm
from librosa.filters import mel as librosa_mel_fn
from torchvision import transforms

class SegnetDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode):
        manifest_folder = self.get_filelist(config)
        manifest_file = os.path.join(manifest_folder, mode + '.txt')
        with open(manifest_file, 'r') as f:
            lines = f.read().split('\n')[:-1]
        if config.data.apply_filter:
            lines = [l for l in lines if config.data.filter in l]
        self.files = lines
        print(len(self.files))
        self.transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize(64),
                            transforms.ToTensor()])
        

    def get_filelist(self, config):
        train_set = sorted(config.data.train_subjects)
        test_set = sorted(config.data.test_subjects)
        assert len(test_set) != 0
        manifest_folder = '_'.join(train_set) + '_train_' + '_'.join(test_set) + '_test' + '_segnet'
        manifest_folder = os.path.join(config.data.manifest_loc, manifest_folder)
        if not os.path.exists(manifest_folder):
            os.makedirs(manifest_folder)
        keys = [
                'train',
                'val'
            ]
        filelists = {l:[] for l in keys}
       
        subjects = os.listdir(config.segnet_config.video_path)
        if config.segnet_config.dump_feats:
            for s in tqdm(subjects):
                video_folder = os.listdir(os.path.join(config.segnet_config.video_path, s))
                mat_folder = os.path.join(config.segnet_config.mask_path, s)
                save_path = os.path.join(config.segnet_config.save_path, s)
                if not os.path.exists(save_path): os.makedirs(save_path)
                for filename in video_folder:
                    filename = os.path.join(os.path.join(config.segnet_config.video_path, s), filename)
                    file_id = Path(filename).stem
                    mat_file = os.path.join(mat_folder, file_id.split('_')[-1]+'.mat')
                    assert os.path.exists(mat_file), f'{mat_file}'
                    matfile_data = loadmat(mat_file)['masks'][0]
                    vidcap = cv2.VideoCapture(filename)
                    success,image = vidcap.read()
                    
                    count = 0
                    
                    while success:
                        path = os.path.join(save_path, f'{file_id}_{count}.npy')
                        with open(path, 'wb') as f:
                            np.save(f, [image, matfile_data[count]])
                        success,image = vidcap.read()
                        count += 1
        for sub in os.listdir(config.segnet_config.video_path):
        
            utts = sorted(os.listdir(os.path.join(config.segnet_config.video_path, sub)))
            train_utts = set([Path(l).stem.split('_')[-1] for l in utts[:9]])
            eval_utts = set([Path(l).stem.split('_')[-1] for l in utts[9:]])
            saved_data = [os.path.join(config.segnet_config.save_path, sub, l) for l in sorted(os.listdir(os.path.join(config.segnet_config.save_path, sub)))]
            filelists['train'].extend([l for l in saved_data if Path(l).stem.split('_')[1] in train_utts])
            filelists['val'].extend([l for l in saved_data if Path(l).stem.split('_')[1] in eval_utts])
        print(manifest_folder)
        for key in filelists:
            with open(os.path.join(manifest_folder, key+'.txt'), 'w') as f:
                for l in filelists[key]:
                    f.write(l+'\n')
        return manifest_folder

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with open(self.files[i], 'rb') as f:
            data, (mask1, mask2, mask3) = np.load(f, allow_pickle=True)
        return self.transform(data), (self.transform(mask1), self.transform(mask2), self.transform(mask3))
        
       

def get_file_paths(main_path, subjects, process_mode, folder, extension):
    top_paths = [Path(os.path.join(main_path, sub, process_mode, folder)).expanduser().resolve() for sub in subjects]
    paths = [list(path.rglob(f'*{extension}')) for path in top_paths]
    paths = [j for sub in paths for j in sub]
    return paths


   