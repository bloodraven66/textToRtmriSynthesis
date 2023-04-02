import os
from common.utils import get_files
from common.logger import logger
import scipy.io
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import librosa
from multiprocessing import Pool
from tqdm import tqdm
from librosa.filters import mel as librosa_mel_fn
from torchvision import transforms

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode):
        manifest_folder = self.get_filelist(config)
        manifest_file = os.path.join(manifest_folder, mode + '.txt')
        with open(manifest_file, 'r') as f:
            lines = f.read().split('\n')[:-1]
        if config.data.apply_filter:
            lines = [l for l in lines if config.data.filter in l]
        self.files = lines
        self.transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize(64),
                            transforms.ToTensor()])

    def get_filelist(self, config):
        train_set = sorted(config.data.train_subjects)
        test_set = sorted(config.data.test_subjects)
        assert len(test_set) != 0
        manifest_folder = '_'.join(train_set) + '_train_' + '_'.join(test_set) + '_test'
        manifest_folder = os.path.join(config.data.manifest_loc, manifest_folder)
        if not os.path.exists(manifest_folder):
            os.makedirs(manifest_folder)

        
        keys = [
                'unseen_sent_seen_spk_test',
                'unseen_sent_unseen_spk_test',
                'seen_sent_unseen_spk_test',
                'train',
                'val'
            ]
        filelists = {l:[] for l in keys}
        path_not_found = False
        for key in filelists:
            if not os.path.exists(os.path.join(manifest_folder, key+'.txt')):
                path_not_found = True
        if not path_not_found: return manifest_folder
        subjects = os.listdir(config.data.data_path)
        
        for s in subjects:
            ids = os.listdir(os.path.join(config.data.data_path, s, config.data.phon_folder))
            ids = [l.split('_')[1].split('.')[0] for l in ids]
            
            
            if config.data.dump_feats:
                all_data = {}
                for l in tqdm(ids):
                    all_data['label'] = os.path.join(config.data.data_path, s, config.data.phon_folder, s+'_'+l+'.npy')
                    all_data['label_id'] = os.path.join(config.data.data_path, s, config.data.phonid_folder, s+'_'+l+'.npy')
                    all_data['labelseq'] = os.path.join(config.data.data_path, s, config.data.phonseq_folder, s+'_'+l+'.npy')
                    all_data['labelseq_id'] = os.path.join(config.data.data_path, s, config.data.phonseqid_folder, s+'_'+l+'.npy')
                    all_data['dur'] = os.path.join(config.data.data_path, s, config.data.dur_folder, s+'_'+l+'.npy')
                    all_data['images'] = os.path.join(config.data.data_path, s, config.data.video_folder, s+'_'+l+'.npy')
                    all_data['audio'] = os.path.join(config.data.data_path, s, config.data.audio_folder, s+'_'+l+'.npy')
                    res = {}
                    for key in all_data:
                        with open(all_data[key], 'rb') as f:
                            data = np.load(f)
                        if key in ['label', 'labelseq', 'audio']:
                            res[key] = data
                        else:
                            res[key] = torch.from_numpy(data)
                    path = os.path.join(config.data.feats, s)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    path = os.path.join(path, l+'.pt')
                    torch.save(res, path)
            for l in tqdm(ids):
                path = os.path.join(config.data.feats, s, l+'.pt')
                if s in train_set:
                    if l[-1] == '0': filelists['unseen_sent_seen_spk_test'].append(path)
                    elif l[-1] == '1': filelists['val'].append(path)
                    else: filelists['train'].append(path)
                elif s in test_set:
                    if l[-1] == '0': filelists['unseen_sent_unseen_spk_test'].append(path)
                    elif l[-1] == '1': pass
                    else: filelists['seen_sent_unseen_spk_test'].append(path)
        for key in filelists:
            with open(os.path.join(manifest_folder, key+'.txt'), 'w') as f:
                for l in filelists[key]:
                    f.write(l+'\n')
        return manifest_folder

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        data = torch.load(self.files[i])
        vid = torch.stack([self.transform(img.numpy()) for img in data["images"]])
        l =  data["labelseq_id"]
        ll = data["label"]
        d = data["dur"]
        return vid, l, d, self.files[i]
       

def get_file_paths(main_path, subjects, process_mode, folder, extension):
    top_paths = [Path(os.path.join(main_path, sub, process_mode, folder)).expanduser().resolve() for sub in subjects]
    paths = [list(path.rglob(f'*{extension}')) for path in top_paths]
    paths = [j for sub in paths for j in sub]
    return paths


   