import os
import torch
import librosa
import numpy as np
import scipy.stats
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
from common.logger import logger
import torch.nn.functional as F
import matplotlib.pyplot as plt
from common.wandb_logger import WandbLogger
import librosa

os.environ["WANDB_SILENT"] = "true"
import common


class Operate():
    def __init__(self, params):
        self.numEpochs = int(params.generative_config.epochs)
        self.modelName = params.select_model.name
        self.config = params
        if params.logging.disable:
            logger.info('wandb logging disabled')
            os.environ['WANDB_MODE'] = 'offline'
        self.logger = WandbLogger(params)
        logger.info(f'Predicting {self.modelName}')
        self.subs = list(params.data.train_subjects)
        if params.data.apply_filter:
             self.subs = [params.data.filter]
        self.subs = '_'.join(self.subs)
        self.chk_name = self.subs + '_' + params.select_model.name + '_' + '.pth'

        logger.info(f'using subs {self.subs}, chk={self.chk_name}')

    def esCheck(self):
        self.saveCheckpoint()
        
    def saveCheckpoint(self):
        save_path = os.path.join('saved_models', self.chk_name)
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, save_path)
    
    
    def trainloop(self, loader, mode, break_run=False):
        if mode == 'train': self.model.train()
        elif mode == 'val': self.model.eval()
        else: raise NotImplementedError
        self.reset_iter_metrics()
        losses_to_upload     = {'loss':[], 're':[], 'kl':[]}
        with tqdm(loader, unit="batch") as tepoch:
            for counter, (data, text) in enumerate(tepoch):
                imgs = data
                # text = text.unsqueeze(-1)
                # imgs = imgs.squeeze()
                imgs, text = self.set_device([imgs, text], ignoreList=[])
                # print(rep_phon)
                pred_imgs = self.model(imgs, labels=text)
                train_loss = self.model.loss_function(*pred_imgs, M_N = 1.0)
                loss = train_loss[0]
                if mode == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                loss_dict = {'total':loss.item(), 'recon':train_loss[1].item(), 'kld_loss':train_loss[-1].item()}
                self.handle_metrics(loss_dict, mode)
                tepoch.set_postfix(loss=loss_dict["total"])
                if break_run:
                    break   
                # break
                # break
        self.end_of_epoch(mode, pred_imgs[0].permute(0, 2, 3, 1).detach().cpu().numpy() ,imgs.permute(0, 2, 3, 1).detach().cpu().numpy())
        return
    
    
        
    def end_of_epoch(self, mode, out, real):
        if mode == "val":
            if self.epoch % self.config.generative_config.upload_freq == 0:
                logger.info('uploading_samples')
                self.logger.plot_imgs(out, real, num_samples=4)

        for key in self.epoch_loss_dict:
            self.epoch_loss_dict[key] = sum(self.epoch_loss_dict[key])/len(self.epoch_loss_dict[key])      
        self.logger.log(self.epoch_loss_dict)
        
    def reset_iter_metrics(self):
        self.epoch_loss_dict = {}
        self.skipped = 0


    def handle_metrics(self, iter_loss_dict, mode):
        for key in iter_loss_dict:
            if f'{key}_{mode}' not in self.epoch_loss_dict:
                self.epoch_loss_dict[f'{key}_{mode}'] = [iter_loss_dict[key]]
            else:
                self.epoch_loss_dict[f'{key}_{mode}'].append(iter_loss_dict[key])


    def trainer(self, model, loaders):
        trainLoader, valLoader, _ = loaders
        self.optimizer = self.get_trainers(model)
        self.model = model
        logger.info(f'loaders length {len(trainLoader)}, {len(valLoader)}')
        if not self.config.common.infer:

            self.optimizer = self.get_trainers(self.model)      
            for epoch in range(int(self.config.generative_config.epochs)):
                self.epoch = epoch
                self.trainloop(trainLoader, 'train')
                self.trainloop(valLoader, 'val')
                self.esCheck()
            logger.info('Training completed')

        else:
            logger.info('Starting Inference')
            # self.model.load_state_dict(torch.load(self.config.earlystopper.checkpoint))
            # self.trainloop(valLoader, 'val', break_run=True)



   
    
    def get_trainers(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.config.optimizer.lr), weight_decay=float(self.config.optimizer.weightdecay))

        return optimizer

    def set_device(self, data, ignoreList):

        if isinstance(data, list):
            return [data[i].to(self.config.common.device).float() if i not in ignoreList else data[i] for i in range(len(data))]
        else:
            raise Exception('set device for input not defined')

    


    
    

            