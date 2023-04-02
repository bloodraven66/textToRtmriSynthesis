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
        self.numEpochs = int(params.common.epochs)
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
        
        self.gen_model = False
        if params.select_model.use_vae:
            self.gen_model = True
            logger.info('Using generative prior')
        self.chk_name = '_'.join([self.subs,
                        params.select_model.name,
                        params.common.chk_postfix + '.pth'])
        logger.info(f'using subs {self.subs}, chk={self.chk_name}')
        self.best_loss = 1000

    def esCheck(self):
        if self.current_loss < self.best_loss:
            self.saveCheckpoint()
            self.best_loss = self.current_loss
            logger.info('Updated weights')

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
        softmax = torch.nn.Softmax(1)
        criterion = nn.CrossEntropyLoss(reduction='sum')
        with tqdm(loader, unit="batch") as tepoch:
            for counter, data in enumerate(tepoch):
                imgs, (mask1, mask2, mask3) = data
                
                imgs = imgs.squeeze()
                if len(imgs.shape) == 3: 
                    continue
                imgs, mask1, mask2, mask3 = self.set_device([imgs, mask1, mask2, mask3], ignoreList=[])
                mask1 = (mask1 > 0).int()
                mask2 = (mask2 > 0).int()
                mask3 = (mask3 > 0).int()
                # print(torch.unique(mask1, return_counts=True))
                loss_dict = {}
                mask1_out, mask2_out, mask3_out = self.model(imgs)
                loss_dict['mask1_loss'] = criterion(mask1_out, mask1.squeeze().long()) / (len(mask1_out))
                loss_dict['mask2_loss'] = criterion(mask2_out, mask2.squeeze().long()) / (len(mask1_out))
                loss_dict['mask3_loss'] = criterion(mask3_out, mask3.squeeze().long()) / (len(mask1_out))
                loss_dict['total'] = loss_dict['mask1_loss'] + loss_dict['mask2_loss'] + loss_dict['mask3_loss']
                
                if mode == 'train':
                    self.optimizer.zero_grad()
                    loss_dict["total"].backward()
                    self.optimizer.step()
                self.handle_metrics(loss_dict, mode)
                tepoch.set_postfix(loss=loss_dict["total"].item())
                if break_run:
                    break   
                # break
        mask1_out = torch.argmax(softmax(mask1_out), 1)[0].squeeze().detach().cpu().numpy()
        print(np.sum(mask1_out))
        mask2_out = torch.argmax(softmax(mask2_out), 1)[0].squeeze().detach().cpu().numpy()
        mask3_out = torch.argmax(softmax(mask3_out), 1)[0].squeeze().detach().cpu().numpy()
        mask1 = mask1[0].squeeze().detach().cpu().numpy()
        mask2 = mask2[0].squeeze().detach().cpu().numpy()
        mask3 = mask3[0].squeeze().detach().cpu().numpy()
        self.end_of_epoch(mode,( mask1_out, mask2_out, mask3_out), (mask1, mask2, mask3))

        return
    
    def test(self, loader):
        self.model.eval()
        criterion = nn.CrossEntropyLoss(reduction='sum')
        softmax = torch.nn.Softmax(1)
        with tqdm(loader, unit="batch") as tepoch:
            for counter, data in enumerate(tepoch):
                imgs, (mask1, mask2, mask3) = data
                
                imgs = imgs.squeeze()
                imgs, mask1, mask2, mask3 = self.set_device([imgs, mask1, mask2, mask3], ignoreList=[])
                loss_dict = {}
                

                mask1_out, mask2_out, mask3_out = self.model(imgs)
                loss_dict['mask1_loss'] = criterion(mask1_out, mask1.squeeze().long()) / (len(mask1_out))
                loss_dict['mask2_loss'] = criterion(mask2_out, mask2.squeeze().long()) / (len(mask1_out))
                loss_dict['mask3_loss'] = criterion(mask3_out, mask3.squeeze().long()) / (len(mask1_out))
                loss_dict['total'] = loss_dict['mask1_loss'] + loss_dict['mask2_loss'] + loss_dict['mask3_loss']
                print(mask1_out.shape, loss_dict['total'])
                mask1_out = torch.argmax(softmax(mask1_out), 1)
                mask2_out = torch.argmax(softmax(mask2_out), 1)
                mask3_out = torch.argmax(softmax(mask3_out), 1)
                
                plt.imshow(mask1_out[0].squeeze().detach().cpu().numpy()+mask2_out[0].squeeze().detach().cpu().numpy()+mask3_out[2].squeeze().detach().cpu().numpy())
                plt.savefig('mask_out')
                plt.imshow(mask1[0].squeeze().detach().cpu().numpy()+mask2[0].squeeze().detach().cpu().numpy()+mask3[0].squeeze().detach().cpu().numpy())
                plt.savefig('mask_in')
                exit()


                
    def end_of_epoch(self, mode, out, real):
        if mode == "val":
            if self.epoch % self.config.common.upload_freq == 0:
                logger.info('uploading_samples')
                self.logger.plot_segnet(out, real)

        for key in self.epoch_loss_dict:
            self.epoch_loss_dict[key] = sum(self.epoch_loss_dict[key])/len(self.epoch_loss_dict[key])      
            if mode == "val":
                self.current_loss = self.epoch_loss_dict['total_val']
        self.logger.log(self.epoch_loss_dict)
        
    def reset_iter_metrics(self):
        self.epoch_loss_dict = {}
        self.skipped = 0


    def handle_metrics(self, iter_loss_dict, mode):
        for key in iter_loss_dict:
            if f'{key}_{mode}' not in self.epoch_loss_dict:
                self.epoch_loss_dict[f'{key}_{mode}'] = [iter_loss_dict[key].item()]
            else:
                self.epoch_loss_dict[f'{key}_{mode}'].append(iter_loss_dict[key].item())


    def trainer(self, model, loaders):
        trainLoader, valLoader = loaders
        
        self.optimizer = self.get_trainers(model)
        self.model = model
        if not self.config.common.infer:
            if self.config.segnet_config.load_chk:
                chk = os.path.join('saved_models', 'pooled_segnet.pth')
                self.model.load_state_dict(torch.load(chk)['model_state_dict'])
            self.optimizer = self.get_trainers(self.model)      
            for epoch in range(int(self.config.common.epochs)):
                self.epoch = epoch
                self.trainloop(trainLoader, 'train')
                self.trainloop(valLoader, 'val')
                self.esCheck()
            logger.info('Training completed')

        else:
            logger.info('Starting Inference')
            chk = os.path.join('saved_models', self.config.data.filter+'_'+self.config.select_model.name+'_.pth')
            self.model.load_state_dict(torch.load(chk)['model_state_dict'])
            logger.info(f'Loaded {chk} for spk {self.config.data.filter}')
            assert self.config.data.filter in chk
            self.test(valLoader)
            # self.trainloop(valLoader, 'val', break_run=True)



   
    
    def get_trainers(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.config.optimizer.lr), weight_decay=float(self.config.optimizer.weightdecay))

        return optimizer

    def set_device(self, data, ignoreList):

        if isinstance(data, list):
            return [data[i].to(self.config.common.device).float() if i not in ignoreList else data[i] for i in range(len(data))]
        else:
            raise Exception('set device for input not defined')

    


    
    

            