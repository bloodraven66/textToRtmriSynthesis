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
from models import segnet

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
                        'gen_model'+f'{self.gen_model}',
                        params.common.chk_postfix + '.pth'])
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
        if self.gen_model: self.gen_model.eval()
        else: raise NotImplementedError
        self.reset_iter_metrics()
        losses_to_upload = {'mel':[], 'dur':[], 'total':[]}
        with tqdm(loader, unit="batch") as tepoch:
            for counter, data in enumerate(tepoch):
                imgs, rep_phon, dur = data
                
                imgs = imgs.squeeze()
                if imgs.shape[0] != rep_phon.shape[1]: logger.info('mismatch, skip!');continue
                imgs, rep_phon = self.set_device([imgs, rep_phon], ignoreList=[])
                if self.gen_model:
                    samples = self.gen_model.sample(len(imgs), rep_phon.device, y=rep_phon.squeeze())
                pred_imgs = self.model(rep_phon.long(), samples, use_gen=self.gen_model)
                loss_dict = self.model.basic_loss(imgs, pred_imgs)
                if mode == 'train':
                    self.optimizer.zero_grad()
                    loss_dict["total"].backward()
                    self.optimizer.step()
                self.handle_metrics(loss_dict, mode)
                tepoch.set_postfix(loss=loss_dict["total"].item())
                if break_run:
                    break   
                # break
        self.end_of_epoch(mode, pred_imgs ,imgs)
        return
    
    def test(self, loader):
        self.model.eval()
        chk = os.path.join('saved_models', f'{self.config.data.filter}_segnet_.pth')
        model_segnet = segnet.SegNet().to(self.config.common.device)
        model_segnet.load_state_dict(torch.load(chk)['model_state_dict'])
        model_segnet.eval()
        model_segnet = model_segnet #.cpu()
        softmax = torch.nn.Softmax(1)
        mask_wise_score = []
        with tqdm(loader, unit="batch") as tepoch:
            for counter, data in enumerate(tepoch):
                imgs, rep_phon, dur, filename = data
                filename = filename[0]
                save_name = filename.split('/')[-2] + '_' + Path(filename).stem + '.mp4'
                gnd_path = os.path.join('real_samples', save_name)
                save_name = os.path.join(self.config.common.plots_folder, save_name)
                imgs = imgs.squeeze()
                def make_image(data, outputname, size=(1, 1), dpi=80):
                    fig = plt.figure()
                    fig.set_size_inches(size)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    plt.set_cmap('hot')
                    ax.imshow(data, aspect='equal')
                    plt.savefig(outputname, dpi=dpi)
                make_image(imgs[0].permute(1, 2, 0).detach().cpu().numpy(), 'one_image.png')
                exit()
                if imgs.shape[0] != rep_phon.shape[1]: logger.info('mismatch, skip!');continue
                imgs, rep_phon = self.set_device([imgs, rep_phon], ignoreList=[])
                if self.gen_model:
                    self.gen_model = self.gen_model.to(imgs.device)
                    samples = self.gen_model.sample(len(imgs), rep_phon.device, y=rep_phon.squeeze())
                    pred_imgs = self.model(rep_phon.long(), samples, use_gen=self.gen_model)
                    self.gen_model.cpu()
                    samples = samples.detach().cpu()
                else:
                    pred_imgs = self.model(rep_phon.long(), use_gen=self.gen_model)
                rep_phon = rep_phon.detach().cpu()
                
                model_segnet = model_segnet.to('cpu')
                mask1_out, mask2_out, mask3_out = model_segnet(pred_imgs.cpu()) #.cpu()

                model_segnet = model_segnet.cpu()
                mask1_real, mask2_real, mask3_real = model_segnet(imgs.cpu())
        return
        
    def end_of_epoch(self, mode, out, real):
        if mode == "val":
            if self.epoch % self.config.common.upload_freq == 0:
                logger.info('uploading_samples')
                self.logger.plot_vids(out, real, num_samples=4, plot_img=True)

        for key in self.epoch_loss_dict:
            self.epoch_loss_dict[key] = sum(self.epoch_loss_dict[key])/len(self.epoch_loss_dict[key])      
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
        trainLoader, valLoader, testLoader = loaders
        if self.config.select_model.use_vae:
            model, gen_model = model
            self.gen_model = gen_model
            self.gen_model.load_state_dict(torch.load(f'saved_models/{self.config.data.filter}_cvae_.pth', map_location='cpu')["model_state_dict"])
        self.optimizer = self.get_trainers(model)
        self.model = model
        if not self.config.common.infer:

            self.optimizer = self.get_trainers(self.model)      
            for epoch in range(int(self.config.common.epochs)):
                self.epoch = epoch
                self.trainloop(trainLoader, 'train')
                self.trainloop(valLoader, 'val')
                self.esCheck()
            logger.info('Training completed')

        else:
            logger.info('Starting Inference')
            chk = os.path.join('saved_models', self.config.data.filter+self.config.select_model.infer_chk)
            self.model.load_state_dict(torch.load(chk, map_location='cpu')['model_state_dict'])
            logger.info(f'Loaded {chk} for spk {self.config.data.filter}')
            assert self.config.data.filter in chk
            self.test(testLoader)



   
    
    def get_trainers(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.config.optimizer.lr), weight_decay=float(self.config.optimizer.weightdecay))

        return optimizer

    def set_device(self, data, ignoreList):

        if isinstance(data, list):
            return [data[i].to(self.config.common.device).float() if i not in ignoreList else data[i] for i in range(len(data))]
        else:
            raise Exception('set device for input not defined')

    


    
    

            