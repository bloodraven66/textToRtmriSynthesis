
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import *
import math

class ResBlock2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        # print(x.shape)
        b = self.block(x)
        # print(b.shape)
        return x + b

class ResBlock3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv3d(dim, dim, 3, 1, 1),
            nn.BatchNorm3d(dim),
            nn.ReLU(True),
            nn.Conv3d(dim, dim, 1),
            nn.BatchNorm3d(dim)
        )

    def forward(self, x):
        # print(x.shape)
        b = self.block(x)
        # print(b.shape)
        return x + b

class ResBlock3D_last(nn.Module):
    def __init__(self, dim, input_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv3d(dim, dim, 3, 1, 1),
            nn.BatchNorm3d(dim),
            nn.ReLU(True),
            nn.Conv3d(dim, input_dim, 1),
        )

    def forward(self, x):
        # print(x.shape)
        b = self.block(x)
        # print(b.shape)
        return b

class FastSpeech(nn.Module):
    def __init__(self, use_conv_on_gen, use_gen, n_symbols, padding_idx, decoder_conv,
                 symbols_embedding_dim, in_fft_n_layers, in_fft_n_heads,
                 in_fft_d_head,
                 in_fft_conv1d_kernel_size, in_fft_conv1d_filter_size,
                 in_fft_output_size,
                 p_in_fft_dropout, p_in_fft_dropatt, p_in_fft_dropemb,
                 out_fft_n_layers, out_fft_n_heads, out_fft_d_head,
                 out_fft_conv1d_kernel_size, out_fft_conv1d_filter_size,
                 out_fft_output_size,
                 p_out_fft_dropout, p_out_fft_dropatt, p_out_fft_dropemb,
                 dur_predictor_kernel_size, dur_predictor_filter_size,
                 p_dur_predictor_dropout, dur_predictor_n_layers,
                 speaker_emb_weight, use_spk_embed, image_size,concat_and_reduce):
        super(FastSpeech, self).__init__()
        self.encoder = FFTransformer(
            n_layer=in_fft_n_layers, n_head=in_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=in_fft_d_head,
            d_inner=in_fft_conv1d_filter_size,
            kernel_size=in_fft_conv1d_kernel_size,
            dropout=p_in_fft_dropout,
            dropatt=p_in_fft_dropatt,
            dropemb=p_in_fft_dropemb,
            embed_input=True,
            d_embed=symbols_embedding_dim,
            n_embed=n_symbols,
            padding_idx=padding_idx)


        if use_spk_embed:
            self.speaker_emb = nn.Embedding(n_speakers, symbols_embedding_dim)
            # logger.info('Using speaker embed')
        else:
            self.speaker_emb = None
            # logger.info('No speaker embed')

        self.speaker_emb_weight = speaker_emb_weight

        self.duration_predictor = TemporalPredictor(
            in_fft_output_size,
            filter_size=dur_predictor_filter_size,
            kernel_size=dur_predictor_kernel_size,
            dropout=p_dur_predictor_dropout, n_layers=dur_predictor_n_layers
        )

        self.decoder = FFTransformer(
            n_layer=out_fft_n_layers, n_head=out_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=out_fft_d_head,
            d_inner=out_fft_conv1d_filter_size,
            kernel_size=out_fft_conv1d_kernel_size,
            dropout=p_out_fft_dropout,
            dropatt=p_out_fft_dropatt,
            dropemb=p_out_fft_dropemb,
            embed_input=False,
            d_embed=symbols_embedding_dim
        )
       
        self.proj = nn.Linear(out_fft_output_size, image_size*4*image_size, bias=True)
        self.image_size = image_size
        dim = image_size
        self.dim = int(math.sqrt(image_size*4))
        input_dim = 3
        self.decoder_conv = decoder_conv
        if decoder_conv == '2d':
            self.resnet_decoder = nn.Sequential(
                ResBlock2D(dim),
                ResBlock2D(dim),
                nn.ReLU(True),
                nn.ConvTranspose2d(dim, dim, 4, 2, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(True),
                nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
                nn.Tanh()
            )
        if decoder_conv == '3d':
            self.resnet_decoder_resblocks1 = nn.Sequential(
                ResBlock3D(dim),
                ResBlock3D(dim),)
            self.resnet_decoder_upsampling = nn.Sequential(nn.ReLU(True),
                nn.ConvTranspose2d(dim, dim, 4, 2, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(True),
                nn.ConvTranspose2d(dim, dim//2, 4, 2, 1))
            self.resnet_decoder_last = nn.Sequential(
                ResBlock3D_last(dim//2, input_dim),
               
            )
        
        if use_gen:
            if use_conv_on_gen:
                # logger.info('using gen, conv over feats')
                self.gen_con = nn.Conv2d(64,64, 3, 1, 1)
            else:
                self.gen_con = None
            
            if concat_and_reduce:
                # logger.info('using gen, concat_and_reduce')
                self.reduce = nn.Conv2d(128, 64, 1, 1)
            else:
                self.reduce = None
            
             
        
    def forward(self, inputs, img=False, use_gen=False):

        
        spk_emb = 0
        enc_out, _ = self.encoder(inputs, conditioning=spk_emb)
        dec_out, dec_mask = self.decoder(enc_out, torch.tensor([enc_out.shape[1]]).to(enc_out.device))
        feat_out = self.proj(dec_out).permute(1, 0, 2)        
        feat_out = feat_out.reshape(-1, self.image_size, self.dim ,self.dim)   
        # print(feat_out.shape, img.shape)
        
        if self.decoder_conv == '3d': 
            feat_out = feat_out.unsqueeze(0).permute(0, 2, 1, 3, 4)
            # print(feat_out.shape)
            feat_out = self.resnet_decoder_resblocks1(feat_out).squeeze().permute(1, 0, 2, 3)
            # print(feat_out.shape, img.shape)
            # exit()
            if use_gen != False: 
                if self.gen_con is not None:
                    img = self.gen_con(img)
                if self.reduce is not None:
                   
                    data = torch.cat([feat_out, img], 1)
                    # print(data.shape)
                    feat_out = self.reduce(data)
                    # print(feat_out.shape)
                else:
                    feat_out = feat_out + img
            feat_out = self.resnet_decoder_upsampling(feat_out)
            # print(feat_out.shape)
            feat_out = self.resnet_decoder_last(feat_out.unsqueeze(0).permute(0, 2, 1, 3, 4)).squeeze().permute(1, 0, 2, 3)
            # print(feat_out.shape)
        else:
            feat_out = self.resnet_decoder(feat_out)
        # exit()
        return feat_out
    
    def basic_loss(self, inputs, targets):
        loss_dict = {}
        # print(targets.shape, inputs.shape)

        loss = F.mse_loss(targets.unsqueeze(0), inputs.unsqueeze(0), reduction='sum')/(self.image_size*self.image_size*3)
        # loss = F.mse_loss(targets.unsqueeze(0), inputs.unsqueeze(0), reduction='mean')
        # print(loss)
        # exit()
        loss_dict["total"] = loss
        loss_dict["recon"] = loss
        return loss_dict
    


