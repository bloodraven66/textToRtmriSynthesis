import os
import yaml
import torch
import numpy as np
from attrdict import AttrDict
import fastspeech_with_resnet

def read_yaml(yamlFile):
    with open(yamlFile) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        cfg = AttrDict(config)
    return cfg

def plot(samples, path):
    samples = samples.permute(0, 2, 3, 1).detach().cpu().numpy()
    animate(samples, path)

def animate(data, filename):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    fps = 23
    snapshots = data
    fig = plt.figure( figsize=(3,3) )
    a = snapshots[0]
    im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)

    def animate_func(i):
        im.set_array(snapshots[i])
        return [im]

    anim = animation.FuncAnimation(
                                fig, 
                                animate_func, 
                                frames = len(data),
                                interval = 1000 / fps,
                                )
    writergif = animation.FFMpegWriter(fps=fps)
    anim.save(filename, writer=writergif)
        
def load_model(config, modelname):
    model_config = read_yaml('fs.yaml')
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
    model.load_state_dict(torch.load(modelname, map_location='cpu')['model_state_dict'])
    return model

def run(text, path, modelname):
    config = read_yaml('hparams.yaml')
    model = load_model(config, modelname)
    pred_imgs = model(text)
    plot(pred_imgs, path)

text = ['sil', 'sil', 'dh', 'dh', 'dh', 's', 's', 's', 'ah', 'ah', 'ah']
mapping = {'sil': 41, 'dh': 27, 'ih': 5, 's': 15, 'w': 29, 'ah': 2, 'z': 23, 'iy': 24, 'f': 26, 'er': 19, 'jh': 22, 'ey': 21, 'n': 9, 'm': 30, 'ao': 38, 'r': 8, 'b': 31, 'ay': 39, 'k': 14, 'ng': 3, 'hh': 35, 'aa': 28, 'd': 7, 'sh': 10, 'th': 33, 'ae': 25, 't': 20, 'ow': 17, 'eh': 34, 'v': 12, 'y': 1, 'l': 6, 'uw': 32, 'uh': 11, 'g': 37, 'aw': 36, 'oy': 13, 'p': 16, 'ch': 4, 'zh': 18, 'oov': 41}
text = torch.tensor([mapping[t] for t in text]).unsqueeze(0)
run(text, 'demo.mp4', modelname='path_to_pretrained_model')
