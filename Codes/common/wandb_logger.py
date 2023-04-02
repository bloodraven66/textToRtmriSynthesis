import wandb
import matplotlib.pyplot as plt
import numpy as np

class WandbLogger():
    def __init__(self, config):
        self.run = wandb.init(
                            name=config.logging.run_name,
                            project=config.logging.project_name,
                            config=config,
                            notes=config.logging.notes,
                            tags=config.logging.tags)

    def log(self, dct):
        wandb.log(dct)

    def plot_vids(self, samples, gnds, num_samples=4, plot_img=False):
        samples = samples.permute(0, 2, 3, 1).detach().cpu().numpy()
        gnds = gnds.permute(0, 2, 3, 1).detach().cpu().numpy()
        sample = samples[:num_samples]
        gnd = gnds[:num_samples]
        assert num_samples % 2 == 0
        fig, ax = plt.subplots(2, num_samples//2)
        for i in range(2):
            for j in range(num_samples//2):
                ax[i][j].imshow(gnd[i*2+j])     
        plt.tight_layout()
        wandb.log({"real": wandb.Image(plt)})
        plt.clf()
        fig, ax = plt.subplots(2, num_samples//2)
        for i in range(2):
            for j in range(num_samples//2):
                ax[i][j].imshow(sample[i*2+j])     
        plt.tight_layout()
        wandb.log({"pred": wandb.Image(plt)})
        plt.clf()
        print(np.max(gnds), np.min(gnds))
        # wandb.log({"real_seq": wandb.Video(gnds.transpose(0, 3, 1, 2), fps=4, format="webm")})
        self.animate(gnds, 'wandb/samples/to_upload_gnd.mp4')
        wandb.log({"real_seq": wandb.Video('wandb/samples/to_upload_gnd.mp4', fps=23, format="mp4")})
        self.animate(samples, 'wandb/samples/to_upload_pred.mp4')
        wandb.log({"pred_seq": wandb.Video('wandb/samples/to_upload_pred.mp4', fps=23, format="mp4")})

    def plot_final_vids(self, samples, gnds, gnd_path, pred_path):
        samples = samples.permute(0, 2, 3, 1).detach().cpu().numpy()
        gnds = gnds.permute(0, 2, 3, 1).detach().cpu().numpy()
        self.animate(gnds, gnd_path)
        self.animate(samples, pred_path)
    
    def plot_segnet(self, out, gnd):
        fig, ax = plt.subplots(1, 3)
        for l in range(len(out)):
            ax[l].imshow(out[l])
            ax[l].imshow(out[l])
            ax[l].imshow(out[l])
            ax[l].set_title(f'pred_mask{l}')
        
        plt.tight_layout()
        wandb.log({f'pred_mask': wandb.Image(plt)})
        plt.clf()
        fig, ax = plt.subplots(1, 3)
        for l in range(len(gnd)):
            ax[l].imshow(gnd[l])
            ax[l].imshow(gnd[l])
            ax[l].imshow(gnd[l])
            ax[l].set_title(f'gnd_mask{l}')
        
        plt.tight_layout()
        wandb.log({'gnd_mask': wandb.Image(plt)})
        plt.clf()
        

    def plot_imgs(self, samples, gnds, num_samples=4):
        assert num_samples %2 == 0
        samples = samples[:num_samples]
        gnds = gnds[:num_samples]
        plt.clf()
        fig, ax = plt.subplots(2,num_samples//2)
        for i in range(2):
            for j in range(num_samples//2):
                ax[i][j].imshow(gnds[i*2+j])
        plt.title('gnd')
        plt.tight_layout()
        wandb.log({"gnd": wandb.Image(plt)})
        plt.clf()
        fig, ax = plt.subplots(2,num_samples//2)
        for i in range(2):
            for j in range(num_samples//2):
                ax[i][j].imshow(samples[i*2+j])
        plt.title('pred')
        plt.tight_layout()
        wandb.log({"pred": wandb.Image(plt)})
        plt.clf()
        
    def animate(self, data, filename):
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
                                    interval = 1000 / fps, # in ms
                                    )
        writergif = animation.FFMpegWriter(fps=fps)
        anim.save(filename, writer=writergif)

    def summary(self, dct):
        for key in dct:
            wandb.run.summary[key] = dct[key]


    def end_run(self):
        self.run.finish()
