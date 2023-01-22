# text_to_rtmri_synthesis
This is the code base for ICASSP 2023 submission

All unseen sentence generated videos are shared here, you can check the corresponding files from real_samples_rtmri (for the actual rtMRI data), baseline_samples_rtmri (with the baseline transformer-CNN seq-seq model) and cvae_samples_rtmri (with the transformer-CNN seq-seq model with intermediatary features by sampling from a CVAE model).

Two particular examples are shown outside these folders, for F1 and M1 models for utterance 010. You can observe that for F1, the sample quaility is close for both baseline and cvae models compared to the real sample. On the otherhand, for M1, baseline model fails spectacularly when CVAE is able to generate realistic video due to the conditioning from the variational autoencoder features.
