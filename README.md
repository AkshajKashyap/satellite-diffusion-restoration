
Satellite Diffusion Restoration

U-Net and DDPM satellite image restoration project for synthetic cloud/noise/blur removal.

Goal

Build a deep learning restoration system that:

loads satellite imagery
creates controlled image corruptions
trains a baseline U-Net denoiser
trains a DDPM-style noise prediction model
evaluates restoration quality with PSNR, SSIM, and visual reports
exposes restoration through a lightweight API/demo
First milestone

Train a plain U-Net denoiser on corrupted EuroSAT-style image patches before building DDPM.
