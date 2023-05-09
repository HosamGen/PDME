# PDME
## Prompt-Driven Masked Editing for Images

Project for AI702: Deep Learning at MBZUAI !

To set up this repository, please first install the required libraries listed in "requirements.txt".
The system used in development is Ubuntu 22.04 with CUDA Toolkit 11.7

Then, download the required model weights from the following link:
https://mbzuaiac-my.sharepoint.com/:u:/g/personal/kane_lindsay_mbzuai_ac_ae/EcIJJlGPIPpIhPA9KH7EbooB--JFof5wLPMbbWlp_2wLAw?e=NQ8JHe 

Finally, place the downloaded files in the following folders to set up the project: 

In "/checkpoints/"
* 256\*256_diffusion_uncond.pt

In "/LAVT-RIS/checkpoints/":
* gref_google.pth
* gref_umd.pth
* refcoco.pth
* refcoco+.pth
* swin_base_patch4_window12_384_22k.pth

In "/LAVT_RIS/pretrained_weights/"
* lavt_one_8_cards_ImgNet22KPre_swin-base-window12_refcocogGOOG_adamw_b32lr0.00005wd1e-2_E40.pth

In "/LAVT_RIS/refer/data" (optional)
* images
* refclef
* refcoco
* refcoco+
* refcocog

PDME is best run using command line arguments. The file 'run.sh' is provided for convenience with some preset arguments as an example.
You can provide paths to your own images, target prompts, and edit prompts to produce edits from your own images.
