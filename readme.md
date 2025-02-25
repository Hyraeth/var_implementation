Unit test in each files.

VAE : Encoder and Decoder definition
Multiscale_VAE_Quant : Multiscale features quantizer
Multiscale_VQVAE : VAE + Multiscale_VAE_Quant + Training
VAR Block : Transformer decoder blocks with class conditioning and Adaptative layer normalization
VAR : Full VAR decoder : Main contain inference
Train VAR : Training script

Launch Multiscale_VQVAE.py to train vqvae
Launch train_var.py to train VAR.

Test with var.py

I have 30mb left on the ssd so no cifar10 or imagenet training, nor huggingface weights.