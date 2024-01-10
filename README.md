## Introduction
This is a experimental code adpated from offical [DDIM](https://github.com/ermongroup/ddim), whose aim at replicating the simplified
version introduced from [keras](https://github.com/keras-team/keras-io/blob/master/examples/generative/ddim.py) 
and following the experiment mentioned on [MINI DDIM](https://stackoverflow.com/questions/76590848/minimal-diffusion-model-ddim-for-mnist) of Stack Overflow.

## Motivation
My inital training using keras' offical code generated bad results which can be seen in the ISSUE I pulled at .
To find out what is going wrong, I decided to transfer the same experimental setting from keras to pytorch.
But results yied from my current works still far from the ideal one post on [keras's blog](https://keras.io/examples/generative/ddim/).
I am still working on it, trying my best to make it work as the offical did.

## Method
I converted keras's network structure into a torch one, then trained it under modified DDIM's training procedure where continuous
time step and positional embedding for noise rate using cosine schedule were added.

## Results
The following pictures are some generated images from my trained models:
BK stands for baseline_keras, i.e. model structure with same setting as keras
BBNO stands for baseline_bn_norm_out, i.e. model structure with same setting as DDIM but removing attention block and using batch
normalization without affine transformation, also, positional embedding only input once on the top layer.
BGANO stands for baseline_gn_affine_norm_out, i.e. model structure with same setting as BBNO but using learnable group normalization

generated results including middle outputs by DDIM sampling from time step 1 to 20

generated results by DDIM sampling after 50 timestep

generated results by DDIM sampling after 100 timestep

generated results by DDPM sampling after 1000 timestep

It would be greateful if anyone could help with this project, or any suggection for my potential code or experimental fault.
P.S. I also upload a experimental report from which more detail is available.
