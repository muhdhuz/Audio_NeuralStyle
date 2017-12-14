An implementation of Neural Style Transfer for Audio using Pytorch.
============================

This is an implementation of neural style transfer using the alogrithm developed in
[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge.

However instead of learning style from images we use the spectrogram in place to carry out the procedure on audio.

General implementation is based off the [Pytorch tutorial on Neural Transfer](http://pytorch.org/tutorials/advanced/neural_style_tutorial.html) by Alexis Jacq. Also inspired by [Audio texture synthesis](https://github.com/DmitryUlyanov/neural-style-audio-torch) by Dmitry Ulyanov.

**To use**:  
1. Open neural_style_audio_pretrained.ipynb jupyter notebook. If required constant-Q scaling, use  	neural_style_audio_cqt.ipynb instead  
2. Some important parameters you may want to change:  
..* filename parameters  
..* librosa Fourier parameters
..* model_path: input filepath if loading a pretrained model (.pth file), else input None to initialize a model with random weights  
..* param_pkl: to load existing model paramaters from file. Alternatively leave blank and input model parameters (eg. layer size, kernel size etc.) manually in the CNN function (loaded from audiocnn.py)  
..* use01scale: True or False. Whether to scale images to [0,1] prior to input into network  
..* boundopt: True or False. Whether to regularize optimization result to [0,1]  
..* whichChannel: "freq","time" or "2D". Freq and time as channels will use a 1D orientation  
3. Loading the model:  
..* Look for the cell headed by "Here we create a custom network"  
..* To use the supplied 3-layer CNN (audiocnn.py), just initialize it with the desired network parameters (either from a pickle file or manually in the function)  
..* If you want to design your own network, remove or add layers to style_net and initialize that instead  
4. Choose which layers to grab the statistics from:  
..* content_layers_default: for content statistics  
..* style_layers_default: for style statistics  
..* layers names followed by underscore and the layer number eg. relu_1, batchnorm_2, conv_3, pool_1  
5. Noise image initialization:  
..* The noise image you optimize over can be initialzed with different options  
..* Look for the cell headed by """image to input in generative network"""  
..* samples: truncated normal distribution (lower,upper,mu,sigma can be defined)  
..* rand: uniform distribution ranging from 0 to 0.1 (most texture synthesis experiments use this)  
..* randn: normal distribution centred on 0  
..* cont_img.clone: input the content image, normally used in style transfer  
..* select the above options by comment/uncommenting the code  
6. Running the optimization:
..* Tweak the style_weight, content_weight and reg_weights as desired  
..* style_weight: for experiments 1e9 used for texture synthesis, 1e7 used for style transfer  
..* content_weight: usually left at 1, vary style_weight to change their relative strengths instead   
..* reg_weight: if boundopt=True, the regularizer to bound the optimization result to 0,1 is weighted by this number. Usually left at 1e-3 for experiments. 


