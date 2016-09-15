# WaveNet
Implementation of WaveNet: A Generative Model for Raw Audio  

![figure](https://storage.googleapis.com/deepmind-live-cms.google.com.a.appspot.com/documents/BlogPost-Fig2-Anim-160908-r01.gif)

Implementation includes:
- Causal convolutional layers (implemented by masking)
- Dilated (à trous) convolutional layer blocks
- 256-class softmax 
- Sample generation
- Downsampling from 48kHz to 24kHz
- Conversion of bitrate from 16 bit to 8 bit via μ-law algorithm
- Gated convolution (tanh * sigmoid) [TODO]
- Conditional distribution (speaker, text) [TODO]
- Context stacks [TODO]
- Tested on VCTK (Yamagishi, 2012) data set [In-Progress]
- Testing on music datasets [TODO]

Blog  
https://deepmind.com/blog/wavenet-generative-model-raw-audio/

Paper  
https://arxiv.org/pdf/1609.03499.pdf

Parts adapted from   
https://github.com/igul222/pixel_rnn and  
https://github.com/kundan2510/pixelCNN
