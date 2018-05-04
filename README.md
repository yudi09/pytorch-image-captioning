# pytorch-image-captioning

## Abstract
In this project, I have implemented an end-to-end Deep Learning model for Image Captioning. The architecture consists of Encoder and Decoder Networks. Encoder is one of the pre-trained CNN architectures to get image embedding. Decoder is LSTM network with un-intialized word embeddings.

## Dependencies
python3, pytorch, pytorch-vision, pillow, nltk, pickle

## Dataset
[Flickr8K](http://www.jair.org/papers/paper3994.html)
train examples: 6000
dev examples: 1000
test examples: 1000

## Instructions to run the code

### 1. Pre-Processing
```bash
python3 Preprocess.py
```

### 2. Train 
```bash
python3 train.py -model <encoder_architecture> -dir <train_dir_path> -save_iter <model_checkpoint> -learning_rate <learning_rate> -epoch <re-train_epoch> -gpu_device <gpu_device_number> -hidden_dim <lstm_hidden_state_dim> -embedding_dim <encoder_output>
```
##### args:
```bash
 -model        : one of the cnn architectures - alexnet, resnet18, resnet152, vgg, inception, squeeze, dense
 -dir          : training directory path
 -save_iter    : create model checkpoint after some iterations, default = 10
 -learning_rate: default = 1e-5
 -epoch        : re-train the network from saved checkpoint epoch
 -gpu_device   : gpu device number in case multiple gpus are installed on server
 -hidden_dim   : number of neurons for lstm's hidden state, default = 512
 -embedding_dim: output of cnn encode model, default = 512
```

### 3. Test
```bash
python3 test.py -model <encoder_architecture> -i <image_path> -epoch <saved_model> -gpu_device <gpu_device_number>
```
##### args: 
```bash 
 -i : path of image for generating caption
 ```
 [Download trained model](https://drive.google.com/open?id=1xF8dfIDsz57ZrX7bKApOakyjm1GoelJm)
 
## Results
### Check whether the model is training or not by overfitting on small dataset.
![Screen Shot](train_pic.png)
Since training error is decreasing it seems like model is working just fine.

### Train vs validation loss
[![Screen Shot](train_valid_loss.png)](https://docs.google.com/spreadsheets/d/1VBz6r91D6P_9rybGmbaVNm-P1PM-xcnIWQ0wxQOWFzM/edit?usp=sharing)


Image  |Original Captions|Predicted Captions
----|----|----
![Screen Shot](check/1.jpg)   | 1. a beagle and a golden retriever wrestling in the grass <br> 2. Two dogs are wrestling in the grass <br> 3. Two puppies are playing in the green grass <br> 4. two puppies playing around in the grass <br> 5. Two puppies play in the grass | 50. a brown and white dog is running through a grassy field . <br> 100. a brown dog in a field .  <br> 150. a brown dog is running through a grassy field .  <br> 200. a brown and white dog is laying with its mouth open and people up in the grass . <br> 230. a brown dog running through grass .<br>
![Screen Shot](check/2.jpg)    |  1. a brightly decorated bicycle with cart with people walking around in the background  <br> 2. A street vending machine is parked while people walk by  <br> 3. A street vendor on the corner of a busy intersection  <br> 4. People on the city street walk past a puppet theater  <br> 5. People walk around a mobile puppet theater in a big city . | 50. a man with a green shirt is standing in front of a &lt;unk&gt; at a &lt;unk&gt; . <br> 100.  a group of people standing outside a building . <br> 150. a group of people standing around a outside of building . <br> 200.  a group of people are standing around a city street . <br> 230.  a man in a green shirt &lt;unk&gt; a &lt;unk&gt; at a carnival .  <br>

## References
 * [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)

 * [Deep Visual-Semantic Alignments for Generating Image Descriptions](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf)
 
 * [Flickr 8K Dataset](http://www.jair.org/papers/paper3994.html)
