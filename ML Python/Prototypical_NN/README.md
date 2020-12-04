<img src="https://raw.githubusercontent.com/oscarknagg/few-shot/master/assets/proto_nets_diagram.png">

# Few-shot learning - Prototypical Networks
My version of Prototypical Networks for Few-shot Learning.

Thanks to [mpatacchiola](https://github.com/mpatacchiola) for sharing those useful links and for this explanation:
https://www.youtube.com/watch?v=rHGPfl0pvLY&ab_channel=MassimilianoPatacchiola

It helped me a lot on the process of understanding the idea about the prototypical's working process.

### Observations:
- To run the project just execute the <b>main.py</b> file.
- The database is the famous "omniglot". I've changed the structure inside the repository to a format that was easier for me to work on. The "data.zip" file has the structure that I used within this project. Unzip the file before executing the project.

### Requeriments:
- numpy: 1.17.4
- Pillow: 8.0.1
- torch: 1.6.0+cu101
- torchvision: 0.7.0+cu101

## The problem:
Classification problems usually require big databases with a good distribution of each class, especially if the problem is related to image classification. Thinking about it I've started some studies on meta-learning aiming to find a technique that could help with this limitation. Why do I need so many images to build a classifier whether humans can do it just by looking at a few examples? 
So the problem here is: It's possible to build a classifier that can classify images looking just over a few examples of the whole context.

## The solution:
As result, I've found this technique - Few-shot learning: Prototypical Networks. The big thing about this technique is the way you "teach" the model. Instead of apply correction creating the correlation between the image and its class, the correction is made in a way where a correlation between two images from the same class be built. In other words, if I provide the model two images from the same class, it must say that these images are "equal". Check the links I've provided above for more details.

About the neural network architecture, it is a simple convolutional architecture with a flattened output. As I've said, the big thing in Prototypical Netwroks is how to compute the error.

### The hyperparameters and model:
- The file with the hyperparameters configuration is the <b>main.py</b>. Look for the comment "HYPERPARAMETERS".

- The actual configuration of the hyperparameters is: 
  - Learning Rate: 1e-3 (Adam)
  - N_CLASS: 60 (the number of different classes into each batch)

- For the neural model I used PyTorch with the following architecture:
  - Encoder    
                     
    - Conv Block 
      - conv1 = Conv2d( in_channels=1, out_channels=32, kernel_size=3, padding=1)
      - bn1   = BatchNorm2d(32)
      - Relu
      - mp1   = MaxPool2d(2)

    - Conv Block
      - conv2 = Conv2d( in_channels=32, out_channels=32, kernel_size=3, padding=1)
      - bn2   = BatchNorm2d(32)
      - Relu
      - mp2   = MaxPool2d(2)

    - Conv Block
      - conv3 = Conv2d( in_channels=32, out_channels=32, kernel_size=3, padding=1)
      - bn3   = BatchNorm2d(32)
      - Relu
      - mp3   = MaxPool2d(2)

    - Conv Block
      - conv4 = Conv2d( in_channels=32, out_channels=32, kernel_size=3, padding=1)
      - bn4   = BatchNorm2d(32)
      - Relu
      - mp4   = MaxPool2d(2)        

  - Output
    Flatten (600)
