# Machine_Learning
Building machine Learning models using Pytorch

# Implemented a simplified ResNet, which solves the problem of "Vanishing Gradient" while dealing with more number of layers in Neural Networks :
 Where Can I potentially use this? :  Image/Text classification, Object Recognition tasks, Language modeling.
 Part A: Building Class "Block", 
 The input to Block is of shape (N, C, H, W), where N denotes the batch size, C denotes the number of channels, and H and W are the height and width of each channel.   For each data example x with shape (C, H, W), the output of block is
          Block(x) = σr(x + f(x)),
          where σr denotes the ReLU activation, and f(x) also has shape (C, H, W) and thus can be added to x.
 1. Following are the layers used in building them: 
   i. A Conv2d with C input channels, C output channels, kernel size 3, stride 1, padding 1, and no bias term.
   ii. A BatchNorm2d with C features.
   iii. A ReLU layer.
   iv. Another Conv2d with the same arguments as i above.
   v. Another BatchNorm2d with C features.
   Because 3 × 3 kernels and padding 1 are used, the convolutional layers do not change the shape of each channel. Moreover, the number of channels are also kept        unchanged. Therefore f(x) does have the same shape as x 
 2.  Implementing a (shallow) ResNet consists of the following parts:
  i. A Conv2d with 1 input channel, C output channels, kernel size 3, stride 2, padding 1, and no bias term.
  ii. A BatchNorm2d with C features.
  iii. A ReLU layer.
  iv. A MaxPool2d with kernel size 2.
  v. A Block with C channels.
  vi. An AdaptiveAvgPool2d which for each channel takes the average of all elements.
  vii. A Linear with C inputs and 10 output
  
Experiment : Tried working with diffrent values of C = {1,2,4,64} and training the model for 4000 epochs using SGD with mini batch size 128 and step size 0.1

Results: As the C value increases it means we are training the model with more features, which means increasing the complexity of the model. So, increasing to a far extent might lead to Overfitting the curve, where models tend to memorise the data rather than learning it. As C increases the training error reduces because we are working on reduction in loss function of training data. Whereas test error trend depends on the Overfitting or Underfitting of training data to the model.Here , On C = 1 the trend of training and test values are following a similar pattern and there is no sign of Overfitting as we have only used one channel

![image](https://github.com/LokeshSreenathJ/Machine_Learning/assets/115972450/6eb50c8a-e779-49ea-95e7-ebed3e21baa2)
![image](https://github.com/LokeshSreenathJ/Machine_Learning/assets/115972450/1c80260b-e794-46da-949c-412f302cdac8)
![C=4 Total](https://github.com/LokeshSreenathJ/Machine_Learning/assets/115972450/47ed9ec3-6486-4746-8337-fa0c0e5d9a7f)
![C =64 Total](https://github.com/LokeshSreenathJ/Machine_Learning/assets/115972450/49a18c0a-accf-4f8b-93a3-21d8ad1b5902)
As we currently have C = 64, that is our model is learning 64 different features, higher the complexity of the model. So the model can learn the trends quickly compared to C =4. So upon training it for more and more epochs we can see that it overfits the data. The indication is test error keeps on increasing while training error is non-increasing, the gap between them increases. This gap was present initially when model is learning all the features, once our model learns them then training and test error at each epoch remains same, upon further training it again the gap rises as test error starts to increase, because model kept trained on train data for longer time so model cannot identify the newer observation.

