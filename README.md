# Alleviated Pose Attentive Capsule Agreement (ALPACA)
Official implementation of Pose Attentive Feature Agreement (ALPACA) of Capsules. ALPACA is an alternative routing method for capsules that contain feature vector, pose and activation at the same time. ALPACA increase run time speed by almost 10 times compared to other methods and increase the performance on novel viewpoint dataset(NVPD). Implementations of Quaternion Capsule Networks, Matrix Capsules with EM routing also available. Small and large CNN versions are readily available for testing.
[NVPD](https://drive.google.com/file/d/1U8w_2fr9pfGUFWxONXf1pGoc3EQDQaM7/view?usp=sharing) is available in 7z format and explanation site will be available soon. 

Also includes capsule network analysis on a NVPD. Includes render script from shapenet, dataloader train/test schemes, models such as Quaternion Capsule networks Matrix Capsules with EM routing, Capsule Routing via Variational Bayes, Star Caps, Capsules with Inverted Dot-Product Attention Routing. NVPD Benchmarks for DenseNet, ResNet50 and SqueezeNet along with two custom CNNs are also available. 

In order to test custom models you can easily integrate your model with compatible input/output to the engine by adding your model to the model creator. There are several engines that can be used for different types of outputs.

## Arhitecture
![](https://github.com/Boazrciasn/PAFA-Capsules/blob/3ea0f8bf61919655313162d4b49436ca2c2af798/images/architecture.jpg)



## NVPD Dataset

Common shapenet objects renedered without texture from multiple viewpoints and distances. 

### Dataset splits
![](https://github.com/Boazrciasn/PAFA-Capsules/blob/master/images/dataset_splits.jpg)

### NVPD samples
![](https://github.com/Boazrciasn/PAFA-Capsules/blob/3ea0f8bf61919655313162d4b49436ca2c2af798/images/_dAirplane.gif) ![](https://github.com/Boazrciasn/PAFA-Capsules/blob/3ea0f8bf61919655313162d4b49436ca2c2af798/images/_dCar.gif)
![](https://github.com/Boazrciasn/PAFA-Capsules/blob/3ea0f8bf61919655313162d4b49436ca2c2af798/images/_dChair.gif) ![](https://github.com/Boazrciasn/PAFA-Capsules/blob/3ea0f8bf61919655313162d4b49436ca2c2af798/images/_dGuitar.gif)
![](https://github.com/Boazrciasn/PAFA-Capsules/blob/3ea0f8bf61919655313162d4b49436ca2c2af798/images/_dLamp.gif)

![](https://github.com/Boazrciasn/PAFA-Capsules/blob/3ea0f8bf61919655313162d4b49436ca2c2af798/images/_dMotorbike.gif) ![](https://github.com/Boazrciasn/PAFA-Capsules/blob/3ea0f8bf61919655313162d4b49436ca2c2af798/images/_dMug.gif)
![](https://github.com/Boazrciasn/PAFA-Capsules/blob/3ea0f8bf61919655313162d4b49436ca2c2af798/images/_dSofa.gif) ![](https://github.com/Boazrciasn/PAFA-Capsules/blob/3ea0f8bf61919655313162d4b49436ca2c2af798/images/_dTable.gif)
![](https://github.com/Boazrciasn/PAFA-Capsules/blob/3ea0f8bf61919655313162d4b49436ca2c2af798/images/_dTrain.gif)


### Results

![](https://github.com/Boazrciasn/PAFA-Capsules/blob/master/images/results.jpg)
