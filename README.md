# Pose Attentive Feature Agreement (PAFA) of Capsules
Official implementation of Pose Attentive Feature Agreement (PAFA) of Capsules. PAFA is an alternative routing method for capsules that contain feature vector, pose and activation at the same time. PAFA increase run time speed by almost 10 times compared to other methods and increase the performance on novel viewpoint dataset(NVPD). Implementations of Quaternion Capsule Networks, Matrix Capsules with EM routing also available. Small and large CNN versions are readily available for testing. 
Repository also include a link to dataset files: [NVPD](https://drive.google.com/file/d/1U8w_2fr9pfGUFWxONXf1pGoc3EQDQaM7/view?usp=sharing) and explanation site will be available soon. Repository also inclodes render script from ShapeNet to create NVPD of your own, dataloader train test schemes.

Also includes capsule network analysis on a NVPD. Includes render script from shapenet, dataloader train/test schemes, models such as Quaternion Capsule networks Matrix Capsules with EM routing, Dynamic Routing with Capsules in pytorch. 

## Arhitecture
![](https://github.com/Boazrciasn/PAFA-Capsules/blob/3ea0f8bf61919655313162d4b49436ca2c2af798/images/architecture.jpg)



## NVPD Dataset

Common shapenet objects renedered without texture from multiple viewpoints and distances. 

### Dataset splits
![](https://github.com/Boazrciasn/PAFA-Capsules/blob/3ea0f8bf61919655313162d4b49436ca2c2af798/images/dataset%20visuals.jpg)

### NVPD samples
![](https://github.com/Boazrciasn/PAFA-Capsules/blob/3ea0f8bf61919655313162d4b49436ca2c2af798/images/_dAirplane.gif) ![](https://github.com/Boazrciasn/PAFA-Capsules/blob/3ea0f8bf61919655313162d4b49436ca2c2af798/images/_dCar.gif)
![](https://github.com/Boazrciasn/PAFA-Capsules/blob/3ea0f8bf61919655313162d4b49436ca2c2af798/images/_dChair.gif) ![](https://github.com/Boazrciasn/PAFA-Capsules/blob/3ea0f8bf61919655313162d4b49436ca2c2af798/images/_dGuitar.gif)
![](https://github.com/Boazrciasn/PAFA-Capsules/blob/3ea0f8bf61919655313162d4b49436ca2c2af798/images/_dLamp.gif)

![](https://github.com/Boazrciasn/PAFA-Capsules/blob/3ea0f8bf61919655313162d4b49436ca2c2af798/images/_dMotorbike.gif) ![](https://github.com/Boazrciasn/PAFA-Capsules/blob/3ea0f8bf61919655313162d4b49436ca2c2af798/images/_dMug.gif)
![](https://github.com/Boazrciasn/PAFA-Capsules/blob/3ea0f8bf61919655313162d4b49436ca2c2af798/images/_dSofa.gif) ![](https://github.com/Boazrciasn/PAFA-Capsules/blob/3ea0f8bf61919655313162d4b49436ca2c2af798/images/_dTable.gif)
![](https://github.com/Boazrciasn/PAFA-Capsules/blob/3ea0f8bf61919655313162d4b49436ca2c2af798/images/_dTrain.gif)


### Results

![](https://github.com/Boazrciasn/PAFA-Capsules/blob/3ea0f8bf61919655313162d4b49436ca2c2af798/images/results.jpg)
