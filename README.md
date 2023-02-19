# Pneumonia_Detection

## Introduction:
### Problem:
Pneumonia is one of the very contagious diseases, the so-called "Lung Infection" that has affected the lives of millions of people around the world. You're more likely to get pneumonia if you have asthma, chronic obstructive pulmonary disease (COPD) or heart disease. Most often, this disease causes respiratory symptoms that can be very similar to shortness of breath, acute respiratory viral infections or tachycardia. Pneumonia can range in seriousness from mild to life-threatening. Pneumonia can be quickly and accurately diagnosed using computed tomography (CT) and chest X-ray (CXR). However, since it takes a long time and is very prone to human error, identifying an infection manually using a radio image is quite difficult. Our goal is to create an image recognition model that will allow us to determine the presence of Pneumonia from an X-ray image of the patient's lungs. 

### Literature review with links (another solutions):
Author of this model used a custom deep convolutional neural network and retraining a pre-trained “InceptionV3" model to identify pneumonia from x-ray images. For retraining, he removed the output layers, freezed the first few layers, and fine-tuned the model for two Pneumonia and Normal classes..
https://github.com/anjanatiha/Pneumonia-Detection-from-Chest-X-Ray-Images-with-Deep-Learning

Author of this project divided the train set into training and validation data. He solved the problem of data imbalance using stratified class. Training and validation set split changed from 99:1 ratio to 90:10 ratio.
https://www.kaggle.com/code/wirachleelakiatiwong/pneumania-classification-transfered-cnn/notebook

In this project, the author compared pre-trained models such as VGG16, CNN_2, ResNet, InceptionNet, DenseNet  with their mean absolute error. Also, the author built his own CNN model that showed 91% of accuracy.
https://github.com/0xpranjal/Pneumonia-Detection-using-Deep-Learning

### Current work (description of the work) 
We used Google Colab as IDE and installed kaggle to upload the dataset into Colab files. Our work has 3 data sets: train (5216 images), validation (16 images), and test (624 images) sets. As the number of images in the validation set is low, we moved the images from the validation set to the train set. . As the validation data set has a low number of images, we splitted the train set into training data (80%) and validation data (20%). 
Keras class ImageDataGenerator used to enable a quick setup of Python generators that can automatically turn image files on disk into batches of pre-processed tensors. With this class train data set was divided into two sets and data augmentation was applied to train and validation sets. All images were rescaled by 1/255 and  were resized  to 150x150. X rays of a healthy person and person with pneumonia were visualized with matplotlib.pyplot library.
We used the GPU effectively in this project. First, we loaded the image data, followed by the creation and training of a convolutional neural network, which allowed us to fine-tune and order the model while also predicting the results. Because X-rays are only taken in one orientation, data magnification is not included in the model, and there will be no changes like flips on real X-ray images. Metric Recall creates two local variables, true_positives and false_negatives, that are used to compute the recall. 

## Data and Methods 
### Information about the data 
The model is built from a dataset from Kaggle and this dataset is divided into train, test and validation data. Overall, there are 5856 X ray images (JPEG) and each of the 3 folders are breached into 2 subfolders (Normal / Pneumonia), containing first chest X ray images of a healthy person and second is for the X-rays of a person with  pneumonia. Train set has 3875 x-ray images in class Pneumonia and 1341 x-ray images in class Normal. Test set has 390 x-ray images in class Pneumonia and 234 x-ray images in class Normal. Validation set has 8 x-ray images in class Pneumonia and 8 x-ray images in class Normal. As the validation data set has only 16 images,  validation images were moved to the train set. Then the training set was splitted into training and validation (90% - training, 10% - validation) in the ImageDataGenerator. Figure 1 depicts what x-rays of a person with pneumonia and a healthy person look like.It can be noticed that normal x-rays have less white shading in the lung area and are more clear, while Pneumonia x-rays are more opaque.

![2023-02-19 (3)](https://user-images.githubusercontent.com/124452311/219939283-b6d361c7-9771-4844-88f0-fc15d524d089.png)
                                                          Figure 1. X-ray images

It was found that the images of pneumonia are much more than normal during familiarization with the data set. Bar graph (Figure 2) below proves that the number of normal x-ray images in the training set is less, or the data in that set is unbalanced. Also, Class 0 (Normal) weight is greater than class 1 weight (Pneumonia), meaning that the number of the normal x-ray images is less. To make the training data balanced, each normal image was weighted more to balance the data. Also, data augmentation was used for better accuracy and to avoid overfitting.

![bar](https://user-images.githubusercontent.com/124452311/219939584-54c4d96e-2e9e-42b8-86c1-87f65325520b.png)

  keras.preprocessing.image class Image Data Generator generates batches of tensor image data with real-time data augmentation. We do not have enough diverse sets of images, so for data cleaning and preprocessing data augmentation creates more training samples. Parameters such as rescale (img pixels between 0 and 1), rotation_range (range for rotation), shear_range (shear intensity), zoom_range (the image is enlarged) and horizontal_flip (randomly flips the images horizontally) were used for augmentation on the training set. All images were rescaled by 1/255 and  were resized  to 150x150. Validation split is set to 0.1, so training data (90%) and validation data (10%) will be created from the data of the training set. flow_from_directory reads the images from folders. Target size was set to 150x150 and batch size is 32. Image batch shape : (32, 150, 150, 3).

### Description of the ML models you used with some theory 
For our project, we have created Convolutional Neural Networks architecture. A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm that can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image, and be able to differentiate one from the other. CNNs are especially useful for finding patterns in images for recognizing objects, classes, and categories (Figure 3). The model is built with 4 convolutional blocks consisting of convolutional layers, max-pooling. The depth of the feature maps increased from 32 to 64 and size of the feature maps decreased from 150x150 to 10x10. In the end, we used the Dense layer of size 3 and sigmoid function as this is a binary classification problem. For compilation, Adam is the optimizer and binary cross-entropy is the loss. In between, Dropout is used to avoid overfitting. As this is the imbalanced data set, for the metric, Precision and Recall are included and they show a more accurate view of how effective the model is. Additionally, precision measures the accuracy of positive predictions, while recall measures the completeness of positive predictions. Labels were not defined as 0 and 1 since ImageDataGenerator already made that (rescale = 1/255). Also, we used fine-tuning to speed up the training and overcome the small dataset size.  Early Stopping  callback is used to avoid overfitting by stopping the epochs depending on a metric and conditions. restore_best_weights is set to true, so the returned model is the model with the best weights. Reduce learning rate when a metric has stopped improving.
![CNN model](https://user-images.githubusercontent.com/124452311/219939776-d6ff6e86-bb9f-405f-91e5-fca6637fbc4d.png)

## Results 
After training, we obtained the plots of accuracy and loss of training and validation. As can be seen in  Figure 4, training accuracy or the number of correct predictions increases, and its loss or summation of errors decreases. When it comes to validation, its accuracy rises and loss decreases, but it fluctuates a little. In 10 epochs, the model achieved  94% of training and 91% of  validation accuracy.

![fig4](https://user-images.githubusercontent.com/124452311/219940047-7bf0143a-272b-4bf3-99de-1fc78164fc4d.png)
We made an evaluation on the test set. Results showed that we got test accuracy of 92.15%, test loss of 0.22, test precision of 93.38%, and test recall of 94.10%.

![fig5](https://user-images.githubusercontent.com/124452311/219940051-64f0cc35-f0d0-4f33-bd6a-f1de611e3775.png)
Classification report details on the normal and pneumonia class is in Figure 6. We obtained a recall of 94% for Pneumonia class and Precision of 93%, which are pretty good.

![fig6](https://user-images.githubusercontent.com/124452311/219940062-f967aaaa-7f0a-46a1-b805-61963181f94a.png)
False Negatives for Model are just 23 and False Positives are 26. Our model is able to classify 575 x ray images correctly out of 624 test images.

![confusion](https://user-images.githubusercontent.com/124452311/219940068-f4c34ca3-65e6-4872-9673-0f99bd969218.png)
ROC Curve with AUC score (0.97) is shown below. 

![fig8](https://user-images.githubusercontent.com/124452311/219940444-ffe0e467-a9d2-4eb2-81c4-a1c03dbbef1d.png)

![fig9](https://user-images.githubusercontent.com/124452311/219940453-9e3eb27d-d04d-4c40-8afa-a1840fe0db7d.png)

To test the model with images we used Gradio, which is a free and open-source Python library that allows us to develop an easy-to-use customizable component demo for a machine learning model that anyone can use anywhere. Figure 10 and Figure 11 depicts how the model classified correctly normal and pneumonia x-ray images which were taken from the test set’s normal and pneumonia folders.

![fig10-11](https://user-images.githubusercontent.com/124452311/219940702-ecbc4666-c222-4dac-a79f-8b3a19c56bcd.png)

## Discussion 
### Critical review of results 
We think that our model makes good predictions as test accuracy was 92%. 
As mentioned, our project is based on the method of training using CNN to identify normal patients and patients with pneumonia on chest X-rays. The most successful CNN dataset was chosen for this purpose. The project was executed well, but during the simulation it was not without some difficulties.
1) According to the data, as you can see, the model works very well, it is worth mentioning that in our model 49 photos do not take out the correct data. In the future, you can improve the results at a higher accuracy by resolving this error.
2) Next, during the construction of the model, or rather in the image data generator, we encountered problems with the choice of parameters. We tried to choose a better generator, thereby using, for example, rotation ranges, rescales and so on.
3) And finally, since our dataset was not balanced, we faced distribution problems. Having solved this problem, we came to the conclusion that we need more X-ray images for a more accurate result and preferably with equal data.
