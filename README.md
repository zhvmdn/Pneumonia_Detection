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
The model is built from a dataset from Kaggle and this dataset is divided into train, test and validation data. Overall, there are 5856 X ray images (JPEG) and each of the 3 folders are breached into 2 subfolders (Normal / Pneumonia), containing first chest X ray images of a healthy person and second is for the X-rays of a person with  pneumonia. Train set has 3875 x-ray images in class Pneumonia and 1341 x-ray images in class Normal. Test set has 390 x-ray images in class Pneumonia and 234 x-ray images in class Normal. Validation set has 8 x-ray images in class Pneumonia and 8 x-ray images in class Normal. As the validation data set has only 16 images,  validation images were moved to the train set. Then the training set was splitted into training and validation (90% - training, 10% - validation) in the ImageDataGenerator. Figure 1 depicts what x-rays of a person with pneumonia and a healthy person look like.


![bar char](https://user-images.githubusercontent.com/124452311/219938830-02ba38ba-d18e-4fdc-a344-527ac6d8cc14.png)




![n and p](https://user-images.githubusercontent.com/124452311/219939041-f848df48-4505-4d5a-bea0-94f5cb245545.png)
