# Detecting Abnormalities in Chest X-rays

## Objective
The objective for this project is to create neural network machine learning models using AlexNet, Inception, and ResNet architectures. The hope is to have at least one of these three models produce high accuracy metrics, if not all. In addition, exploring the different architectures will help reveal the best structure for this type of dataset. Creating these models is extremely beneficial in the healthcare industry because they will be able to detect pneumonia or pulmonary disease efficiently to flag at-risk patients. 

Pneumonia is diagnosed in about four million adults per year in the United States and about 15% are hospitalized due to it and for the people whom are hospitalized, the aggregate cost comes to about four billion dollars a year (Fine et al., 2019). According to the Centers for Disease Control and Prevention (CDC), pneumonia has a mortality rate of 15.1 per 100,000 of patients with pneumonia and about 15.3 per 100,000 for pulmonary disease (CDC, 2019).

Given the brief details of each disease, having a model to help detect pulmonary or pneumonia would benefit victims of these diseases greatly. 

## Dataset
The pneumonia dataset has been organized into 3 folders, train, test, and validation which are distributed into normal and pneumonia (bacterial and viral) images. There is a total of 5,863 X-ray JPEG images of the anterior-posterior side of the chest from pediatric patients. The images were screened for image quality and diagnosed by two physicians.

The pulmonary dataset contains 336 pulmonary png images and 326 normal png images from the National Library of Medicine. The data contains abnormalities of effusions and miliary patterns. The images were screened for image quality and diagnosed by physicians.

I retrieved both datasets from Kaggle, each being from different problems. Simply by downloading the dataset and unzipping the file via git GUI and placing in my directory. My goal to bring to datasets together from two different problems was to make the problem more challenging. Most of the deep learning problems were binary, such as the two datasets I found individually. However, merging the two datasets provided multi-classification problem. 

## Preprocesssing
The first transformation I made was converting the PNG pulmonary files to JPEG. After this, I created a new folder to house all the images. The sub-folders were created to mirror the pneumonia dataset having train and test folders. The data was split using a 70/30 percentage and all the images, whether normal, pneumonia, or pulmonary, were distributed evenly by type. 

## Exploratory Data Analysis
I started by accessing images from the trainset folder to view 10 random images. Figure 1 shows normal chest x-rays, figure 1.1 shows pneumonia chest x-rays, and figure 1.2 shows pulmonary x-rays. 

Once I have seen samples of my images, I created bar plots to view the number of images in each folder. Figure 2 provides the total training images by type and figure 2.1 shows the total testing images by type.

Figure 1.
![image](https://user-images.githubusercontent.com/43620431/56870567-4eeb5f00-69c6-11e9-9358-73bfdbd5b0d2.png)

Figure 1.1
![image](https://user-images.githubusercontent.com/43620431/56870587-865a0b80-69c6-11e9-8d09-8f1c98b4c71d.png)

Figure 1.2
![image](https://user-images.githubusercontent.com/43620431/56870591-8f4add00-69c6-11e9-864b-0b0c21f090a6.png)

Figure 2.
![image](https://user-images.githubusercontent.com/43620431/56870614-c9b47a00-69c6-11e9-86a1-976a5947f9d4.png)

Figure 2.1
![image](https://user-images.githubusercontent.com/43620431/56870616-cde09780-69c6-11e9-8586-d1eb538bfa7d.png)

## Model and Architecture
I used three different architectures, AlexNet, Inception, and ResNet, to find an optimal model.

### AlexNet Model
Figure 3 shows an example of an AlexNet model and figure 3.1 provides the summary of the model derived after creating several different models and tuning them. Initially I tried running the model without any dropouts or batch normalizations and added regulizers within each convolution. However, my final model consisted of no regulizers within the convolutions and inserted batch normalizations with dropouts to help with the overfitting problem I was running into.

Figure 3
![image](https://user-images.githubusercontent.com/43620431/56874580-a3570480-69ef-11e9-8207-a5adbc48b679.png)

Figure 3.1
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================

conv2d_1 (Conv2D)            (None, 222, 222, 128)     3584      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 220, 220, 128)     147584    
_________________________________________________________________
batch_normalization_1 (Batch (None, 220, 220, 128)     512       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 110, 110, 128)     0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 110, 110, 128)     0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 108, 108, 128)     147584    
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 106, 106, 128)     147584    
_________________________________________________________________
batch_normalization_2 (Batch (None, 106, 106, 128)     512       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 53, 53, 128)       0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 51, 51, 64)        73792     
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 49, 49, 64)        36928     
_________________________________________________________________
batch_normalization_3 (Batch (None, 49, 49, 64)        256       
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 24, 24, 64)        0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 24, 24, 64)        0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 22, 22, 64)        36928     
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 20, 20, 64)        36928     
_________________________________________________________________
batch_normalization_4 (Batch (None, 20, 20, 64)        256       
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 20, 20, 64)        0         
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 18, 18, 32)        18464     
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 16, 16, 32)        9248      
_________________________________________________________________
batch_normalization_5 (Batch (None, 16, 16, 32)        128       
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 16, 16, 32)        0         
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 14, 14, 32)        9248      
_________________________________________________________________
batch_normalization_6 (Batch (None, 14, 14, 32)        128       
_________________________________________________________________
flatten_1 (Flatten)          (None, 6272)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               802944    
_________________________________________________________________
dropout_3 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 387       
=================================================================
Total params: 1,472,995
Trainable params: 1,472,099
Non-trainable params: 896
_________________________________________________________________



### Inception Model
Figure 4 shows an example of an Inception model and figure 4.1 provides the summary of the model derived after creating several different models and tuning them as well. This model consisted of 5 convolution towers within the Inception model and for this portion I used dropouts, batch normalizations, and regulizers to help with overfitting. 

Figure 4
![image](https://user-images.githubusercontent.com/43620431/56876379-c7204780-69fb-11e9-8680-9f3f7573c9e4.png)

Figure 4.1
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 224, 224, 3)  0                                            
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 224, 224, 32) 896         input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 224, 224, 32) 2432        input_1[0][0]                    
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 224, 224, 3)  0           input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 224, 224, 32) 9248        conv2d_2[0][0]                   
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 224, 224, 3)  12          input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 224, 224, 32) 25632       conv2d_5[0][0]                   
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 224, 224, 3)  12          input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 224, 224, 32) 128         max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 224, 224, 32) 0           conv2d_3[0][0]                   
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 224, 224, 32) 896         batch_normalization_1[0][0]      
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 224, 224, 32) 0           conv2d_6[0][0]                   
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 224, 224, 32) 2432        batch_normalization_2[0][0]      
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 224, 224, 160 0           conv2d_1[0][0]                   
                                                                 dropout_1[0][0]                  
                                                                 conv2d_4[0][0]                   
                                                                 dropout_2[0][0]                  
                                                                 conv2d_7[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 112, 112, 160 0           concatenate_1[0][0]              
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 112, 112, 160 640         max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 112, 112, 32) 46112       batch_normalization_3[0][0]      
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 112, 112, 32) 128032      batch_normalization_3[0][0]      
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 112, 112, 160 0           batch_normalization_3[0][0]      
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 112, 112, 32) 9248        conv2d_9[0][0]                   
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 112, 112, 160 640         batch_normalization_3[0][0]      
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 112, 112, 32) 25632       conv2d_12[0][0]                  
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 112, 112, 160 640         batch_normalization_3[0][0]      
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 112, 112, 32) 5152        max_pooling2d_3[0][0]            
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 112, 112, 32) 0           conv2d_10[0][0]                  
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 112, 112, 32) 46112       batch_normalization_4[0][0]      
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, 112, 112, 32) 0           conv2d_13[0][0]                  
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 112, 112, 32) 128032      batch_normalization_5[0][0]      
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 112, 112, 160 0           conv2d_8[0][0]                   
                                                                 dropout_3[0][0]                  
                                                                 conv2d_11[0][0]                  
                                                                 dropout_4[0][0]                  
                                                                 conv2d_14[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 56, 56, 160)  0           concatenate_2[0][0]              
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 56, 56, 160)  640         max_pooling2d_4[0][0]            
__________________________________________________________________________________________________
dropout_5 (Dropout)             (None, 56, 56, 160)  0           batch_normalization_6[0][0]      
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 56, 56, 32)   46112       dropout_5[0][0]                  
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 56, 56, 32)   128032      dropout_5[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_5 (MaxPooling2D)  (None, 56, 56, 160)  0           dropout_5[0][0]                  
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 56, 56, 32)   9248        conv2d_16[0][0]                  
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 56, 56, 160)  640         dropout_5[0][0]                  
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 56, 56, 32)   25632       conv2d_19[0][0]                  
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 56, 56, 160)  640         dropout_5[0][0]                  
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 56, 56, 32)   5152        max_pooling2d_5[0][0]            
__________________________________________________________________________________________________
dropout_6 (Dropout)             (None, 56, 56, 32)   0           conv2d_17[0][0]                  
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 56, 56, 32)   46112       batch_normalization_7[0][0]      
__________________________________________________________________________________________________
dropout_7 (Dropout)             (None, 56, 56, 32)   0           conv2d_20[0][0]                  
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 56, 56, 32)   128032      batch_normalization_8[0][0]      
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 56, 56, 160)  0           conv2d_15[0][0]                  
                                                                 dropout_6[0][0]                  
                                                                 conv2d_18[0][0]                  
                                                                 dropout_7[0][0]                  
                                                                 conv2d_21[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_6 (MaxPooling2D)  (None, 28, 28, 160)  0           concatenate_3[0][0]              
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 28, 28, 160)  640         max_pooling2d_6[0][0]            
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 125440)       0           batch_normalization_9[0][0]      
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 128)          16056448    flatten_1[0][0]                  
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 3)            387         dense_1[0][0]                    
==================================================================================================
Total params: 16,879,643
Trainable params: 16,877,391
Non-trainable params: 2,252
__________________________________________________________________________________________________

### ResNet Model



Figure 5


Figure 5.1
