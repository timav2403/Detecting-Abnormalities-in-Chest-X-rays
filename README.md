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











