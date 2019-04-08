## How to Model Human Activity From Smartphone Data

How to Model Human Activity From Smartphone Data
by Jason Brownlee on September 17, 2018 in Deep Learning for Time Series
Tweet   Share
Human activity recognition is the problem of classifying sequences of accelerometer data recorded by specialized harnesses or smart phones into known well-defined movements.

It is a challenging problem given the large number of observations produced each second, the temporal nature of the observations, and the lack of a clear way to relate accelerometer data to known movements.

Classical approaches to the problem involve hand crafting features from the time series data based on fixed-sized windows and training machine learning models, such as ensembles of decision trees. The difficulty is that this feature engineering requires deep expertise in the field.

Recently, deep learning methods such as recurrent neural networks and one-dimensional convolutional neural networks, or CNNs, have been shown to provide state-of-the-art results on challenging activity recognition tasks with little or no data feature engineering.

In this tutorial, you will discover the ‘Activity Recognition Using Smartphones‘ dataset for time series classification and how to load and explore the dataset in order to make it ready for predictive modeling.

*After completing this tutorial, you will know:*

- How to download and load the dataset into memory.
- How to use line plots, histograms, and boxplots to better understand the structure of the motion data.
- How to model the problem, including framing, data preparation, modeling, and evaluation

<hr>

### topics of what we are going to do :

1. Human Activity Recognition
2. Activity Recognition Using Smartphones Dataset
3. Download the Dataset
4. Load Data
5. Balance of Activity Classes
6. Plot Time Series Data for One Subject
7. Plot Histograms Per Subject
8. Plot Histograms Per Activity
9. Plot Activity Duration Boxplots
10. Approach to Modeling




## 1.Human Activity Recognition

Human Activity Recognition, or HAR for short, is the problem of predicting what a person is doing based on a trace of their movement using sensors.

Movements are often normal indoor activities such as standing, sitting, jumping, and going up stairs.

Sensors are often located on the subject, such as a smartphone or vest, and often record accelerometer data in three dimensions (x, y, z).

Human Activity Recognition (HAR) aims to identify the actions carried out by a person given a set of observations of him/herself and the surrounding environment. Recognition can be accomplished by exploiting the information retrieved from various sources such as environmental or body-worn sensors.

The idea is that once the subject’s activity is recognized and known, an intelligent computer system can then offer assistance.

It is a challenging problem because there is no clear analytical way to relate the sensor data to specific actions in a general way. It is technically challenging because of the large volume of sensor data collected (e.g. tens or hundreds of observations per second) and the classical use of hand crafted features and heuristics from this data in developing predictive models.

More recently, deep learning methods have been demonstrated successfully on HAR problems given their ability to automatically learn higher-order features.

Sensor-based activity recognition seeks the profound high-level knowledge about human activities from multitudes of low-level sensor readings. Conventional pattern recognition approaches have made tremendous progress in the past years. However, those methods often heavily rely on heuristic hand-crafted feature extraction, which could hinder their generalization performance. […] Recently, the recent advancement of deep learning makes it possible to perform automatic high-level feature extraction thus achieves promising performance in many areas.



## 2.Activity Recognition Using Smartphones Dataset

A standard human activity recognition dataset is the ‘Activity Recognition Using Smartphones‘ dataset made available in 2012.

It was prepared and made available by Davide Anguita, et al. from the University of Genova, Italy and is described in full in their 2013 paper “A Public Domain Dataset for Human Activity Recognition Using Smartphones.” The dataset was modeled with machine learning algorithms in their 2012 paper titled “Human Activity Recognition on Smartphones using a Multiclass Hardware-Friendly Support Vector Machine.”

The dataset was made available and can be downloaded for free from the UCI Machine Learning Repository:

* [link](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

The data was collected from 30 subjects aged between 19 and 48 years old performing one of 6 standard activities while wearing a waist-mounted smartphone that recorded the movement data. Video was recorded of each subject performing the activities and the movement data was labeled manually from these videos.

below is an example of how the subjects were performing :

<a href="http://www.youtube.com/watch?feature=player_embedded&v=XOEN9W05_4A"><img src="http://img.youtube.com/vi/XOEN9W05_4A/0.jpg" 
alt="subject performing test" width="560" height="315" border="10" /></a>


