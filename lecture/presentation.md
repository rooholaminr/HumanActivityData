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
9. Approach to Modeling




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

<hr>

#### since part 3 and 4 doesn't include any technical difficulties I skip.

<br> 

## 5. Balance of Activity Classes
A good first check of the data is to investigate the balance of each activity.

We believe that each of the 30 subjects performed each of the six activities.

Confirming this expectation will both check that the data is indeed balanced, making it easier to model, and confirm that we are correctly loading and interpreting the dataset.

We can develop a function that summarizes the breakdown of the output variables, e.g. the y variable.

The function class_breakdown() below implements this behavior, first wrapping the provided NumPy array in a DataFrame, grouping the rows by the class value, and calculating the size of each group (number of rows). The results are then summarized, including the count and the percentage.

```python
# summarize the balance of classes in an output variable column
def class_breakdown(data):
	# convert the numpy array into a dataframe
	df = DataFrame(data)
	# group data by the class value and calculate the number of rows
	counts = df.groupby(0).size()
	# retrieve raw rows
	counts = counts.values
	# summarize
	for i in range(len(counts)):
		percent = counts[i] / len(df) * 100
		print('Class=%d, total=%d, percentage=%.3f' % (i+1, counts[i], percent))
 ```

It may be useful to summarize the breakdown of the classes in the train and test datasets to ensure they have a similar breakdown, then compare the result to the breakdown on the combined dataset.

The complete example is listed below.

```python
# summarize class balance
from numpy import array
from numpy import vstack
from pandas import read_csv
from pandas import DataFrame

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# summarize the balance of classes in an output variable column
def class_breakdown(data):
	# convert the numpy array into a dataframe
	df = DataFrame(data)
	# group data by the class value and calculate the number of rows
	counts = df.groupby(0).size()
	# retrieve raw rows
	counts = counts.values
	# summarize
	for i in range(len(counts)):
		percent = counts[i] / len(df) * 100
		print('Class=%d, total=%d, percentage=%.3f' % (i+1, counts[i], percent))

# load train file
trainy = load_file('HARDataset/train/y_train.txt')
# summarize class breakdown
print('Train Dataset')
class_breakdown(trainy)

# load test file
testy = load_file('HARDataset/test/y_test.txt')
# summarize class breakdown
print('Test Dataset')
class_breakdown(testy)

# summarize combined class breakdown
print('Both')
combined = vstack((trainy, testy))
class_breakdown(combined)
```
Running the example first summarizes the breakdown for the training set. We can see a pretty similar distribution of each class hovering between 13% and 19% of the dataset.

The result on the test set and on both datasets together look very similar.

It is likely safe to work with the dataset assuming the distribution of classes is balanced per train and test set and perhaps per subject.

```
Train Dataset
Class=1, total=1226, percentage=16.676
Class=2, total=1073, percentage=14.595
Class=3, total=986, percentage=13.411
Class=4, total=1286, percentage=17.492
Class=5, total=1374, percentage=18.689
Class=6, total=1407, percentage=19.138

Test Dataset
Class=1, total=496, percentage=16.831
Class=2, total=471, percentage=15.982
Class=3, total=420, percentage=14.252
Class=4, total=491, percentage=16.661
Class=5, total=532, percentage=18.052
Class=6, total=537, percentage=18.222

Both
Class=1, total=1722, percentage=16.720
Class=2, total=1544, percentage=14.992
Class=3, total=1406, percentage=13.652
Class=4, total=1777, percentage=17.254
Class=5, total=1906, percentage=18.507
Class=6, total=1944, percentage=18.876
```

<hr>

## 6. Plot Time Series Data for One Subject

We are working with time series data, therefore an import check is to create a line plot of the raw data.

The raw data is comprised of windows of time series data per variable, and the windows do have a 50% overlap. This suggests we may see some repetition in the observations as a line plot unless the overlap is removed.

in this section since the code chunks needed an immense refactor I decided just to bring in the plots of subject activities and features of their activities over time.
I did it for two subjects as following.

![Line plot for all variables for a single subject](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2018/09/Line-plot-for-all-variables-for-a-single-subject.png)

![Line plot for all variables for a second single subject](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2018/09/Line-plot-for-all-variables-for-a-single-second-subject.png)


we see a lot of commonality across the nine variables. It is very likely that only a subset of these traces are required to develop a predictive model.


## 7. Plot Histograms Per Subject

As the problem is framed, we are interested in using the movement data from some subjects to predict activities from the movement of other subjects.

This suggests that there must be regularity in the movement data across subjects. We know that the data has been scaled between -1 and 1, presumably per subject, suggesting that the amplitude of the detected movements will be similar.

We would also expect that the distribution of movement data would be similar across subjects, given that they performed the same actions.

We can check for this by plotting and comparing the histograms of the movement data across subjects. A useful approach would be to create one plot per subject and plot all three axis of a given data (e.g. total acceleration), then repeat this for multiple subjects. The plots can be modified to use the same axis and aligned horizontally so that the distributions for each variable across subjects can be compared.

plot_subject_histograms() function to next plot the distributions of the body acceleration :

```python
# plot histograms for multiple subjects
def plot_subject_histograms(X, y, sub_map, n=10):
	pyplot.figure()
	# get unique subjects
	subject_ids = unique(sub_map[:,0])
	# enumerate subjects
	xaxis = None
	for k in range(n):
		sub_id = subject_ids[k]
		# get data for one subject
		subX, _ = data_for_subject(X, y, sub_map, sub_id)
		# body acc
		for i in range(3):
			ax = pyplot.subplot(n, 1, k+1, sharex=xaxis)
			ax.set_xlim(-1,1)
			if k == 0:
				xaxis = ax
			pyplot.hist(to_series(subX[:,:,3+i]), bins=100)
	pyplot.show()

```

Histograms of the body acceleration data for 10 subjects :

Running the updated example creates the same plot with very different results.

Here we can see all data clustered around 0.0 across axis within a subject and across subjects. This suggests that perhaps the data was centered (zero mean). This strong consistency across subjects may aid in modeling, and may suggest that the differences across subjects in the total acceleration data may not be as helpful.
![alt image](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2018/09/Histograms-of-the-body-acceleration-data-for-10-subjects.png)





## 8. Plot Histograms Per Activity

We are interested in discriminating between activities based on activity data.

The simplest case for this would be to discriminate between activities for a single subject. One way to investigate this would be to review the distribution of movement data for a subject by activity. We would expect to see some difference in the distribution between the movement data for different activities by a single subject.

We can review this by creating a histogram plot per activity, with the three axis of a given data type on each plot. Again, the plots can be arranged horizontally to compare the distribution of each data axis by activity. We would expect to see differences in the distributions across activities down the plots.

I'm going to print out the output plots and skip the codes since the codes are not worthwhile for our lecture :

our goal is to find characteristics of this dataset this time based on activities and the data we have. 

1. Histograms of the total acceleration data by activity

![img Alt](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2018/09/Histograms-of-the-total-acceleration-data-by-activity.png)
We can see that each activity has a different data distribution, with a marked difference between the large movement (first three activities) with the stationary activities (last three activities). Data distributions for the first three activities look Gaussian with perhaps differing means and standard deviations. Distributions for the latter activities look multi-modal (i.e. multiple peaks).

2. Histograms of the body acceleration data by activity

![alt img](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2018/09/Histograms-of-the-body-acceleration-data-by-activity.png)


3. Histograms of the body gyroscope data by activity

![img3 gyro](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2018/09/Histograms-of-the-body-gyroscope-data-by-activity.png)


<hr><br>
## 9. Approach to Modeling
In this section, we summarize some approaches to modeling the activity recognition dataset.

These ideas are divided into the main themes of a project.

### Problem Framing

The first important consideration is the framing of the prediction problem.

The framing of the problem as described in the original work is the prediction of activity for a new subject given their movement data, based on the movement data and activities of known subjects.

We can summarize this as:

Predict activity given a window of movement data.
This is a reasonable and useful framing of the problem.

Some other possible ways to frame the provided data as a prediction problem include the following:

* Predict activity given a time step of movement data.
* Predict activity given multiple windows of movement data.
* Predict the activity sequence given multiple windows of movement data.
* Predict activity given a sequence of movement data for a pre-segmented activity.
* Predict activity cessation or transition given a time step of movement data.
* Predict a stationary or non-stationary activity given a window of movement data

