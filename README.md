# Human Activity Data
### this project is about cleaning and tidying a dataset which is collected from wearable gadgets (samsung S2)

this project is an assignment of ***Getting and Cleaning Data*** course of [Coursera](https://www.coursera.org) created by ***Johns Hopkinz University***

here is the explanation of project :

Getting and Cleaning Data Course Project: 
The purpose of this project is to demonstrate your ability to collect, work with, and clean a data set. The goal is to prepare tidy data that can be used for later analysis. You will be graded by your peers on a series of yes/no questions related to the project. You will be required to submit: 
1. a tidy data set as described below, 
2. a link to a Github repository with your script for performing the analysis, and 
3. a code book that describes the variables, the data, and any transformations or work that you performed to clean up the data called CodeBook.md. You should also include a README.md in the repo with your scripts. This repo explains how all of the scripts work and how they are connected.

One of the most exciting areas in all of data science right now is wearable computing - see for example this article . Companies like Fitbit, Nike, and Jawbone Up are racing to develop the most advanced algorithms to attract new users. The data linked to from the course website represent data collected from the accelerometers from the Samsung Galaxy S smartphone. A full description is available at the site where the data was obtained:

http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

Here are the data for the project:

https://d396qusza40orc.cloudfront.net/getdata%2Fprojectfiles%2FUCI%20HAR%20Dataset.zip
<br/><hr/>
You should create one R script called run_analysis.R that does the following.

1. Merges the training and the test sets to create one data set.
2. Extracts only the measurements on the mean and standard deviation for each measurement.
3. Uses descriptive activity names to name the activities in the data set
4. Appropriately labels the data set with descriptive variable names.
5. From the data set in step 4, creates a second, independent tidy data set with the average of each variable for each activity and each subject.




<hr/>

## So the way I did this project:
1. setting up folders and exploring files [some did it inside code but I did it by extracting data and by windows]
2. import datasets; train , test , feature , subject
3. merging data using ``` rbind() cbind() ```
4. filtering data catagories by ***std*** and ***mean*** and relabeling all variable names
5. grouping and summarizing all data in means in 180 subject-acticity catagories.[30 subjects and 6 activities]


## documenting proccess :
1. setting up github reapo with a readme.md
2.commenting the whole code in an intuitive way
3. adding codebook which contains variable names and variables of [tidy_data.txt](https://github.com/rooholaminr/HumanActivityData/blob/master/tidy_data.txt)
4.updating readme 


### hope you find it useful
