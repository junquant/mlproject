# Report

## Problem Description

* Human Activity Recognition using Machine Learning techniques

Wearable technologies are getting more and more common and machine learning plays an important role in enabling the machine to recognize a person's activity. With some physical parameters measurement of human body, a computer is smart enough to tell the activity carried out by a person,i.e. sitting, running or climbing up stairs. While typical machine learning projects seeks to classify the activity performed, this project attempts to go a step further and aims to develop a method to best classify the activity and the person performing the activity. Using the T,P,E framework, the problem can be summarized into:

* **Task** - Predict the activity *and* the person performing the activity
* **Performance** - Number of actions *and* person performing the activity correctly classified
* **Experience** - PAMAP2 data set of labeled IMU readings available from the UCI Machine Learning Repository 

A few classifier models have been built and compared: Naive Bayes classifier, Support Vector Machine classifier, Logistic Regression classifier.

The constructed model may be deployed in a health care setting, to monitor a patient's activities. With the ability to differentiate the activities and the person performing it, it also opens up the possibility of sharing the wearable device.

## Data Set

The PAMAP2 data set available from UCI Machine Learning Repository [(Link)](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring) consist the data of 9 subjects performing 18 different physical activities such as sitting, lying, walking and running in a controlled environment - all went through the exact same sequence of activities with optional activities performed by some of the subjects. Optional activities performed but will not be in the scope of this project.  The subjects (8 males, 1 female – ages: 27-31 years old) who participated were mainly employees or students at the German Research Center for Artificial Intelligence (DFKI). The data collection was done by attaching 3 Colibri wireless inertial measurement units (IMUs) to each subject's body. 

The full data set exists in 9 separate .dat files, one for each subject, of 54 columns containing data such as Timestamps, Activity IDs, Heart Rate and IMU readings on the subjects' hand, chest and ankle. Each IMU sensory data (hand, chest and ankle) contains the temperature, 3D acceleration data, 3D gyroscope data, 3D magnetometer data. Orientation data is also included in the data set but invalid in this data set as mentioned in the code book available with the PAMAP2.

## Data Preparation
**Missing Values**
The missing values were caused by lost of signals. As such, missing values are populated with the last valid value for the subject and if there is no valid value before, the first valid value after the record was used. 

## Methodology

## Results

## Summary

## Discussion


---

Things that came up during discussions with prof

Potential Techniques
* Potential techniques not required in proposal yet
* Principle Component Analysis
* Decision Trees (may some ensemble?)
* SVM ?

Things to do 
* Predict the person
* Predict the activity
* Make the prediction 2 layers 
* first layer to classify the person
* second layer to classify the activity
* can also do a joint classifier 
* Need to interpret classifier (eg. speed to determine whether person is walking or running)
* for eg. what dimension needs to 
