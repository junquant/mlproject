# Proposal

## Problem Description

This project seeks to compare 3 types of classification methodologies in the classification of human activity based on the PAMAP2 data retrieved from the UCI Machine Learning Repository. The eventual outcome is to deduce the most suitable method of predicting an action that is carried out by a unique individual who was part of the group of test subjects.

The 3 methodologies that will be compared can be summarised as follows:

1. Classify subject --> Given subject, classify action of the subject
2. Classify action --> Given action, classify subject
3. Classify both subject and action simultaneously

## Data Set

The Physical Activity Monitoring for Aging People (PAMAP) organisation collected the data of 9 subjects. The 9 subjects (8 males, 1 female) who participated were mainly employees or students at the German Research Center for Artificial Intelligence (Deutsches Forschungszentrum für Künstliche Intelligenz, DFKI). They are aged between 27 and 31 years old and have BMIs between 25kgm^-2 and 28kgm^-2. 

The 9 subjects carried out 18 different physical activities in a controlled environment - all subjects went through the exact same sequence of activities. Data collection was done by attaching 3 [Colibri wireless inertial measurement units (IMUs)] (http://www.trivisio.com/index.php/products/motiontracking/colibriwireless) to 3 main parts of each subject's body - the wrist of the dominant arm, the chest, and the ankle of the dominant leg.

The data collected can be classified in the following categories:

Column # | Data
--- | ---
1 | Timestamp (s)
2 | Activity ID
3 | Heart rate (bpm)
4 - 20 | IMU Hand
21 - 37 | IMU Chest
38 - 54 | IMU ankle

Each IMU sensory data (hand, chest and ankle) contains the following:
* Temperature
* 3D acceleration data
* 3D gyroscope data
* 3D magnetometer data
* Orientation (this is invalid in this dataset)

These activities can be classified as follows:

ID | Activity
--- | ---
1 | Lying
2 | Sitting
3 | Standing
4 | Walking
5 | Running
6 | Cycling
7 | Nordic walking
9 | Watching TV
10 | Computer work
11 | Car driving
12 | Ascending stairs
13 | Descending stairs
16 | Vacuum cleaning
17 | Ironing
18 | Folding Laundry
19 | House cleaning
20 | Playing soccer
24 | Rope Jumping
0 | Other (Transient activities)

## Data Exploration & Preparation

The dataset exists in 9 separate .dat files, one for each subject. As such, a script was prepared to read the data and consolidate it into one .txt file for easier processing.

Initial exploration of the data set reveals several important points that will provide the basis of our data preparation methodology:

Exploration item | Remarks & Potential implication
---|---
Subject 9 has significantly lower amounts of recorded data than the other subjects | Including the subject will severely distort the analysis as he is an obvious outlier. As such, we exclude subject 9 from our analysis.
The dataset consists of a significant amount of NaN data | Cannot conduct PCA with missing values. There are missing values mainly because of technical issues and the fact that heartrate and other recorded variables were measured at different frequencies. As such, we have to handle the missing data.
Timestamps are available | Including the timestamp in the analysis will cause the model to be biased - it is possible to predict results solely based on timestamps. As such, we have to exclude timestamps.
Some activities were only carried out by a few subjects (e.g. watching TV was only carried out by 1 subject) | Activities carried out by less than 6 out of 8 subjects were left out in order to have enough data to train the model accurately
Plot correlation matrix ![Correlation Matrix](../report/img/correlation_matrix.png) |  3D Accelerators (16g) are highly correlated (postively) with 3D Acclerators (6g); 3D Magnetometer (Chest) highly correlated (negatively) with both 3D accelerometers (Chest)


## Potential Techniques
* Potential techniques not required in proposal yet
* Principle Component Analysis
* Decision Trees (may some ensemble?)
* SVM ?

## Things to do 
* Predict the person
* Predict the activity
* Make the prediction 2 layers 
* first layer to classify the person
* second layer to classify the activity
* can also do a joint classifier 
* Need to interpret classifier (eg. speed to determine whether person is walking or running)
* for eg. what dimension needs to 
