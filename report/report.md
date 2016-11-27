# Report


## Introduction

Wearable technologies are getting more and more common and machine learning plays an important role in enabling the machine to recognize a person's activity. With some physical parameters measurement of human body, a computer is smart enough to tell the activity carried out by a person,i.e. sitting, running or climbing up stairs. The constructed model may be deployed in a health care setting, to monitor a patient's activities. With the ability to differentiate the activities and the person performing it, it also opens up the possibility of sharing the wearable device. Typical machine learning projects seeks to classify the activity performed, this project attempts to go a step further and aims to develop a method to best classify the activity and the person performing the activity. Using the T,P,E framework, the problem can be summarized into:

* **Task** - Predict the activity *and* the person performing the activity
* **Performance** - Percentage of actions *and* person performing the activity correctly classified
* **Experience** - PAMAP2 data set of labeled IMU readings available from the UCI Machine Learning Repository 

## Objective

The objective of this project is to evaluate empirically the performance of various machine learning algorithms in terms of the time taken to train the model, accuracy, precision and recall. The project also aims to empirically evaluate the performance of methodology used to predict both the activity and subject. For the above objectives, a few classifier models have been built and compared. They are namely, the Naive Bayes classifier, Support Vector Machine classifier, Logistic Regression classifier.

## About the Data Set

The PAMAP2 data set available from UCI Machine Learning Repository [(Link)](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring) consist the data collected from 9 subjects(persons).

These 9 subjects are mainly employees or students at DFKI and consists of
* 1 female, 8 males
* aged 27.22 ± 3.31 years
* BMI 25.11 ± 2.62 kgm<sup>-2</sup>

Each subject performs 12 different physical activities such as sitting, lying, walking and running in a controlled environment - all went through the exact same sequence of activities. In addition, 6 optional activities were performed by some of the subjects. In this project optional activities will be out of the scope.

The 12 activities in scope for this projects are:
* Lying
* Sitting
* Standing
* Ironing
* Vacuuming
* Walking upstairs
* Walking downstairs
* Normal walk
* Nordic walk
* Cycling
* Running
* Rope Jumping

It is noted that all above 12 activities are the most commonly daily activities, except Nordic walk which requires a person to walk with specially designed walking poles. The data are collected via a heart rate monitor and 3 Colibri wireless inertial measurement units (IMUs) attached to each subject's body: one over the wrist, one on the chest and one on ankle.

The heart rate monitor has sampling rate of 9Hz and each IMU generate following data with 17 columns:

* 1 temperature (°C)
* 2-4 3D-acceleration data (ms<sup>-2</sup>), scale: ±16g, resolution: 13-bit
* 5-7 3D-acceleration data (ms<sup>-2</sup>), scale: ±6g, resolution: 13-bit
* 8-10 3D-gyroscope data (rad/s)
* 11-13 3D-magnetometer data (μT)
* 14-17 orientation

The full data set exists in 9 separate .dat files, one for each subject, of 54 columns (17 columns x 3 IMU + Heart Rate + Activity ID + Time Stamp). Orientation data is also included in the data set but invalid in this data set as mentioned in the code book available with the PAMAP2.

## Data Preparation

**Missing Values**
The missing values were caused by lost of signals. As such, missing values are populated with the last valid value for the subject and if there is no valid value before, the first valid value after the record was used. This was done for each subject's data. 

**Invalid Data**
Orientation is not valid in this data set as stated in the codebook and was removed. Accelerometer data for with the scale of ±6g was also removed from the data set as recommended by the codebook as readings are saturated for high impact movements such as running. 

## Approach

The approach that we propose would be to first explore in detail to extract the features most representative of the activities and the subject. Next, 3 different models will be compared in the classification of human activity and the person performing it. The 3 models that will be compared are summarised as follows:

1. **Model 1** - Classify Subject (Person) --> Feed subject back into model to classify action of the subject
2. **Model 2** - Classify Action --> Feed action back into model to the classify subject
3. **Model 3** - Classify both subject and action simultaneously

The most suitable model (in terms of accuracy, precision and recall) to classify an activity that is carried out by a unique individual will be selected. Supervised learning methods will explored and used to construct the model. The model will then be interpreted to extract insights on how are the actions and subjects classified. 

Hold-out and k-fold cross validation were used for model validation. Source control will be done using Github. 

## Interpreting the Results


## Conclusion


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
