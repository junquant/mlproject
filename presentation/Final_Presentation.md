% Human Activity Recognition using Machine Learning techniques
% THAM JUN QUAN, THOMAS JIANG CHENYANG, AUSTIN ZHANG YI
% Dec 02, 2016

# Machine Learning -- T.P.E

- Task: Predict the activity and the person performing the activity
- Performance: Percentage of actions and person performing the activity correctly classified
- Experience: PAMAP2 data set of labeled IMU readings available from the UCI Machine Learning Repository

# PAMAP2 Data Set

- A Physical Activity Monitoring dataset

- 3 wireless inertial measurement units (IMU):
   - sampling frequency: 100Hz  on wrist, chest and ankle
   - Record temperature, acceleration, 3D-magnetometer data, 3D-gyroscope data, orientation...
   
- 1 heart rate monitor:
   - sampling frequency: ~9Hz

- Activities:
   - Lying, Sitting, Standing, Ironing, Vacuuming, Walking upstairs, Walking downstairs, Normal walk, Nordic walk, Cycling, Running

# Model Construction Methods

1. Classify Subject (Person) --> Feed subject back into model to classify action of the subject
2. Classify Action --> Feed action back into model to the classify subject
3. Classify both subject and action simultaneously

![The steps](../plots/Prediction_Steps.jpg?raw=true "Prediction Steps")

# Data Preparation


# Data Preparation

# Data Exploration