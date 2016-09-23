# Code Directory

This is the directory for all data processing and learning codes. Below is a description of what each file does. 

### 00_read_data.py
* Create the labels for the columns based on the codebook provided together with the PAMAP2 dataset
* Read the .dat files for all 9 subjects from the protocol folder. Data from the optional folder are excluded as not all subjects performed the activities. 
* Tag each record with the subject that perform the activity
* Output subject 101-109 into a single file (../data/consolidate.txt)
* Count the number of records read, included or excluded

### 01_cleaning.py
General data cleaning - e.g. missing data etc.

### 02_preprocessing.py
Data exploration, PCA, and other preprocessing methods
