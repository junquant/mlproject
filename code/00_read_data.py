# python3.5.2

from datetime import datetime
import os, sys, traceback
import csv

# Utility class for timing the script
# --------------------------------------
class Timer:
    def __init__(self):
        pass

    def getTimeString(self, mydate):
        return str(mydate.strftime('%Y-%m-%d %H:%M:%S'))

    def getTime(self):
        return datetime.now()

print('------------------------------------------------------------')
print('Reading files ... ')
timer = Timer()
startTime = timer.getTime()
print('------------------------------------------------------------')

# Parameters
inputDir  = '../data/protocol/'
outputDir = '../data/'

activityCol = 1
activities = [1,2,3,4,5,6,7,12,13,16,17]

# 1 – lying (# subj - 8)
# 2 – sitting (# subj - 8)
# 3 – standing (# subj - 8)
# 4 – walking (# subj - 8)
# 5 – running (# subj - 6)
# 6 – cycling (# subj - 7)
# 7 – Nordic walking (# subj - 7)
# 9 – watching TV (# subj - 1)
# 10 – computer work  (# subj - 4)
# 11 – car driving (# subj - 1)
# 12 – ascending stairs  (# subj - 8)
# 13 – descending stairs  (# subj - 8)
# 16 – vacuum cleaning  (# subj - 8)
# 17 – ironing (# subj - 8)
# 18 – folding laundry  (# subj - 4)
# 19 – house cleaning  (# subj - 5)
# 20 – playing soccer (# subj - 2)
# 24 – rope jumping (# subj - 6)

# Counters
readRows = 0
excludedRows = 0
includedRows = 0

# Start Code

# Create column names
columnNames = ['timestamp', 'activity_id', 'heartrate_bpm']

imuColumns = ['%s_temp_c', '%s_3d_accel_16_1', '%s_3d_accel_16_2', '%s_3d_accel_16_3',
              '%s_3d_accel_6_1','%s_3d_accel_6_2', '%s_3d_accel_6_3',
              '%s_3d_gyroscope_1', '%s_3d_gyroscope_2', '%s_3d_gyroscope_3',
              '%s_3d_magnetometer_1','%s_3d_magnetometer_2', '%s_3d_magnetometer_3',
              '%s_orientation_1', '%s_orientation_2','%s_orientation_3','%s_orientation_4']

# Prefixes to be appended to imuColumns
prefix = ['hand', 'chest', 'ankle']

# Append prefixes to columns
for l in range(0, 3):
    for imu in imuColumns:
        imu = str(imu) % prefix[l]
        columnNames.append(imu)

columnNames.append('subject')

try:
    timer = Timer()
    print('Start Time : ', timer.getTime())  # Get the start time for tracking purposes

    # Open output path for writing
    outputFile = open(outputDir + 'consolidated.txt', 'wt')
    dataWriterOut = csv.writer(outputFile, delimiter=",", quoting=csv.QUOTE_NONE)
    dataWriterOut.writerow(columnNames)

    # Read each file in the input folder
    for subdir, dirs, files in os.walk(inputDir):  # Walk through each file in the dir
        for file in files:

            inputFilePath = os.path.join(subdir, file)
            inputFileName = os.path.splitext(file)[0]
            inputFileExt = os.path.splitext(file)[1]

            # Check for .dat extension
            if inputFileExt != '.dat': continue

            # Open input file
            print('Reading data file : ', inputFilePath)  # Opening file ...

            inputFile = open(inputFilePath,'r')
            dataReader = csv.reader(inputFile, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)

            # read all protocol files 1 by 1 and append to output file
            for line in dataReader:
                readRows += 1
                if line[activityCol] not in activities:
                    excludedRows += 1
                    continue
                else:
                    # Remove orientation (invalid in this PAMAP 2 data set)
                    line = line[0:50]

                    # Append subject
                    subjectId = int(inputFileName[-3:])
                    line.append(subjectId)

                    # write line to file
                    dataWriterOut.writerow(line)

                    # uncomment if only sampling 10 rows for testing
                    # if includedRows == 10:
                    #     break
                    #
                    includedRows += 1

                if readRows % 100000 == 0:
                    print('Read records : %s | Included records : %s | Excluded records : %s' %
                          (str(readRows), str(includedRows), str(excludedRows)))

            # uncomment break just to read subject101 file
            # break

            # close input file
            inputFile.close()

except:
    error_type = sys.exc_info()[0]
    error_message = sys.exc_info()[1]
    error_line = sys.exc_info()[2].tb_lineno

    print('******************************************************')
    print('traceback information')
    print('******************************************************')
    traceback.print_tb(sys.exc_info()[2])
    print('******************************************************')
    print('error information')
    print('******************************************************')
    print(("Error | %s - %s | Line %s" % (error_type, error_message, error_line)))
    print('******************************************************')


finally:
    # close all output files
    outputFile.close()

    print('Total Records Read: ', str(readRows))
    print('Total Records Written: ', str(includedRows))
    print('Total Records Excluded: ', str(excludedRows))
    print('End Time : ',timer.getTime())