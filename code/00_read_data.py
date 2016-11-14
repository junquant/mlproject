# python3.5.2
import os, sys, traceback
import csv
from code_utilities.custom_utilities import Timer, MetaData

print('------------------------------------------------------------')
print('Reading files ... ')
timer = Timer()
startTime = timer.getTime()
print('------------------------------------------------------------')

# Parameters
inputDir  = '../data/protocol/'
outputDir = '../data/'

# Get the data file stuff from DataFile object in custom_utilities:
metadata = MetaData()
activityCol = metadata.activityCol
activities = metadata.activities
columnNames = metadata.getRawColumnNames()

# Counters
readRows = 0
excludedRows = 0
includedRows = 0

# Start Code
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