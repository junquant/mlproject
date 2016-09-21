# python3.5.2

from datetime import datetime
import os, sys, traceback

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

# Start Code
try:
    timer = Timer()
    print('Start Time : ', timer.getTime())  # Get the start time for tracking purposes

    for subdir, dirs, files in os.walk(inputDir):  # Walk through each file in the dir
        for file in files:

            inputFilePath = os.path.join(subdir, file)
            inputFileName = os.path.splitext(file)[0]
            inputFileExt = os.path.splitext(file)[1]

            print('Reading data file : ', inputFilePath)  # Opening file ...

            # Open input file

            # Check for .dat extension

            # Open an output file with extension .txt

            # read all protocol files 1 by 1 and append to output file

            # save output file

            # close input file
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
    # close all input files
    # close all output files

    print('End Time : ',timer.getTime())