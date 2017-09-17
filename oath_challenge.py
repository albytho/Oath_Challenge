import scipy.io.wavfile
import math
from pylab import*
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import csv
from sklearn import svm

#The sampFreq and rms_val of each file
train_x = []

#The instrument of each file
train_y = []

#Each file's file name.  This is used to write out the result in excel
filenames = []

#Goes through all the training data
for filename in os.listdir("/Users/Alby/Desktop/yahoo_challenge/train_data"):
	
	#Calculates the frequency and rms_val of the specific wav file and puts thoese to in an array called data. 
	sampFreq, values = scipy.io.wavfile.read("/Users/Alby/Desktop/yahoo_challenge/train_data/" + filename)
	values = values / (2.**15)
	rms_val = sqrt(mean(values**2))

	#Appends the filename into the filenames array to keep track of which instrument goes with which file
	#Also deletes the last 7 characters of the filename string so that we are left with just the instrument name
	filenames.append(filename)
	instrument = filename[:-7]

	#Appends the instrument name into an array as well as its corresponding rsm and frequency
	train_y.append(instrument)
	train_x.append([sampFreq,rms_val])


#Use a random forest classification model to predict what each instrument is being played
clf = RandomForestClassifier(n_estimators=1000)
clf = clf.fit(train_x, train_y)







test_filenames = []
test_x = []

for filename in os.listdir("/Users/Alby/Desktop/yahoo_challenge/test_data"):
	sampFreq_test, values_test = scipy.io.wavfile.read("/Users/Alby/Desktop/yahoo_challenge/test_data/" + filename)
	values_test = values_test / (2.**15)
	rms_val = sqrt(mean(values_test**2))
	test_filenames.append(filename)
	test_x.append([sampFreq_test,rms_val])

#Test the predictor and store the predictions into test_results
test_y = clf.predict(test_x)

#Go through and put the results into the .csv file
with open("Alby_Thomas.csv", "a") as resultFile:
	resultFileWriter = csv.writer(resultFile)
	for i in range(len(test_filenames)):
		resultFileWriter.writerow([test_filenames[i],test_y[i]])

#46% Prediction Accuracy when tested against 39 of the training files





