**Requirements**
=============

1. python (3.0)
2. Python packages:
	a. matplotlib
	b. scikit-learn
	c. pandas
	d. csv


**Usage**
======

$ make

if Make is not installed
------------------------
$ python main.py


**Notes**
=====

1. Make sure "Data" folder have 10 text file  <id>.txt
2. Make sure you have "figures" folder
3. This code implements following paper: <br/>
    Paudel R, Dunn K, Eberle W, and Chaung D, [Cognitive Health Prediction on the Elderly Using Sensor Data in Smart Homes](https://aaai.org/ocs/index.php/FLAIRS/FLAIRS18/paper/view/17622/16833). (FLAIRS-31)

**Description**
This code first parse the sensor log file (1.txt, 2.txt ... 10.txt) for 10 different patient.
These sensor log files are in "Data" folder.
It then run following machine learning algorithms.
+ Logistic Regression
+ Linear Discriminant analysis (LDA) • Decision Tree
+ Support Vector Machine (SVM)
+ K-Nearest Neighbor (KNN)
+ Random Forest
+ Ada Boosting
+ One-Class SVM



