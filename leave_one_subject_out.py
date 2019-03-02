import numpy as np
from datetime import datetime
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import discriminant_analysis
from sklearn import svm
from sklearn import linear_model
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneGroupOut
from sklearn import neighbors
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

class LeaveOneOutClassifiers:
    """
    Class containing methods for different ML classifiers
    @author Ramesh Paudel
    """
    K_FOLD = 10

    ALGORITHMS = {
        "ann":"ann",
        "svm":"svm",
        "random_forest":"random_forest",
        "dt":"dt",
        "ada":"ada",
        "knn":"knn",
        "lreg":"lreg",
        "log":"log",
        "osvm":"osvm",
        "lda":"lda",
        "qda":"qda"
    }
    def read_dataset_one(self,filename):
        data_file = pd.read_csv(filename)
        encoded_data = data_file.apply(LabelEncoder().fit_transform)
        #input_cols = ['month', 'action', 'followed_by', 'preceed_by', 'duration', 'hr_of_day', 'start_pt', 'end_pt', 'start_time', 'end_time', 'is_weekend']
        input_cols = ['month', 'action', 'preceed_by', 'followed_by', 'light_sensor_count', 'time_to_next_activity',
                      'duration', 'hr_of_day', 'light_on_count', 'motion_sensor_activated', 'light_off_count',
                      'start_pt', 'end_pt', 'start_time', 'end_time', 'day_of_week', 'is_weekend']

        group = encoded_data['id']
        cv = LeaveOneGroupOut().split(encoded_data[input_cols], encoded_data['mci'], group)
        return encoded_data[input_cols], encoded_data['mci'], cv

    def read_dataset(self):
        data_file = pd.read_csv("Data/balanced.csv")
        encoded_data = data_file.apply(LabelEncoder().fit_transform)
        group = encoded_data['id']
        input_cols = ['month', 'action', 'preceed_by', 'followed_by',
                      'duration', 'hr_of_day',  'start_pt', 'end_pt', 'start_time', 'end_time', 'is_weekend', 'day_of_week',
                      'light_on_count', 'motion_sensor_activated', 'light_off_count', 'light_sensor_count', 'time_to_next_activity']


        cv = LeaveOneGroupOut().split(encoded_data[input_cols], encoded_data['mci'], group)
        return encoded_data[input_cols], encoded_data['mci'], cv

    def __init__(self):
        print "\n\nStarting Leave One Subject Out Prediction ------ "
        #self.X, self.Y, self.cv = self.read_dataset()
        #print self.cv

    def knn_classify(self):
        X, Y, cv = self.read_dataset()
        X_scaled = preprocessing.scale(X)
        #print self.X
        poly = preprocessing.PolynomialFeatures(degree=2, interaction_only=True)
        X_scaled = poly.fit_transform(X_scaled)
        neigh = neighbors.KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto')
        y_pred = cross_val_predict(neigh, X_scaled, Y, cv=10)
        print "KNN Classifier Report"
        print "===========================================\n"
        self.display_report(y_pred, Y, self.ALGORITHMS["knn"])

    def logistic_regression(self):
        X, Y, cv = self.read_dataset()
        log = linear_model.LogisticRegression()
        X_scaled = preprocessing.scale(X)
        poly = preprocessing.PolynomialFeatures(degree=2, interaction_only=True)
        X_scaled = poly.fit_transform(X_scaled)
        print X_scaled.shape
        before = datetime.now()
        print before
        predicted = cross_val_predict(log, X_scaled, Y, cv=10)
        after = datetime.now()
        print after
        runtime = (after-before).total_seconds()
        print "Time: ", runtime
        print "Logistic Regression Report"
        print "===========================================\n"
        self.display_report(predicted, Y, self.ALGORITHMS["log"])

    def linear_regression_duration(self):
        lr = linear_model.LogisticRegression()
        # Get Model 1
        data, target, cv = self.read_dataset_one("Data/finalsensor.csv")
        print "--------- Model 1 Y~ ",list(data)
        data = preprocessing.scale(data)
        lr.fit(data, target)
        predicted = cross_val_predict(lr, data, target, cv=cv)
        print "R- Square, Adjusted R-Square ", lr.score(data, target), 1 - (1 - lr.score(data, target)) * (
        len(target) - 1) / (len(target) - data.shape[1] - 1)

        # Get Model 2
        data, target, cv = self.read_dataset_one("Data/finalsensor.csv")
        data = preprocessing.scale(data)
        print data.shape
        poly = preprocessing.PolynomialFeatures(degree=2, interaction_only= True)
        data = poly.fit_transform(data)
        print "--------- Model 2 Y~ -------", poly.get_feature_names()
        print data.shape
        lr.fit(data, target)
        predicted = cross_val_predict(lr, data, target, cv=cv)
        # print predicted
        print "R- Square, Adjusted R-Square ", lr.score(data, target), 1 - (1 - lr.score(data, target)) * (
            len(target) - 1) / (len(target) - data.shape[1] - 1)

        # Get Model 3
        data, target, cv = self.read_dataset_one("Data/finalsensor.csv")
        data = preprocessing.scale(data)
        print data.shape
        poly = preprocessing.PolynomialFeatures(degree=2)
        data = poly.fit_transform(data)
        print "--------- Model 3 Y~ -------", poly.get_feature_names()
        print data.shape
        lr.fit(data, target)
        predicted = cross_val_predict(lr, data, target, cv=cv)
        # print predicted
        print "R- Square, Adjusted R-Square ", lr.score(data, target), 1 - (1 - lr.score(data, target)) * (
            len(target) - 1) / (len(target) - data.shape[1] - 1)

        # Get Model 4
        data, target, cv = self.read_dataset_one("Data/finalsensor.csv")
        data = preprocessing.scale(data)
        print data.shape
        poly = preprocessing.PolynomialFeatures(degree=3, interaction_only=True)
        data = poly.fit_transform(data)
        print "--------- Model 4 Y~ -------", poly.get_feature_names()
        print data.shape
        lr.fit(data, target)
        predicted = cross_val_predict(lr, data, target, cv=cv)
        # print predicted
        print "R- Square, Adjusted R-Square ", lr.score(data, target), 1 - (1 - lr.score(data, target)) * (
            len(target) - 1) / (len(target) - data.shape[1] - 1)

        # Get Model 5
        data, target, cv = self.read_dataset_one("Data/finalsensor.csv")
        data = preprocessing.scale(data)
        print data.shape
        poly = preprocessing.PolynomialFeatures(degree=3)
        data = poly.fit_transform(data)
        print "--------- Model 5 Y~ -------", poly.get_feature_names()
        print data.shape
        lr.fit(data, target)
        predicted = cross_val_predict(lr, data, target, cv=cv)
        # print predicted
        print "R- Square, Adjusted R-Square ", lr.score(data, target), 1 - (1 - lr.score(data, target)) * (
            len(target) - 1) / (len(target) - data.shape[1] - 1)

    #def linear_regression_plot(self):


    def linear_regression(self):
        lr = linear_model.LinearRegression()

        #Get Model 1
        '''print "--------- Model 1 Y~ Id+Date+Action+Wknd+PointofDay -------"
        data, target = self.get_model1("papermodel.csv")
        data = preprocessing.scale(data)
        #poly = preprocessing.PolynomialFeatures(degree=5,interaction_only=True)
        #out = poly.fit_transform(data)
        lr.fit(data, target)
        predicted = cross_val_predict(lr, data, target, cv=self.cv)
        #print predicted

        # The coefficients
        print('Coefficients: \n', lr.coef_)
        # The mean squared error
        print("Mean squared error: %.2f"
              % mean_squared_error(target, predicted))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % r2_score(target, predicted))

        fig, ax = plt.subplots()
        ax.scatter(target, predicted, edgecolors=(0, 0, 0))
        ax.plot([target.min(), target.max()], [target.min(), target.max()], 'k--', lw=4)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        plt.show()'''

        # Plot outputs
        '''plt.scatter(data, target, color='black')
        plt.plot(data, target, color='blue', linewidth=3)

        plt.xticks(())
        plt.yticks(())

        plt.show()

        print "R- Square, Adjusted R-Square ", lr.score(data,target), 1 - (1-lr.score(data, target))*(len(target)-1)/(len(target)-data.shape[1]-1)
        '''
        # Get Model 2
        '''print "--------- Model 2 Y~ Id+Action+Wknd+PointofDay -------"
        data, target = self.get_model2("papermodel.csv")
        data = preprocessing.scale(data)
        lr.fit(data, target)
        predicted = cross_val_predict(lr, data, target, cv=self.cv)
        # print predicted
        print "R- Square, Adjusted R-Square ", lr.score(data, target), 1 - (1 - lr.score(data, target)) * (
        len(target) - 1) / (len(target) - data.shape[1] - 1)


        # Get Model 3
        print "--------- Model 3 Y~ Id+Action+PointofDay -------"
        data, target = self.get_model3("papermodel.csv")
        data = preprocessing.scale(data)
        lr.fit(data, target)
        predicted = cross_val_predict(lr, data, target, cv=self.cv)
        # print predicted
        print "R- Square, Adjusted R-Square ", lr.score(data, target), 1 - (1 - lr.score(data, target)) * (
            len(target) - 1) / (len(target) - data.shape[1] - 1)

        # Get Model 4
        print "--------- Model 4 Y~ Id+Action+wknd -------"
        data, target = self.get_model4("papermodel.csv")
        data = preprocessing.scale(data)
        lr.fit(data, target)
        predicted = cross_val_predict(lr, data, target, cv=self.cv)
        # print predicted
        print "R- Square, Adjusted R-Square ", lr.score(data, target), 1 - (1 - lr.score(data, target)) * (
            len(target) - 1) / (len(target) - data.shape[1] - 1)'''

        # Get Model 5
        print "--------- Model 5 Y~ IC+ IC^2 + Id*date + Id*PointofDay + Date + Action + Wknd + PointofDay -------"
        data, target = self.get_model5("papermodel.csv")
        data = preprocessing.scale(data)
        lr.fit(data, target)
        predicted = cross_val_predict(lr, data, target, cv=self.cv)
        fig, ax = plt.subplots()
        ax.scatter(target, predicted, edgecolors=(0, 0, 0))
        ax.plot([target.min(), target.max()], [target.min(), target.max()], 'k--', lw=4)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        plt.show()
        print "R- Square, Adjusted R-Square ", lr.score(data, target), 1 - (1 - lr.score(data, target)) * (
            len(target) - 1) / (len(target) - data.shape[1] - 1)

        #self.display_report(predicted, self.ALGORITHMS["lreg"])
        #print confusion_matrix(target, predicted)'''

        #print predicted
        '''fig, ax = plt.subplots()
        ax.scatter(target, predicted, edgecolors=(0, 0, 0))
        ax.plot([target.min(), target.max()], [target.min(), target.max()], 'k--', lw=4)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        plt.show()'''

    def linear_discriminative_analysis(self):
        X, Y, cv = self.read_dataset()
        lda = discriminant_analysis.LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
        X_scaled = preprocessing.scale(X)
        # cross_val_predict returns an array of the same size as `y` where each entry
        # is a prediction obtained by cross validation:
        poly = preprocessing.PolynomialFeatures(degree=2, interaction_only=True)
        X_scaled = poly.fit_transform(X_scaled)
        before = datetime.now()
        print before
        predicted = cross_val_predict(lda, X_scaled, Y, cv=10)
        after = datetime.now()
        print after
        print "LDA Classifier Report"
        print "===========================================\n"
        self.display_report(predicted, Y, self.ALGORITHMS["lda"])
        runtime = (after - before).total_seconds()
        print "Time: "
        print runtime
        #print confusion_matrix(self.Y, predicted)

    def quardatic_discriminative_analysis(self):
        X, Y, cv = self.read_dataset()
        qda = discriminant_analysis.QuadraticDiscriminantAnalysis()
        X_scaled = preprocessing.scale(X)
        # cross_val_predict returns an array of the same size as `y` where each entry
        # is a prediction obtained by cross validation:
        #poly = preprocessing.PolynomialFeatures(degree=2, interaction_only=True)
        #X_scaled = poly.fit_transform(X_scaled)
        predicted = cross_val_predict(qda, X_scaled, Y, cv=cv)
        print "QDA Classifier Report"
        print "===========================================\n"
        self.display_report(predicted, Y, self.ALGORITHMS["qda"])
        # print confusion_matrix(self.Y, predicted)

    def svm_classify(self):
        X, Y, cv = self.read_dataset()
        """
        Uses Support Vector Machine Classifier
        :return: None
        """
        #for kernel in svm_kernels:
        X_scaled = preprocessing.scale(X)
        poly = preprocessing.PolynomialFeatures(degree=2, interaction_only=True)
        X_scaled = poly.fit_transform(X_scaled)

        svm_classifier = svm.SVC(kernel="rbf")
        before = datetime.now()
        print before
        print X_scaled.shape
        svm_output = cross_val_predict(svm_classifier, X_scaled, Y, cv=10)
        after = datetime.now()
        print after
        print "Support Vector Machine Cross Val"
        print "===========================================\n"
        self.display_report(svm_output, Y, self.ALGORITHMS["svm"])
        runtime = (after - before).total_seconds()
        print "Time: "
        print runtime

    def svm_classify_cv(self):
        X, Y, cv = self.read_dataset()
        """
        Uses Support Vector Machine Classifier
        :return: None
        """
        #for kernel in svm_kernels:
        X_scaled = preprocessing.scale(X)
        #poly = preprocessing.PolynomialFeatures(degree=2, interaction_only=True)
        #X_scaled = poly.fit_transform(X_scaled)

        svm_classifier = svm.SVC(kernel="rbf")
        before = datetime.now()
        print before
        print X_scaled.shape
        svm_output = cross_val_predict(svm_classifier, X_scaled, Y, cv=cv)
        after = datetime.now()
        print after
        print "Support Vector Machine Leave out"
        print "===========================================\n"
        self.display_report(svm_output, Y, self.ALGORITHMS["svm"])
        runtime = (after - before).total_seconds()
        print "Time: "
        print runtime

    def svm_classify_interaction(self):
        X, Y, cv = self.read_dataset()
        """
        Uses Support Vector Machine Classifier
        :return: None
        """
        #for kernel in svm_kernels:
        X_scaled = preprocessing.scale(X)
        poly = preprocessing.PolynomialFeatures(degree=2, interaction_only=True)
        X_scaled = poly.fit_transform(X_scaled)

        svm_classifier = svm.SVC(kernel="rbf")
        before = datetime.now()
        print before
        print X_scaled.shape
        svm_output = cross_val_predict(svm_classifier, X_scaled, Y, cv=10)
        after = datetime.now()
        print after
        print "Support Vector Machine Degree 2 Cross Val"
        print "===========================================\n"
        self.display_report(svm_output, Y, self.ALGORITHMS["svm"])
        runtime = (after - before).total_seconds()
        print "Time: "
        print runtime

    def svm_classify_interaction_cv(self):
        X, Y, cv = self.read_dataset()
        """
        Uses Support Vector Machine Classifier
        :return: None
        """
        #for kernel in svm_kernels:
        X_scaled = preprocessing.scale(X)
        poly = preprocessing.PolynomialFeatures(degree=2, interaction_only=True)
        X_scaled = poly.fit_transform(X_scaled)

        svm_classifier = svm.SVC(kernel="rbf")
        before = datetime.now()
        print before
        print X_scaled.shape
        svm_output = cross_val_predict(svm_classifier, X_scaled, Y, cv=cv)
        after = datetime.now()
        print after
        print "Support Vector Machine Degree 2 CV"
        print "===========================================\n"
        self.display_report(svm_output, Y, self.ALGORITHMS["svm"])
        runtime = (after - before).total_seconds()
        print "Time: "
        print runtime

    def random_forest_classify(self):
        X, Y, cv = self.read_dataset()
        X_scaled = preprocessing.scale(X)
        poly = preprocessing.PolynomialFeatures(degree=2, interaction_only= True)
        X_scaled = poly.fit_transform(X_scaled)
        random_forest_classifier = RandomForestClassifier(n_estimators=10, criterion="gini")
        classification = cross_val_predict(random_forest_classifier,X_scaled, Y, cv=10)
        print "Random Forest Classifier Report"
        print "===========================================\n"
        self.display_report(classification, Y, self.ALGORITHMS["random_forest"])

    def decision_tree_classify(self):
        X, Y, cv = self.read_dataset()
        X_scaled = preprocessing.scale(X)
        poly = preprocessing.PolynomialFeatures(degree=2, interaction_only=True)
        X_scaled = poly.fit_transform(X_scaled)
        decision_tree_classifier = DecisionTreeClassifier()
        classification = cross_val_predict(decision_tree_classifier, X_scaled, Y, cv=10)
        print "Decision Tree Classifier Report"
        print "===========================================\n"
        print classification
        self.display_report(classification,Y, self.ALGORITHMS["dt"])

    def ada_boost_classify(self):
        X, Y, cv = self.read_dataset()
        X_scaled = preprocessing.scale(X)
        poly = preprocessing.PolynomialFeatures(degree=2, interaction_only=True)
        X_scaled = poly.fit_transform(X_scaled)
        ada_boost_classifier = AdaBoostClassifier()
        classification = cross_val_predict(ada_boost_classifier, X_scaled, Y, cv=10)
        print "Ada Boost Classifier Report"
        print "===========================================\n"
        self.display_report(classification, Y, self.ALGORITHMS["ada"])

    def display_report(self, prediction, Y, algorithm):
        """
        Displays the classification report
        :param prediction: Prediction of the model
        :param algorithm: Algorithm used to generate predictions
        :return: None
        """
        print "Confusion Matrix: "
        print "------------------\n"
        self.plot_confusion_matrix(confusion_matrix(Y, prediction),algorithm)

        print "\nClassification Report: "
        print "-----------------------"
        print("accuracy: ", metrics.accuracy_score(Y, prediction))
        print("precision: ", metrics.precision_score(Y, prediction))
        print("recall: ", metrics.recall_score(Y, prediction))
        print("f1: ", metrics.f1_score(Y, prediction))
        print("area under curve (auc): ", metrics.roc_auc_score(Y, prediction))
        print classification_report(Y,prediction)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(Y, prediction)
        print ("(auc): ", auc(false_positive_rate, true_positive_rate))

    def plot_confusion_matrix(self, con_mat, algorithm,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.

        :param con_mat: confusion matrix
        :param algorithm: algorithm used
        :param normalize: boolean to normalize or not
        :param title: title of the graph
        :param cmap: map color palate
        :return: None
        """


        plt.figure()
        plt.imshow(con_mat, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["mci","healthy"], rotation=45)
        plt.yticks(tick_marks, ["mci","healthy"])

        if normalize:
            con_mat = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(con_mat)

        thresh = con_mat.max() / 2.
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            plt.text(j, i, con_mat[i, j],
                     horizontalalignment="center",
                     color="white" if con_mat[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig("figures/CASAS_CM_"+algorithm+".png")

