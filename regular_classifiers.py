import numpy as np
from datetime import datetime
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc, mean_squared_error, r2_score
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
from sklearn import neighbors
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

class RegularClassifiers:
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
        #input_cols = ['age', 'start_time', 'end_time', 'duration', 'gender', 'month', 'action', 'preceed_by',
        #              'hr_of_day', 'start_pt']
        input_cols = ['month', 'action', 'followed_by', 'preceed_by', 'duration', 'hr_of_day', 'start_pt', 'end_pt', 'start_time', 'end_time', 'is_weekend']
        # print encoded_data['mci']
        return encoded_data[input_cols], encoded_data['mci']

    def read_dataset(self):
        data_file = pd.read_csv('preceedby.csv')
        encoded_data = data_file.apply(LabelEncoder().fit_transform)
        #input_cols = ['age', 'start_time', 'end_time', 'duration', 'gender', 'month', 'action', 'preceed_by','hr_of_day', 'start_pt']

        input_cols = ['duration', 'month', 'action', 'preceed_by', 'hr_of_day', 'start_pt', 'end_pt', 'start_time', 'end_time', 'is_weekend', 'followed_by']



        #input_cols = ['month', 'age', 'gender', 'action', 'followed_by','preceed_by','duration', 'hr_of_day', 'start_pt', 'end_pt', 'start_time', 'end_time','is_weekend']
        #input_cols = ['ic', 'ic_sq', 'id-date', 'id-pt', 'date', 'action', 'is_weekend', 'pt']
        #print encoded_data['mci']
        return  encoded_data[input_cols], encoded_data['mci']

    def get_model1(self, filename):
        data_file = pd.read_csv(filename)
        encoded_data = data_file.apply(LabelEncoder().fit_transform)
        input_cols = ['id', 'date','action','is_weekend','pt']
        return encoded_data[input_cols], encoded_data['mci']

    def get_model2(self, filename):
        data_file = pd.read_csv(filename)
        encoded_data = data_file.apply(LabelEncoder().fit_transform)
        input_cols = ['id','action','is_weekend','pt']
        return encoded_data[input_cols], encoded_data['mci']

    def get_model3(self, filename):
        data_file = pd.read_csv(filename)
        encoded_data = data_file.apply(LabelEncoder().fit_transform)
        input_cols = ['id', 'action', 'pt']
        return encoded_data[input_cols], encoded_data['mci']
    def get_model4(self, filename):
        data_file = pd.read_csv(filename)
        encoded_data = data_file.apply(LabelEncoder().fit_transform)
        input_cols = ['id', 'action', 'is_weekend']
        return encoded_data[input_cols], encoded_data['mci']

    def get_model5(self, filename):
        data_file = pd.read_csv('papermodel.csv')
        encoded_data = data_file.apply(LabelEncoder().fit_transform)
        input_cols = ['ic', 'ic_sq', 'id-date', 'id-pt', 'date', 'action', 'is_weekend', 'pt']
        return encoded_data[input_cols], encoded_data['mci']

    def __init__(self):
        print "\n\nStarting Regular Classifier Prediction ------ "
        self.X, self.Y = self.read_dataset()
        #print self.X

    def select_best_feature(self):
        # load the iris datasets
        X_scaled = preprocessing.scale(self.X)
        # create a base classifier used to evaluate a subset of attributes
        model = linear_model.LogisticRegression()
        # create the RFE model and select 3 attributes
        rfe = RFE(model, 8)
        rfe = rfe.fit(X_scaled, self.Y)
        # summarize the selection of the attributes
        print(rfe.support_)
        print(rfe.ranking_)


    def ann_classify(self, max_iteration=4000):
        """
        Uses Artificial Neural Network Classifier
        :return: None
        """
        input_feature_count = 5
        output_feature_count = 2
        models_for_selection = []
        for i in range(output_feature_count,(input_feature_count+1)):
            mlp = MLPClassifier(hidden_layer_sizes=(i,), solver="sgd", max_iter=max_iteration,
                                learning_rate_init=0.01, learning_rate="adaptive")
            models_for_selection.append(cross_val_predict(mlp,self.X, self.Y, cv=self.K_FOLD))

        print "Artificial Neural Network Classifier Report"
        print "===========================================\n"
        selected_model = self.select_best_model(models_for_selection)
        self.display_report(selected_model, self.ALGORITHMS["ann"])

    def knn_classify(self):
        X_scaled = preprocessing.scale(self.X)
        #print self.X
        poly = preprocessing.PolynomialFeatures(degree=3)
        data = poly.fit_transform(X_scaled)
        neigh = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
        y_pred = cross_val_predict(neigh, X_scaled, self.Y, cv=self.K_FOLD )
        print "KNN Classifier Report"
        print "===========================================\n"
        self.display_report(y_pred, self.ALGORITHMS["knn"])

    def logistic_regression(self):
        log = linear_model.LogisticRegression()
        X_scaled = preprocessing.scale(self.X)
        # cross_val_predict returns an array of the same size as `y` where each entry
        # is a prediction obtained by cross validation:
        poly = preprocessing.PolynomialFeatures(degree=3)
        X_scaled = poly.fit_transform(X_scaled)
        before = datetime.now()
        print before
        predicted = cross_val_predict(log, X_scaled, self.Y, cv=self.K_FOLD)
        after = datetime.now()
        print after
        runtime = (after-before).total_seconds()
        print "Logistic Regression Report"
        print "===========================================\n"
        self.display_report(predicted, self.ALGORITHMS["log"])
        print "Time: "
        print runtime

    def model(self,var):
        return 1 / (1 + np.exp(-var))

    def linear_regression_duration(self):
        lr = linear_model.LogisticRegression()
        # Get Model 1
        data, target = self.read_dataset_one("preceedby.csv")
        print "--------- Model 1 Y~ ",list(data)
        data = preprocessing.scale(data)
        lr.fit(data, target)
        predicted = cross_val_predict(lr, data, target, cv=self.K_FOLD)
        # and plot the result
        #plt.figure(1, figsize=(4, 3))
        #plt.clf()
        #plt.scatter(predicted, target, color='black', zorder=20)

        #loss = self.model(data * lr.coef_ + lr.intercept_).ravel()
        #plt.plot(predicted, loss, color='red', linewidth=3)
        #plt.show()
        #print loss
        print "R- Square, Adjusted R-Square ", lr.score(data, target), 1 - (1 - lr.score(data, target)) * (
        len(target) - 1) / (len(target) - data.shape[1] - 1)

        # Get Model 2
        data, target = self.read_dataset_one("preceedby.csv")
        data = preprocessing.scale(data)
        print data.shape
        poly = preprocessing.PolynomialFeatures(degree=2, interaction_only= True)
        data = poly.fit_transform(data)
        print "--------- Model 2 Y~ -------", poly.get_feature_names()
        print data.shape
        lr.fit(data, target)
        predicted = cross_val_predict(lr, data, target, cv=self.K_FOLD)
        # print predicted
        print "R- Square, Adjusted R-Square ", lr.score(data, target), 1 - (1 - lr.score(data, target)) * (
            len(target) - 1) / (len(target) - data.shape[1] - 1)

        # Get Model 3
        data, target = self.read_dataset_one("preceedby.csv")
        data = preprocessing.scale(data)
        print data.shape
        poly = preprocessing.PolynomialFeatures(degree=2)
        data = poly.fit_transform(data)
        print "--------- Model 3 Y~ -------", poly.get_feature_names()
        print data.shape
        lr.fit(data, target)
        predicted = cross_val_predict(lr, data, target, cv=self.K_FOLD)
        # print predicted
        print "R- Square, Adjusted R-Square ", lr.score(data, target), 1 - (1 - lr.score(data, target)) * (
            len(target) - 1) / (len(target) - data.shape[1] - 1)

        # Get Model 4
        data, target = self.read_dataset_one("preceedby.csv")
        data = preprocessing.scale(data)
        print data.shape
        poly = preprocessing.PolynomialFeatures(degree=3, interaction_only= True)
        data = poly.fit_transform(data)
        print "--------- Model 4 Y~ -------", poly.get_feature_names()
        print data.shape
        lr.fit(data, target)
        predicted = cross_val_predict(lr, data, target, cv=self.K_FOLD)
        # print predicted
        print "R- Square, Adjusted R-Square ", lr.score(data, target), 1 - (1 - lr.score(data, target)) * (
            len(target) - 1) / (len(target) - data.shape[1] - 1)

        # Get Model 5
        data, target = self.read_dataset_one("preceedby.csv")
        data = preprocessing.scale(data)
        print data.shape
        poly = preprocessing.PolynomialFeatures(degree=3)
        data = poly.fit_transform(data)
        print "--------- Model 5 Y~ -------", poly.get_feature_names()
        print data.shape
        lr.fit(data, target)
        predicted = cross_val_predict(lr, data, target, cv=self.K_FOLD)
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
        predicted = cross_val_predict(lr, data, target, cv=self.K_FOLD)
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
        predicted = cross_val_predict(lr, data, target, cv=self.K_FOLD)
        # print predicted
        print "R- Square, Adjusted R-Square ", lr.score(data, target), 1 - (1 - lr.score(data, target)) * (
        len(target) - 1) / (len(target) - data.shape[1] - 1)


        # Get Model 3
        print "--------- Model 3 Y~ Id+Action+PointofDay -------"
        data, target = self.get_model3("papermodel.csv")
        data = preprocessing.scale(data)
        lr.fit(data, target)
        predicted = cross_val_predict(lr, data, target, cv=self.K_FOLD)
        # print predicted
        print "R- Square, Adjusted R-Square ", lr.score(data, target), 1 - (1 - lr.score(data, target)) * (
            len(target) - 1) / (len(target) - data.shape[1] - 1)

        # Get Model 4
        print "--------- Model 4 Y~ Id+Action+wknd -------"
        data, target = self.get_model4("papermodel.csv")
        data = preprocessing.scale(data)
        lr.fit(data, target)
        predicted = cross_val_predict(lr, data, target, cv=self.K_FOLD)
        # print predicted
        print "R- Square, Adjusted R-Square ", lr.score(data, target), 1 - (1 - lr.score(data, target)) * (
            len(target) - 1) / (len(target) - data.shape[1] - 1)'''

        # Get Model 5
        print "--------- Model 5 Y~ IC+ IC^2 + Id*date + Id*PointofDay + Date + Action + Wknd + PointofDay -------"
        data, target = self.get_model5("papermodel.csv")
        data = preprocessing.scale(data)
        lr.fit(data, target)
        predicted = cross_val_predict(lr, data, target, cv=self.K_FOLD)
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
        lda = discriminant_analysis.LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
        X_scaled = preprocessing.scale(self.X)
        # cross_val_predict returns an array of the same size as `y` where each entry
        # is a prediction obtained by cross validation:
        poly = preprocessing.PolynomialFeatures(degree=3)
        X_scaled = poly.fit_transform(X_scaled)
        before = datetime.now()
        print before
        predicted = cross_val_predict(lda, X_scaled, self.Y, cv=self.K_FOLD)
        after = datetime.now()
        print after
        #data, target = self.read_dataset_one("preceedby.csv")
        #data = preprocessing.scale(data)
        #poly = preprocessing.PolynomialFeatures(degree=3)
        #data = poly.fit_transform(data)
        #predicted = cross_val_predict(lda, data, target, cv=self.K_FOLD)
        #print predicted
        print "LDA Classifier Report"
        print "===========================================\n"
        self.display_report(predicted, self.ALGORITHMS["lda"])
        runtime = (after - before).total_seconds()
        print "Time: "
        print runtime
        #print confusion_matrix(self.Y, predicted)

    def quardatic_discriminative_analysis(self):
        qda = discriminant_analysis.QuadraticDiscriminantAnalysis()
        X_scaled = preprocessing.scale(self.X)
        # cross_val_predict returns an array of the same size as `y` where each entry
        # is a prediction obtained by cross validation:
        poly = preprocessing.PolynomialFeatures(degree=3)
        X_scaled = poly.fit_transform(X_scaled)
        predicted = cross_val_predict(qda, X_scaled, self.Y, cv=self.K_FOLD)
        #data, target = self.read_dataset_one("preceedby.csv")
        #data = preprocessing.scale(data)
        #poly = preprocessing.PolynomialFeatures(degree=3)
        #data = poly.fit_transform(data)
        #predicted = cross_val_predict(qda, data, target, cv=self.K_FOLD)
        # print predicted
        print "QDA Classifier Report"
        print "===========================================\n"
        self.display_report(predicted, self.ALGORITHMS["qda"])
        # print confusion_matrix(self.Y, predicted)

    def svm_classify(self):
        """
        Uses Support Vector Machine Classifier
        :return: None
        """
        #for kernel in svm_kernels:
        X_scaled = preprocessing.scale(self.X)
        poly = preprocessing.PolynomialFeatures(degree=3)
        X_scaled = poly.fit_transform(X_scaled)

        svm_classifier = svm.SVC(kernel="linear")
        before = datetime.now()
        print before
        print X_scaled.shape
        svm_output = cross_val_predict(svm_classifier, X_scaled, self.Y, cv=self.K_FOLD)
        after = datetime.now()
        print after
        print "Support Vector Machine Classifier Report"
        print "===========================================\n"
        self.display_report(svm_output, self.ALGORITHMS["svm"])
        runtime = (after - before).total_seconds()
        print "Time: "
        print runtime

    def random_forest_classify(self):
        X_scaled = preprocessing.scale(self.X)
        poly = preprocessing.PolynomialFeatures(degree=3)
        X_scaled = poly.fit_transform(X_scaled)
        random_forest_classifier = RandomForestClassifier(n_estimators=10, criterion="entropy")
        classification = cross_val_predict(random_forest_classifier,X_scaled, self.Y, cv=self.K_FOLD)
        print "Random Forest Classifier Report"
        print "===========================================\n"
        self.display_report(classification, self.ALGORITHMS["random_forest"])

    def decision_tree_classify(self):
        X_scaled = preprocessing.scale(self.X)
        poly = preprocessing.PolynomialFeatures(degree=3)
        X_scaled = poly.fit_transform(X_scaled)
        decision_tree_classifier = DecisionTreeClassifier()
        classification = cross_val_predict(decision_tree_classifier, X_scaled, self.Y, cv=self.K_FOLD)
        print "Decision Tree Classifier Report"
        print "===========================================\n"
        print classification
        self.display_report(classification, self.ALGORITHMS["dt"])

    def ada_boost_classify(self):
        X_scaled = preprocessing.scale(self.X)
        poly = preprocessing.PolynomialFeatures(degree=3)
        X_scaled = poly.fit_transform(X_scaled)
        ada_boost_classifier = AdaBoostClassifier()
        classification = cross_val_predict(ada_boost_classifier, X_scaled, self.Y, cv=self.K_FOLD)
        print "Ada Boost Classifier Report"
        print "===========================================\n"
        self.display_report(classification, self.ALGORITHMS["ada"])

    def one_class_svm(self):
        # this will return a pandas dataframe.
        data, targs = self.read_dataset_one("preceedby.csv")
        # targs.loc[targs == 1] = -1
        targs.loc[targs == 0] = 1
        data = preprocessing.scale(data)
        # print targs
        train_data, test_data, train_target, test_target = train_test_split(data, targs, train_size=0.8, test_size=0.2)

        # train_data,  train_target = read_dataset("healthy_train.csv")
        # test_data, test_target = read_dataset("healthy_test.csv")
        # print(train_data.shape)
        # print(test_data.shape)
        # print train_data
        onesvm = svm.OneClassSVM(kernel='rbf', nu=0.165, gamma=0.1)
        onesvm.fit(train_data)
        preds = onesvm.predict(train_data)
        # print preds.shape
        # print train_target.shape
        targs = train_target
        n_error_train = preds[preds == -1].size
        print n_error_train

        print("accuracy: ", metrics.accuracy_score(targs, preds))
        print("precision: ", metrics.precision_score(targs, preds))
        print("recall: ", metrics.recall_score(targs, preds))
        print("f1: ", metrics.f1_score(targs, preds))
        # print("area under curve (auc): ", metrics.roc_auc_score(targs, preds))
        self.plot_confusion_matrix(confusion_matrix(targs, preds), "onesvm")
        print classification_report(targs, preds)

        print ("----------Test Result------")
        preds = onesvm.predict(test_data)
        targs = test_target
        n_error_test = preds[preds == -1].size
        print n_error_test

        print("accuracy: ", metrics.accuracy_score(targs, preds))
        print("precision: ", metrics.precision_score(targs, preds))
        print("recall: ", metrics.recall_score(targs, preds))
        print("f1: ", metrics.f1_score(targs, preds))
        print classification_report(targs, preds)
        # print("area under curve (auc): ", metrics.roc_auc_score(targs, preds))'''

        print "Confusion Matrix: "
        print "------------------\n"

    def display_report(self, prediction, algorithm):
        """
        Displays the classification report
        :param prediction: Prediction of the model
        :param algorithm: Algorithm used to generate predictions
        :return: None
        """
        print "Confusion Matrix: "
        print "------------------\n"
        self.plot_confusion_matrix(confusion_matrix(self.Y, prediction),algorithm)

        print "\nClassification Report: "
        print "-----------------------"
        print("accuracy: ", metrics.accuracy_score(self.Y, prediction))
        print("precision: ", metrics.precision_score(self.Y, prediction))
        print("recall: ", metrics.recall_score(self.Y, prediction))
        print("f1: ", metrics.f1_score(self.Y, prediction))
        print("area under curve (auc): ", metrics.roc_auc_score(self.Y, prediction))
        print classification_report(self.Y,prediction)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(self.Y, prediction)
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

