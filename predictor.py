# ******************************************************************************
# predictor.py
#
# Date      Name       Description
# ========  =========  ========================================================
# 3/2/19   Paudel     Initial version,
# ******************************************************************************

from leave_one_subject_out import LeaveOneOutClassifiers
from regular_classifiers import RegularClassifiers

class CognitiveHealthPredictor():
    def __init__(self):
        print
        "\n\nStarting Cognitive Health Prediction ------ "

    def cognitive_health_prediction(self):
        # Use Regurlar 10-Fold Cross validation
        regular_classifier = RegularClassifiers()
        regular_classifier.logistic_regression()
        regular_classifier.linear_discriminative_analysis()
        regular_classifier.quardatic_discriminative_analysis()
        regular_classifier.knn_classify()
        regular_classifier.random_forest_classify()
        regular_classifier.decision_tree_classify()
        regular_classifier.ada_boost_classify()
        regular_classifier.svm_classify()
        regular_classifier.one_class_svm()

        #Use leave one subject out validation
        leave_one_out = LeaveOneOutClassifiers()
        leave_one_out.logistic_regression()
        leave_one_out.linear_discriminative_analysis()
        leave_one_out.quardatic_discriminative_analysis()
        leave_one_out.knn_classify()
        leave_one_out.random_forest_classify()
        leave_one_out.decision_tree_classify()
        leave_one_out.ada_boost_classify()
        leave_one_out.svm_classify()
        leave_one_out.one_class_svm()
