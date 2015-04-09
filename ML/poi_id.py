#!/usr/bin/python

import sys
import pickle
import math
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
import pca_mask
from tester import test_classifier, dump_classifier_and_data
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
### Select features
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary']
features_list = ['poi','salary', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
finance_list = ['poi','salary', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_list = ['poi','to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )


        
### Task 2: Remove outliers
data_dict.pop("TOTAL",0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK",0)
data_dict.pop("LOCKHART EUGENE E",0)



### Create new feature(s)
def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    fraction = 0.
    poi_messages = float(poi_messages)
    all_messages = float(all_messages)
    if not (math.isnan(poi_messages) or math.isnan(all_messages)):
        fraction = poi_messages/all_messages
    return fraction
    

sub_features = {}

for name in data_dict:
 
    data_point = data_dict[name]
    sub_features[name] = {k: data_point[k] for k in finance_list}
        
    
    from_poi_to_this_person = data_dict[name]["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    shared = float(data_point['shared_receipt_with_poi'])
    if math.isnan(shared):
        shared = 0.
    sub_features[name]['shared_receipt_with_poi'] = data_point['shared_receipt_with_poi']
    sub_features[name]['fraction_to_poi'] = fraction_to_poi
    sub_features[name]['fraction_from_poi'] = fraction_from_poi


    color_x = 'b'
    if data_point['poi'] == 1:
        color_x = 'r'
    plt.scatter(fraction_to_poi,fraction_from_poi,color = color_x)

plt.show()

my_dataset = sub_features
my_features_list = ['poi','salary', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
                                   'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'long_term_incentive',
                                   'restricted_stock', 'director_fees','shared_receipt_with_poi','fraction_to_poi','fraction_from_poi']


data = featureFormat(my_dataset, my_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
folds=1000

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.decomposition import PCA
cv = StratifiedShuffleSplit(labels, folds, random_state = 42)

#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()
from sklearn.metrics import precision_score,recall_score


from sklearn.grid_search import GridSearchCV
param_grid = {
         'min_samples_split': [2,3,4,5,6]
          }

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
#clf = GridSearchCV(DecisionTreeClassifier(), param_grid)
clf = DecisionTreeClassifier(criterion="entropy", min_samples_split=2, max_features="auto", random_state=42)
true_negatives = 0
false_negatives = 0
true_positives = 0
false_positives = 0
mask = np.arange(15) < 12
for train_idx, test_idx in cv: 
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )

    to_pca = []
    to_pca_test = []
    not_pca = []
    not_pca_test = []
    for i in features_train:
        to_pca.append(i[mask])
        not_pca.append(i[~mask])
    for i in features_test:
        to_pca_test.append(i[mask])
        not_pca_test.append(i[~mask])

    pca = PCA(n_components = 2).fit(to_pca)
    features_train_pca = pca.transform(to_pca)
    features_test_pca = pca.transform(to_pca_test)

    features_final_train = np.hstack([features_train_pca,not_pca])
    features_final_test = np.hstack([features_test_pca,not_pca_test])

    clf.fit(features_final_train, labels_train)
    predictions = clf.predict(features_final_test)
   # print clf.score(features_final_test,labels_test)
   # print 'precision score:', precision_score(labels_test,predictions)
   # print 'recall score:', recall_score(labels_test,predictions)
    for prediction, truth in zip(predictions, labels_test):
        if prediction == 0 and truth == 0:
            true_negatives += 1
        elif prediction == 0 and truth == 1:
            false_negatives += 1
        elif prediction == 1 and truth == 0:
            false_positives += 1
        else:
            true_positives += 1

total_predictions = true_negatives + false_negatives + false_positives + true_positives
accuracy = 1.0*(true_positives + true_negatives)/total_predictions
precision = 1.0*true_positives/(true_positives+false_positives)
recall = 1.0*true_positives/(true_positives+false_negatives)
f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
print clf
PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)   
