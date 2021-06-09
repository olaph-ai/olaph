from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_curve
from copy import deepcopy
from preprocess import process_examples
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s: %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def _process(orig_data, max_attributes, restructure):
    processed = process_examples(orig_data, max_attributes, restructure)
    processed = list(map(lambda p: p[0] | p[1], processed))
    data = pd.DataFrame(processed).astype(str)
    data = pd.get_dummies(data)
    return data

def train_baselines(orig_data, outliers_fraction, distance_metric, max_attributes, restructure):
    data = _process(orig_data, max_attributes, restructure)
    iforest = IsolationForest(random_state=0, contamination=outliers_fraction)
    iforest.fit(data)
    svm = OneClassSVM(nu=outliers_fraction)
    svm.fit(data)
    lof = LocalOutlierFactor(n_neighbors=min(20,len(orig_data)-1), novelty=True,
                             contamination=outliers_fraction, metric=distance_metric)
    lof.fit(data)
    return iforest, svm, lof, data.columns

def num_baseline_anomalies(clf, window, trained_attrs, max_attributes, restructure):
    data = _process(deepcopy(window), max_attributes, restructure)
    data = data.reindex(columns=trained_attrs, fill_value=0)
    y_pred = clf.predict(data)
    y_scores = clf.decision_function(data)
    anomalies = [window[i] for i in range(len(y_pred)) if y_pred[i] == -1]
    return len(anomalies), anomalies, y_scores

def offline_baseline_roc_aucs(orig_data, labels, outliers_fraction, distance_metric, max_attributes, restructure):
    learn_set, test_set = orig_data[:1000], orig_data[1000:]
    learn_set = _process(learn_set, max_attributes, restructure)
    test_set = _process(test_set, max_attributes, restructure)
    test_set = test_set.reindex(columns=learn_set.columns, fill_value=0)
    iforest = IsolationForest(contamination=outliers_fraction)
    iforest.fit(learn_set)
    log.info('IF fitted')
    svm = OneClassSVM(nu=outliers_fraction)
    svm.fit(learn_set)
    log.info('SVM fitted')
    lof = LocalOutlierFactor(metric=distance_metric, novelty=True, contamination=outliers_fraction)
    lof.fit(learn_set)
    log.info('LOF fitted')
    return roc_curve(labels, iforest.decision_function(test_set), pos_label=-1), roc_curve(labels, svm.decision_function(test_set), pos_label=-1), roc_curve(labels, lof.decision_function(test_set), pos_label=-1)
