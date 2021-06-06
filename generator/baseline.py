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
    y_scores = clf.score_samples(data)
    anomalies = [window[i] for i in range(len(y_pred)) if y_pred[i] == -1]
    return len(anomalies), anomalies, y_scores

def offline_baseline_roc_aucs(orig_data, labels, outliers_fraction, distance_metric, max_attributes, restructure):
    data = _process(orig_data, max_attributes, restructure)
    iforest = IsolationForest()
    iforest.fit(data)
    svm = OneClassSVM()
    svm.fit(data)
    lof = LocalOutlierFactor(metric=distance_metric)
    lof.fit(data)
    return roc_curve(labels, iforest.score_samples(data)), roc_curve(labels, svm.score_samples(data)), roc_curve(labels, lof.negative_outlier_factor_)
