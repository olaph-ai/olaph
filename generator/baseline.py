from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
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

def num_iforest_anomalies(orig_data, max_attributes, restructure):
    data = _process(orig_data, max_attributes, restructure)
    clf = IsolationForest(random_state=0)
    clf.fit(data)
    y_pred = clf.predict(data)
    anomalies = [orig_data[i] for i in range(len(y_pred)) if y_pred[i] == -1]
    return len(anomalies), anomalies

def num_oc_svm_anomalies(orig_data, max_attributes, restructure):
    data = _process(orig_data, max_attributes, restructure)
    clf = OneClassSVM()
    clf.fit(data)
    y_pred = clf.predict(data)
    anomalies = [orig_data[i] for i in range(len(y_pred)) if y_pred[i] == -1]
    return len(anomalies), anomalies

def num_lof_anomalies(orig_data, max_attributes, restructure):
    data = _process(orig_data, max_attributes, restructure)
    clf = LocalOutlierFactor(n_neighbors=min(20,len(orig_data)-1))
    y_pred = clf.fit_predict(data)
    anomalies = [orig_data[i] for i in range(len(y_pred)) if y_pred[i] == -1]
    return len(anomalies), anomalies
