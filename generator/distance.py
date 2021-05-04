import pandas as pd
import numpy as np
from preprocess import get_requests_from_logs
from scipy.spatial import distance
import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

def compute_distances(prev_requests, requests):
    len_prev_requests = len(prev_requests)
    data = pd.json_normalize(prev_requests + requests)
    data = data.astype(str)
    data = pd.get_dummies(data)
    prev_requests, requests = data[:len_prev_requests], data[len_prev_requests:]
    d = distance.cdist(prev_requests, requests, 'euclidean')
    d = d.min(axis=1)
    # log.debug(np.unique(d))
    return d

if __name__ == '__main__':
    all_requests = get_requests_from_logs(f'../data/synheart-controller-opa-istio.log')
    compute_distances(all_requests[:300], all_requests[300:600])
