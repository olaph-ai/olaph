import pandas as pd
import numpy as np
from preprocess import process_examples, get_requests_from_logs
from scipy.spatial import distance
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def compute_distances(requests, prev_requests, max_attributes):
    len_requests = len(requests)
    processed = process_examples(requests + prev_requests, max_attributes)
    processed = pd.DataFrame(list(map(lambda p: p[0] | p[1], processed)))
    data = processed.astype(str)
    data = pd.get_dummies(data)
    requests, prev_requests = data[:len_requests], data[len_requests:]
    d = distance.cdist(requests, prev_requests, 'euclidean')
    d = d.min(axis=1)
    return d

if __name__ == '__main__':
    logging.basicConfig()
    all_requests = get_requests_from_logs(f'../data/synheart-controller-opa-istio.log')
    ds = compute_distances(all_requests[3000:], all_requests[2000:3000], 20)
    # print(np.unique(ds))
