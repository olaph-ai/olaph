import pandas as pd
import numpy as np
from preprocess import process_examples, get_requests_from_logs
from scipy.spatial import distance
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def _process(data, max_attributes, restructure):
    processed = process_examples(data, max_attributes, restructure)
    return list(map(lambda p: p[0] | p[1], processed))

def _preprocess(requests, prev_requests, max_attributes, restructure):
    requests = _process(requests, max_attributes, restructure)
    prev_requests = _process(prev_requests, max_attributes, restructure)
    processed = pd.DataFrame(requests + prev_requests).astype(str)
    one_hot = pd.get_dummies(processed)
    return one_hot[:len(requests)], one_hot[len(requests):]

def compute_distances(requests, prev_requests, distance_measure, max_attributes, restructure):
    requests, prev_requests = _preprocess(requests, prev_requests, max_attributes, restructure)
    return distance.cdist(requests, prev_requests, distance_measure).min(axis=1)

def compute_hd_distance(requests, prev_requests, max_attributes):
    requests, prev_requests = _preprocess(requests, prev_requests, max_attributes)
    return distance.directed_hausdorff(requests, prev_requests)[0]

if __name__ == '__main__':
    logging.basicConfig()
    all_requests = get_requests_from_logs(f'../data/synheart-controller-opa-istio.log')
    requests, prev_requests = all_requests[40:50], all_requests[:40]
    ds = compute_distances(requests, prev_requests, 20)
