import numpy as np
import pandas as pd
import os
import re
from anonymisation.utils import create_taxonomy, read_data_path
from anonymisation.distance_computations import add_semantic_distances, compute_boundary
from anonymisation.S_C.functions import microaggregation, sanitise


def run_experiment4(dataset_name, k, float_vals, num_decimals):
    path_in = os.getcwd()
    pattern = '^.*/thesis-data-anonymisation/'
    path = re.search(pattern, path_in).group(0)

    dataset_path = path + 'data/' + dataset_name + '/' + dataset_name + '.csv'

    data, attributes = read_data_path(dataset_path)

    taxonomies = [create_taxonomy(dataset_name, attr) for attr in attributes]
    add_semantic_distances(taxonomies)
    for taxonomy in taxonomies:
        taxonomy.add_boundary(compute_boundary(taxonomy))

    iterations = 25
    epsilons = [1.0, 2.0]

    output_path = path+'data/result/sc_test/3_1/' + dataset_name + '/datasets'

    cluster_root_path = path+'anonymisation/S_C/clusters/' + dataset_name + '/k_'+ str(k) + '.csv'

    for epsilon in epsilons:
        print('####################### epsilon ' + str(epsilon))
        for i in range(iterations):
            output_file = output_path + '/eps-' + str(epsilon) + '_' + str(i + 1) + '.csv'
            X_bar = microaggregation(data, k, taxonomies, epsilon, add_cluster_noise=True,
                                     cluster_path=cluster_root_path)
            anon_data = sanitise(X_bar.values, epsilon, k, taxonomies, float_vals, num_decimals)
            anonymised = pd.DataFrame(anon_data, columns=data.columns)
            anonymised.to_csv(output_file, index=False)


if __name__ == '__main__':

    k = 32

    housing_float_attributes = [True] * 9
    housing_num_decimals = [2, 2, 0, 0, 0, 0, 0, 2, 0]

    run_experiment4('housing_small', k, housing_float_attributes, housing_num_decimals)

