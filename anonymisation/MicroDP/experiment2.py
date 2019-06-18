import pandas as pd
from anonymisation.utils import read_data_path, create_taxonomy
from anonymisation.distance_computations import add_semantic_distances, compute_boundary
from anonymisation.S_C.functions import microaggregation, sanitise
import os
import re

"""
Test 1.2
"""

def run_experiment_2(dataset_name, float_vals, num_decimals):
    path_in = os.getcwd()
    pattern = '^.*/thesis-data-anonymisation/'
    path = re.search(pattern, path_in).group(0)

    dataset_path = path + 'data/' + dataset_name + '/' + dataset_name + '.csv'

    data, attributes = read_data_path(dataset_path)
    taxonomies = [create_taxonomy(dataset_name, attr) for attr in attributes]
    add_semantic_distances(taxonomies)
    for taxonomy in taxonomies:
        taxonomy.add_boundary(compute_boundary(taxonomy))

    if dataset_name == 'adult':
        ks = list(range(200, 6000, 300))[:16]
    elif dataset_name == 'housing':
        ks = list(range(200, 6000, 300))[:16]
    else:
        raise RuntimeError('Does not recognise dataset', dataset_name)

    iterations = 50
    epsilon = 1.0

    output_path = path+'data/result/sc_test/1_2/' + dataset_name + '/datasets'


    for k in ks:
        cluster_root_path = path+'anonymisation/S_C/clusters/' + dataset_name + '/k_' \
                            + str(k) + '.csv'
        print('####################### k ' + str(k))
        for i in range(iterations):
            output_file = output_path + '/k_'+str(k)+'_'+str(i+1)+'.csv'

            try:
                with open(output_file, 'r') as test_file:
                    print("File found")
            except FileNotFoundError:
                X_bar = microaggregation(data, k, taxonomies, epsilon, add_cluster_noise=True, cluster_path=cluster_root_path)
                anon_data = sanitise(X_bar.values, epsilon, k, taxonomies, float_vals, num_decimals)
                anonymised = pd.DataFrame(anon_data, columns=data.columns)
                anonymised.to_csv(output_file, index=False)


if __name__ == '__main__':
    run_experiment_2('adult', [False]*8, [0]*8)
