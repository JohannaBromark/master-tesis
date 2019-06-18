import os
import re
import pandas as pd
from anonymisation.utils import create_taxonomy, read_data_path
from anonymisation.distance_computations import add_semantic_distances, compute_boundary
from anonymisation.S_C.functions import microaggregation, sanitise

"""
Test 2.1
"""
def run_experiment_3(dataset_name, attributes, k, num_attributes, float_vals, num_decimals):
    epsilon = 1.0
    path_in = os.getcwd()
    pattern = '^.*/thesis-data-anonymisation/'
    path = re.search(pattern, path_in).group(0)

    datasets_path = path + 'data/' + dataset_name + '/' + dataset_name + '.csv'

    iterations = 25

    attr_to_idx = {attr: n for n, attr in enumerate(attributes)}

    all_taxonomies = [create_taxonomy(dataset_name, attribute) for attribute in attributes]
    add_semantic_distances(all_taxonomies)
    for taxonomy in all_taxonomies:
        taxonomy.add_boundary(compute_boundary(taxonomy))

    for a in range(num_attributes, num_attributes+1):
        current_datasets_path = datasets_path+'/'+str(a)
        files = os.listdir(current_datasets_path)

        output_path = path+'data/result/sc_test/2_1/'+dataset_name+'/datasets/'+str(a)
        cluster_root_path = path+'anonymisation/S_C/clusters/'+dataset_name +'/clusters_attribute_subsets/'+str(a)
        for f, filename in enumerate(files):
            data, current_attrs = read_data_path(current_datasets_path+'/'+filename)
            current_taxonomies = [all_taxonomies[attr_to_idx[attr]] for attr in current_attrs]
            cluster_path = cluster_root_path+'/'+'-'.join(current_attrs)+'_'+str(k)+'.csv'
            current_float_vals = [float_vals[attr_to_idx[attr]] for attr in current_attrs]
            current_num_decimals = [num_decimals[attr_to_idx[attr]] for attr in current_attrs]
            for i in range(iterations):
                output_file = output_path+'/'+'-'.join(current_attrs)+'_'+str(i+1)+'.csv'
                try:
                    with open(output_file, 'r') as test_file:
                        print("File found")
                except FileNotFoundError:
                    X_bar = microaggregation(data, k, current_taxonomies, epsilon, add_cluster_noise=True,
                                 cluster_path=cluster_path)
                    anon_data = sanitise(X_bar.values, epsilon, k, current_taxonomies, current_float_vals, current_num_decimals)
                    anonymised = pd.DataFrame(anon_data, columns=data.columns)
                    anonymised.to_csv(output_file, index=False)
            print("---------------- file", f, "of", len(files), "done")
        print("#########################", a, 'attributes done')
    return

if __name__ == '__main__':

    #attributes = ["age_years", "body_pain", "body_temperature", "body_weight_kg",
    # "cold_and_flu_symptoms", "cold_and_flu_symptoms_duration", "medicine_allergy", "sex", "take_medicines",
    # "skin_rash_symptom", "works_with_children_elderly_or_sick_people"]

    #k = 317

    #attributes = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
    #              "population", "households", "median_income", "median_house_value"]
    #k = 143
    # [True]*9, [2, 2, 0, 0, 0, 0, 0, 2, 0]


    dataset_name = 'musk'
    attributes = [str(i+1) for i in range(20)]
    float_vals = [True]*20
    num_decimals = [0]*20
    k = 82

    run_experiment_3(dataset_name, attributes, k, 2, float_vals, num_decimals)
    run_experiment_3(dataset_name, attributes, k, 3, float_vals, num_decimals)
    run_experiment_3(dataset_name, attributes, k, 4, float_vals, num_decimals)
    run_experiment_3(dataset_name, attributes, k, 9, float_vals, num_decimals)
    run_experiment_3(dataset_name, attributes, k, 10, float_vals, num_decimals)
    run_experiment_3(dataset_name, attributes, k, 11, float_vals, num_decimals)


