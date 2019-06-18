from anonymisation.S_C.functions import *
import re


def run_experiment_1(dataset, k, float_vals, num_decimals):
    path_in = os.getcwd()
    pattern = '^.*/thesis-data-anonymisation/'
    path = re.search(pattern, path_in).group(0)

    dataset_path = path+'data/'+dataset+'/'+dataset+'.csv'

    data, attributes = read_data_path(dataset_path)
    taxonomies = [create_taxonomy(dataset, attr) for attr in attributes]
    add_semantic_distances(taxonomies)
    for taxonomy in taxonomies:
        taxonomy.add_boundary(compute_boundary(taxonomy))

    iterations = 50
    epsilons = [2.0, 1.5, 1.25, np.log(3), 1.0, 0.75, np.log(2), 0.5, 0.1, 0.01]

    output_path = path+'data/result/sc_test/1_1/'+dataset+'/datasets'

    cluster_root_path = path+'anonymisation/S_C/clusters/'+dataset+'/k_'+str(k)+'.csv'

    for epsilon in epsilons:
        print('####################### epsilon '+str(epsilon))
        for i in range(iterations):
            output_file = output_path+'/eps-'+str(epsilon)+'_'+str(i+1)+'.csv'
            X_bar = microaggregation(data, k, taxonomies, epsilon, add_cluster_noise=True, cluster_path=cluster_root_path)
            anon_data = sanitise(X_bar.values, epsilon, k, taxonomies, float_vals, num_decimals)
            anonymised = pd.DataFrame(anon_data, columns=data.columns)
            anonymised.to_csv(output_file, index=False)


if __name__ == '__main__':
    # Setup things

    # musk
    #k = 82

    # housing
    #k = 143

    k = 800

    adult_float_vals = [False] * 8
    adult_num_decimals = [0] * 8

    housing_loat_attributes = [True] * 9
    housing_num_decimals = [2, 2, 0, 0, 0, 0, 0, 2, 0]

    run_experiment_1('adult', k, adult_float_vals, adult_num_decimals, 'sc')
