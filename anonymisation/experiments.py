from anonymisation.evaluation_utils import *
from anonymisation.utils import read_data_path, create_taxonomy
from anonymisation.distance_computations import add_semantic_distances, compute_boundary
from anonymisation.disclosure_computations import record_linkage
import pandas as pd
from anonymisation.utils import normalise
from anonymisation.scripts import generate_attribute_combinations
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
import math
import re

def test_1_1(dataset_name, model):
    iterations = 50
    epsilons = [2.0, 1.5, 1.25, 1.0986122886681098, 1.0, 0.75, 0.6931471805599453, 0.5, 0.1, 0.01]

    eps_str = [str(eps) for eps in epsilons]

    path_in = os.getcwd()
    pattern = '^.*/thesis-data-anonymisation/'
    path = re.search(pattern, path_in).group(0)

    original_dataset, attributes = read_data(dataset_name)
    taxonomies = [create_taxonomy(dataset_name, attr) for attr in attributes]
    add_semantic_distances_all(taxonomies)
    for taxonomy in taxonomies:
        taxonomy.add_boundary(compute_boundary(taxonomy))

    # SafePub
    if model == 'safepub':
        top_path = path+"data/result/safepub_test/1_1/"+dataset_name
        data_metrics = ['granularity', 'intensity', 'discernibility', 'entropy', 'groupsize']
        ks = [103, 72, 67, 61, 59, 54, 53, 50, 50, 50]*len(data_metrics)
        dataset_path = top_path + '/datasets'
        eval_metrics = ['discernibility', 'entropy', 'groupsize', 'sse']
        filename_combinations = generate_filename_combos('', '.csv', [data_metrics, ['eps'], eps_str], iterations)
        filename_combinations = [[filename.replace('eps_', 'eps-') for filename in file_list] for file_list in filename_combinations]
        compute_score_iterations2(dataset_path, top_path, filename_combinations, eval_metrics, ks, taxonomies, original_dataset)

        result_file_combinations = generate_filename_combos('result_', '.csv', [data_metrics, ['eps'], eps_str])
        result_file_combinations = [filename.replace('eps_', 'eps-') for filename in result_file_combinations]
        normalise_scores(dataset_name, top_path, result_file_combinations, ks)
#
        norm_result_combinations = generate_filename_combos('norm_result_', '.csv', [data_metrics, ['eps'], eps_str])
        norm_result_combinations = [filename.replace('eps_', 'eps-') for filename in norm_result_combinations]
        compute_mean_var(top_path, norm_result_combinations)
    elif model == 'sc':
        # Soria-Comas
        top_path = path+"data/result/sc_test/1_1/"+dataset_name
        if dataset_name == 'adult':
            ks = [174] * len(epsilons)
        elif dataset_name == 'housing':
            ks = [143] * len(epsilons)
        else:
            raise RuntimeError('Does not recognise the dataset')
        dataset_path = top_path + '/datasets'
        metrics = ['discernibility', 'entropy', 'groupsize', 'sse']
        filename_combinations = generate_filename_combos('eps-', '.csv', [eps_str], iterations)
        compute_score_iterations2(dataset_path, top_path, filename_combinations, metrics, ks, taxonomies,
                                  original_dataset)

        result_file_combinations = generate_filename_combos('result_eps-', '.csv', [eps_str])
        normalise_scores(dataset_name, top_path, result_file_combinations, ks)

        norm_result_combinations = generate_filename_combos('norm_result_eps-', '.csv', [eps_str])
        compute_mean_var(top_path, norm_result_combinations)
    elif model == 'sc_spec':
        top_path = path+"data/result/sc_spec_test/1_1/"+dataset_name
        ks = [800]*len(epsilons)
        result_file_combinations = generate_filename_combos('result_eps-', '.csv', [eps_str])
        normalise_scores(dataset_name, top_path, result_file_combinations, ks)

        norm_result_combinations = generate_filename_combos('norm_result_eps-', '.csv', [eps_str])
        compute_mean_var(top_path, norm_result_combinations)


    elif model == 'k-anonym':
        # k5-anon
        k_name = 'k5_suppression.csv'
        top_path = path+"data/result/k-anonym_test/1_1/"+dataset_name
        dataset_path = top_path+"/datasets"
        data = read_data_path(dataset_path+'/'+k_name)[0]
        metrics = ['discernibility', 'entropy', 'groupsize', 'sse']
        k = 5
        scores = np.array(compute_metric_scores(data, metrics, k, taxonomies, original_dataset)).reshape(1, -1)
        df = pd.DataFrame(scores, columns=metrics)
        df.to_csv(top_path+'/result_'+k_name, index=False)
#
        normalise_scores(dataset_name, top_path, ['result_'+k_name], [k])

    else:
        raise RuntimeError('Does not recognise model', model)


def test_1_1_a(dataset_name, count_records=True):
    """
    Compute the number of suppressed records for safepub
    Compute the number of suppressed values apart from the suppressed records as well?
    :return:
    """

    path_in = os.getcwd()
    pattern = '^.*/thesis-data-anonymisation/'
    path = re.search(pattern, path_in).group(0)

    epsilons = [2.0, 1.5, 1.25, 1.0986122886681098, 1.0, 0.75, 0.6931471805599453, 0.5, 0.1, 0.01]
    metrics = ['granularity', 'intensity', 'discernibility', 'entropy', 'groupsize']

    combinations = list(itertools.product(metrics, epsilons))
    filenames = ['/'+combo[0]+'_eps-'+str(combo[1]) for combo in combinations]

    root = path+'data/result/safepub_test/1_1/'+dataset_name
    datasets_path = root+'/datasets'
    for metric in metrics:
        all_suppressed =[]
        for eps in epsilons:
            num_suppressed = []
            for i in range(50):
                dataset = pd.read_csv(datasets_path+'/'+metric+'_'+'eps-'+str(eps)+'_'+str(i+1)+'.csv')
                groups = dataset.groupby(list(dataset.columns)).groups
                suppressed_record = tuple(['*']*len(dataset.columns))
                if count_records:
                    num_suppressed.append(len(groups[suppressed_record]))
                else:
                    for record in list(groups.keys()):
                        if record != suppressed_record:
                            suppressed_val = 0
                            for val in record:
                                if val == '*':
                                    suppressed_val += 1
                            num_suppressed.append(suppressed_val)
                            break
                    else:
                        num_suppressed.append(len(suppressed_record))
            all_suppressed.append(num_suppressed)
        suppressed = np.array(all_suppressed).T
        df = pd.DataFrame(suppressed, columns=epsilons)
        if count_records:
            filename = 'num_suppressed_records'
        else:
            filename = 'num_suppressed_attributes'
        df.to_csv(root+'/'+filename+'_'+metric+'.csv', index=False)
        print('metric', metric, 'done')
    return


def test_1_1_b(dataset_name):

    # Safepub
    #compute_hist_vals('adult', 'safepub', [['granularity_', 'intensity_', 'discernibility_', 'entropy_', 'groupsize_'], ['eps-'],
    #                       ['2.0', '1.5', '1.25', '1.0986122886681098', '1.0', '0.75', '0.6931471805599453',
    #                       '0.5', '0.1', '0.01']], 50)

    # Soria-Comas
    compute_hist_vals(dataset_name, 'sc',
                      [['eps-'], ['2.0', '1.5', '1.25', '1.0986122886681098', '1.0', '0.75', '0.6931471805599453',
                                  '0.5', '0.1', '0.01']],
                      50)
    # k-anonymity
    #compute_hist_vals('adult', 'k-anonym', [['k5'], ['', '_mdav', '_suppression']], 1)


def test_1_2(dataset_name, model):
    # Read the datasets, compute the score for each dataset
    iterations = 50
    path_in = os.getcwd()
    pattern = '^.*/thesis-data-anonymisation/'
    path = re.search(pattern, path_in).group(0)

    original_dataset, attributes = read_data_path(
        path+'data/' + dataset_name + '/' + dataset_name + '.csv')
    taxonomies = [create_taxonomy(dataset_name, attr) for attr in attributes]
    add_semantic_distances_all(taxonomies)
    for taxonomy in taxonomies:
        taxonomy.add_boundary(compute_boundary(taxonomy))

    if model == 'safepub':
        # SafePub
        top_path = path+"data/result/safepub_test/1_2/" + dataset_name
        dataset_path = top_path+'/datasets'
        metrics = ['discernibility', 'entropy', 'groupsize', 'sse']
        ks = [59, 74, 88, 100, 114, 129, 141, 155, 170, 184, 199, 211, 225, 240, 252, 266]
        deltas = [1E-5, 1E-6, 1E-7, 1E-8, 1E-9, 1E-10, 1E-11, 1E-12, 1E-13, 1E-14, 1E-15, 1E-16, 1E-17, 1E-18, 1E-19, 1E-20]
        deltas_string = ['1.0E-5', '1.0E-6', '1.0E-7', '1.0E-8', '1.0E-9', '1.0E-10', '1.0E-11', '1.0E-12', '1.0E-13',
                         '1.0E-14', '1.0E-15', '1.0E-16', '1.0E-17', '1.0E-18', '1.0E-19', '1.0E-20']
        filename_combinations = generate_filename_combos('delta_', '.csv', [deltas_string], iterations)
        compute_score_iterations2(dataset_path, top_path, filename_combinations, metrics, ks, taxonomies, original_dataset)
        result_file_combinations = generate_filename_combos('result_delta_', '.csv', [deltas_string], 1)
        normalise_scores(dataset_name, top_path, result_file_combinations, ks)
        norm_combinations = generate_filename_combos('norm_result_delta_', '.csv', [deltas_string])
        compute_mean_var(top_path, norm_combinations)

    elif model == 'sc':
        # Soria-Comas
        top_path = path+"data/result/sc_test/1_2/" + dataset_name
        dataset_path = top_path+'/datasets'
        metrics = ['discernibility', 'entropy', 'groupsize', 'sse']
        if dataset_name == 'adult':
            ks = list(range(200, 4701, 300))
        elif dataset_name == 'housing':
            ks = list(range(200, 4701, 300))
        else:
            raise RuntimeError('Does not recognse dataset name', dataset_name)
#
        ks_string = [str(k) for k in ks]
        filename_combinations = generate_filename_combos('k_', '.csv', [ks_string], 50)
        compute_score_iterations2(dataset_path, top_path, filename_combinations, metrics, ks, taxonomies, original_dataset)
        result_file_combinations = generate_filename_combos('result_k_', '.csv', [ks_string], 1)
        normalise_scores(dataset_name, top_path, result_file_combinations, ks)
        norm_combinations = generate_filename_combos('norm_result_k_', '.csv', [ks_string])
        compute_mean_var(top_path, norm_combinations)


def test_2_1(dataset_name, model, attributes):
    # Compute the scores and the normalised scores for each attribute combination

    iterations = 25

    path_in = os.getcwd()
    pattern = '^.*/thesis-data-anonymisation/'
    path = re.search(pattern, path_in).group(0)

    original_dataset_path = path+'data/'+dataset_name+'/attribute_subsets'

    metrics = ['discernibility', 'entropy', 'groupsize', 'sse']

    taxonomies = [create_taxonomy(dataset_name, attr) for attr in attributes]
    add_semantic_distances_all(taxonomies)
    for taxonomy in taxonomies:
        taxonomy.add_boundary(compute_boundary(taxonomy))

    attr_to_idx = {attr: n for n, attr in enumerate(attributes)}

    if dataset_name == 'adult':
        attribute_range = list(range(2, 9))
    elif dataset_name == 'housing':
        attribute_range = list(range(2, 10))
    elif dataset_name == 'musk':
        attribute_range = list(range(2, 21))
    else:
        raise RuntimeError("Does not recognise dataset", dataset_name)

    if model == 'safepub':
        # Safepub
        top_path = path+'data/result/safepub_test/2_1/'+dataset_name
        dataset_paths = top_path+'/datasets'
        if dataset_name == 'musk':
            ks = [45]
        else:
            ks = [59]
        for a in attribute_range:
            output_path = top_path+'/'+str(a)
            original_filenames = os.listdir(original_dataset_path+'/'+str(a))
            for n, filename in enumerate(original_filenames):
                original_dataset, current_attrs = read_data_path(original_dataset_path+'/'+str(a)+'/'+filename)
                filenames = ['-'.join(current_attrs)+'_'+str(i+1)+'.csv' for i in range(iterations)]
#
                current_taxomomies = [taxonomies[attr_to_idx[attr]] for attr in current_attrs]
#
                compute_score_iterations2(dataset_paths+'/'+str(a), output_path, [filenames], metrics, ks,
                                          current_taxomomies, original_dataset)
                print("File", n, "of", len(original_filenames))
            result_filenames = [filen for filen in os.listdir(output_path) if re.match("^result_", filen)]
            normalise_scores_subattributes(dataset_name, output_path, result_filenames, ks*len(result_filenames), a)

            norm_result_filenames = [filen for filen in os.listdir(output_path) if re.match("^norm_result_", filen)]
            compute_mean_var(output_path, norm_result_filenames)
            print("Attribute", a, "done")
    elif model == 'sc':
        # S-C
        top_path = path+'data/result/sc_test/2_1/'+dataset_name
        dataset_paths = top_path+'/datasets'
        if dataset_name == 'adult':
            ks = [174]
        elif dataset_name == 'housing':
            ks = [143]
        elif dataset_name == 'musk':
            ks = [82]
        else:
            raise RuntimeError('Does not recognise dataset', dataset_name)

        for a in attribute_range:
            output_path = top_path + '/' + str(a)
            original_filenames = os.listdir(original_dataset_path+'/'+str(a))
            for n, filename in enumerate(original_filenames):
                original_dataset, current_attrs = read_data_path(original_dataset_path+'/'+str(a)+'/'+filename)
                filenames = ['-'.join(current_attrs)+'_'+str(i+1)+'.csv' for i in range(iterations)]
                current_taxomomies = [taxonomies[attr_to_idx[attr]] for attr in current_attrs]
                compute_score_iterations2(dataset_paths+'/'+str(a), output_path, [filenames], metrics, ks,
                                          current_taxomomies, original_dataset)
                print("File", n, "of", len(original_filenames))
            result_filenames = [filen for filen in os.listdir(output_path) if re.match("^result_", filen)]
            normalise_scores_subattributes(dataset_name, output_path, result_filenames, ks*len(result_filenames), a)
            norm_result_filenames = [filen for filen in os.listdir(output_path) if re.match("^norm_result_", filen)]
            compute_mean_var(output_path, norm_result_filenames)
            print("Attribute", a, "done")
    return


def test_3_1(dataset_name, model):
    iterations = 25

    path_in = os.getcwd()
    pattern = '^.*/thesis-data-anonymisation/'
    path = re.search(pattern, path_in).group(0)

    epsilons = ['1.0', '2.0']
    original_dataset_path = \
        path+'data/' + dataset_name + '/'+dataset_name+'.csv'
    dataset_top_path = path+'data/result/'+model+'_test/3_1/'\
                    +dataset_name
    datasets_path = dataset_top_path+'/datasets'

    metrics = ['sse', 'record_linkage']

    original_dataset, attributes = read_data_path(original_dataset_path)

    taxonomies = [create_taxonomy(dataset_name, attr) for attr in attributes]
    add_semantic_distances_all(taxonomies)
    for taxonomy in taxonomies:
        taxonomy.add_boundary(compute_boundary(taxonomy))

    if model == 'safepub':
        ks = [45, 74]
        filename_combinations = generate_filename_combos('granularity_eps-', '.csv', [epsilons], iterations)
    elif model == 'sc':
        ks = [32]*len(epsilons)
        filename_combinations = generate_filename_combos('eps-', '.csv', [epsilons], iterations)
    elif model == 'k-anonym':
        ks = [5]
        filename_combinations = [generate_filename_combos('k5_suppression', '.csv', [[""]])]
    else:
        raise RuntimeError('Does not recognise model', model)


    compute_score_iterations2(datasets_path, dataset_top_path, filename_combinations, metrics, ks, taxonomies,
                                  original_dataset)
    result_filenames = [filen for filen in os.listdir(dataset_top_path) if re.match("^result_", filen)]
    normalise_scores(dataset_name, dataset_top_path, result_filenames, ks)
    norm_result_filenames = [filen for filen in os.listdir(dataset_top_path) if re.match("^norm_result_", filen)]
    compute_mean_var(dataset_top_path, norm_result_filenames)
    return


def normalise_scores(dataset_name, path, file_combinations, ks):
    original_scores = pd.read_csv(
        path+'data/result/reference/' + dataset_name + '/original.csv')
    suppressed_scores = pd.read_csv(
        path+'data/result/reference/' + dataset_name + '/suppressed.csv')

    for i, filename in enumerate(file_combinations):
        k = ks[i]
        score_o = original_scores.loc[original_scores['k'] == k]
        score_s = suppressed_scores.loc[suppressed_scores['k'] == k]
        normalised_scores = []
        score_df = pd.read_csv(path+'/'+filename)
        metrics = score_df.columns
        for row in score_df.values:
            normalised_score = []
            for n, m in enumerate(metrics):
                if m != 'record_linkage':
                    norm_score = normalise(row[n], score_o[m].values[0], score_s[m].values[0])
                    normalised_score.append(norm_score)
            normalised_scores.append(normalised_score)
        columns = list(score_df.columns)
        if 'record_linkage' in columns:
            columns.remove('record_linkage')
        norm_score_np = np.array(normalised_scores)
        norm_df = pd.DataFrame(norm_score_np, columns=columns)
        norm_df.to_csv(path + '/norm_' + filename, index=False)
    return


def normalise_scores_subattributes(dataset_name, path, file_combinations, ks, a):
    if dataset_name == 'adult':
        attribute_range = list(range(2, 9))
    elif dataset_name == 'housing':
        attribute_range = list(range(2, 10))
    elif dataset_name == 'musk':
        attribute_range = list(range(2, 20))
    else:
        raise RuntimeError("Does not recognise dataset", dataset_name)

    reference_path = path+'data/result/reference/' + dataset_name+'/'+str(a)


    for i, file in enumerate(file_combinations):

        k = ks[i]
        normalised_scores = []
        base_file = re.split("^result_", file)[1]
        original_scores = pd.read_csv(reference_path+'/original_'+base_file)
        suppressed_scores = pd.read_csv(reference_path+'/suppressed_'+base_file)
        score_o = original_scores.loc[original_scores['k'] == k]
        score_s = suppressed_scores.loc[suppressed_scores['k'] == k]
        score_df = pd.read_csv(path+'/'+file)
        metrics = score_df.columns
        for row in score_df.values:
            normalised_score = []
            for n, m in enumerate(metrics):
                norm_score = normalise(row[n], score_o[m].values[0], score_s[m].values[0])
                normalised_score.append(norm_score)
            normalised_scores.append(normalised_score)
        norm_score_np = np.array(normalised_scores)
        norm_df = pd.DataFrame(norm_score_np, columns=score_df.columns)
        norm_df.to_csv(path + '/norm_' + file, index=False)


if __name__ == '__main__':
    adult_attributes = ['age','workclass','education','marital_status','occupation','race','sex','native_country']
    housing_attributes = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
                  "population", "households", "median_income", "median_house_value"]

    #normalise_scores_k('1_1')
    #test_1_1_analysis()
    #plot_stuff()
    #plot_histograms3('original', 'adult', 'adult', 'native_country', '1_1_b_original')
    #plot_histograms3('sc', 'adult', 'hist_eps-2.0.pkl', 'native_country', '1_1_b_eps-2.0')
    #plot_histograms3('sc', 'adult', 'hist_eps-1.0.pkl', 'native_country', '1_1_b_eps-1.0')
    #plot_histograms3('sc', 'adult', 'hist_eps-0.1.pkl', 'native_country', '1_1_b_eps-0.1')
    #plot_histograms3('safepub', 'hist_granularity_eps-2.0.pkl', 'age')
    #plot_different_delctas()
    #plot_different_ks()
    #merge_results('adult', '1_2', [])

    #y_limit = {'discernibility': [-0.1, 1.05], 'entropy': [-0.5, 1.05], 'groupsize': [0, 1.05], 'sse': [0.3, 0.8]}


    #compute_total_mean('adult', 'sc')
    #join_total_mean('adult', 'sc')
    #plot_2_1('adult')

    #test_3_1('housing_small', 'safepub')

    test_2_1('musk', 'safepub', [str(i+1) for i in range(20)])

    pass
