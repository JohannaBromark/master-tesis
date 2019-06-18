import pickle
import os
import re
from anonymisation.utils import *
from anonymisation.evaluation import *
from anonymisation.distance_computations import *
from anonymisation.disclosure_computations import record_linkage, val_to_rank_category, val_to_rank_numerical


def compute_metric(dataset, metric, k, taxonomies, raw_data=[]):
    if metric == 'granularity':
        return granularity(dataset, taxonomies, k)
    elif metric == 'intensity':
        return intensity(dataset, taxonomies, k)
    elif metric == 'discernibility':
        return discernability(dataset, k)
    elif metric == 'entropy':
        return non_uniform_entropy(dataset, k)
    elif metric == 'groupsize':
        return group_size(dataset)
    elif metric == 'sse':
        return SSE(raw_data, dataset, taxonomies)
    elif metric == 'record_linkage':
        return record_linkage(dataset, raw_data, taxonomies)
    raise ValueError('Metric "'+metric+'" specified is not a valid metric')


def compute_mean_metrics(path, iterations, metrics, k, taxonomies):
    """
    For each metric that has been optimised for, computes the mean metric for all metrics.
    :param path:
    :param iterations:
    :param metrics:
    :param k:
    :param taxonomies:
    :return:
    """
    all_results = np.zeros((len(metrics), len(metrics)))
    for r, metric in enumerate(metrics):
        metric_results = []
        for i in range(len(metrics)):
            metric_results.append([])
        for i in range(iterations):
            file_path = path+'/'+metric+'_'+str(i+1)+'.csv'
            dataset = read_data_path(file_path)[0]
            for i, s_metric in enumerate(metrics):
                result = compute_metric(dataset, s_metric, k, taxonomies)
                metric_results[i].append(result)

        for m, res in enumerate(metric_results):
            all_results[r, m] = np.mean(res)

    df = pd.DataFrame(all_results, columns=metrics, index=metrics)
    df.to_csv(path+'/result.csv')


def compute_mean_score_iterations(path, max_iterations, metrics, k, taxonomies):
    results = np.zeros((max_iterations, len(metrics)))

    for i in range(max_iterations+1):
        i = 100
        print('Iteration', i)
        iteration_result = []
        for l in range(len(metrics)):
            iteration_result.append([])
        for j in range(i):
            file_path = path + '/granularity_'+str(i)+'_'+str(j+1)+'.csv'
            dataset = read_data_path(file_path)[0]
            for m, metric in enumerate(metrics):
                result = compute_metric(dataset, metric, k, taxonomies)
                iteration_result[m].append(result)
        if len(iteration_result[0]) > 0:
            for m, res in enumerate(iteration_result):
                results[i-1, m] = np.mean(res)

    df = pd.DataFrame(results, columns=metrics)
    df.to_csv(path+'/result.csv')


def compute_score_percentage(path, score_data, score_original, score_suppressed):

    only_score = score_data.drop('optimised_metric', 1)
    metrics = only_score.columns
    normalised_data = np.zeros(only_score.values.shape)

    for i in range(len(metrics)):
        scores = only_score.values[i, :]
        for s, score in enumerate(scores):
            normalised_data[i, s] = normalise(only_score[i, s], score_suppressed[s], score_original[s])

    norm_df = pd.DataFrame(normalised_data, columns=metrics, index=metrics)
    norm_df.to_csv(path+'/normalised.csv')


def compute_score_boundaries(dataset_path, metrics, taxonomies):
    data_raw = read_data_path(dataset_path)[0]

    suppressed = [['*'] * len(taxonomies) for data in data_raw.values]
    data_suppressed = pd.DataFrame(suppressed, columns=data_raw.columns)

    original_score = []
    suppressed_score = []
    k = 103

    # TODO: Not sure which k to use for the original and suppressed score computations
    for metric in metrics:
        original_score.append(compute_metric(data_raw, metric, k, taxonomies))
        suppressed_score.append(compute_metric(data_suppressed, metric, k, taxonomies))

    pass


def compute_score_iterations_optimised(dataset_path, iterations, evaluation_metrics, data_metrics, epsilons, ks, taxonomies, original_set):

    for metric_data in data_metrics:
        for n, epsilon in enumerate(epsilons):
            print("####################### Metric "+metric_data+" epsilon "+str(epsilon))
            scores = []
            for m in evaluation_metrics:
                scores.append([])
            for i in range(iterations):
                filename = dataset_path+"/datasets/"+metric_data+"_eps-"+str(epsilon)+"_"+str(i+1)+".csv"
                anon_dataset = read_data_path(filename)[0]
                for m, metric in enumerate(evaluation_metrics):
                    score = compute_metric(anon_dataset, metric, ks[n], taxonomies, original_set)
                    scores[m].append(str(score))
            outputfile = dataset_path+"/result_"+metric_data+"_eps-"+str(epsilon)+".csv"
            scores_np = np.array(scores)
            df = pd.DataFrame(scores_np.T, columns=evaluation_metrics)
            df.to_csv(outputfile, index=False)


def compute_score_iterations(dataset_path, iterations, metrics, epsilons, ks, taxonomies, original_set):

    for n, epsilon in enumerate(epsilons):
        print("####################### Epsilon " + str(epsilon))
        scores = []
        for m in metrics:
            scores.append([])
        for i in range(iterations):
            filename = dataset_path + "/datasets/eps-" + str(epsilon) + "_" + str(i + 1)+".csv"
            anon_dataset = read_data_path(filename)[0]
            for m, metric in enumerate(metrics):
                score = compute_metric(anon_dataset, metric, ks[n], taxonomies, original_set)
                scores[m].append(str(score))
        outputfile = dataset_path + "/result_eps-" + str(epsilon) + ".csv"
        scores_np = np.array(scores)
        df = pd.DataFrame(scores_np.T, columns=metrics)
        df.to_csv(outputfile, index=False)


def compute_score_iterations2(dataset_path, output_path, all_filenames,  metrics, ks, taxonomies, original_set):

    if 'record_linkage' in metrics:
        attributes = original_set.columns
        attributes_rank = []

        for i, attr in enumerate(attributes):
            numeric = taxonomies[i].is_numeric()
            if numeric:
                attributes_rank.append(val_to_rank_numerical(original_set[attr]))
            else:
                attributes_rank.append(val_to_rank_category(original_set[attr], taxonomies[i]))
        print('Ranks found')

    for n, filenames in enumerate(all_filenames):
        scores = []
        for _ in metrics:
            scores.append([])
        filename_parts = filenames[0].split('_')
        outputfile = output_path+'/result_'+'_'.join(filename_parts[:-1])+'.csv'

        try:
            with open(outputfile) as file:
                pass
            print('File found')
        except FileNotFoundError:
            for c, filename in enumerate(filenames):
                dataset = pd.read_csv(dataset_path+'/'+filename)
                for m, metric in enumerate(metrics):
                    if metric == 'record_linkage':
                        score = record_linkage(dataset, original_set, taxonomies, attributes_rank)
                        print('File', c, 'of', len(filenames))
                    else:
                        score = compute_metric(dataset, metric, ks[n], taxonomies, original_set)
                    scores[m].append(str(score))


            scores_np = np.array(scores).T
            df = pd.DataFrame(scores_np, columns=metrics)
            try:
                df.to_csv(outputfile, index=False)
            except FileNotFoundError:
                os.mkdir(output_path)
                df.to_csv(outputfile, index=False)
            print("File", n, "done of", len(all_filenames))




def compute_metric_scores(dataset, metrics, k, taxonomies, original_set):
    scores = []
    for m, metric in enumerate(metrics):
        score = compute_metric(dataset, metric, k, taxonomies, original_set)
        scores.append(str(score))
    return scores


def compute_reference_scores(dataset):
    path_in = os.getcwd()
    pattern = '^.*/thesis-data-anonymisation/'
    path = re.search(pattern, path_in).group(0)
    original_dataset, attributes = read_data_path(
        path+'data/'+dataset+'/'+dataset+'.csv')

    suppressed_data = [['*']*len(attributes)]*len(original_dataset.values)
    suppressed_dataset = pd.DataFrame(suppressed_data, columns=original_dataset.columns)

    taxonomies = [create_taxonomy(dataset, attr) for attr in attributes]
    add_semantic_distances_all(taxonomies)
    for taxonomy in taxonomies:
        taxonomy.add_boundary(compute_boundary(taxonomy))

    metrics = ['discernibility', 'entropy', 'groupsize', 'sse']
    if dataset == 'adult':
        ks = [103, 72, 67, 61, 59, 54, 53, 50, 174, 5, 74, 88, 100, 114, 129, 141, 155, 170, 184, 199, 211, 225, 240, 252,
            266] + list(range(200, 4701, 300))
    elif dataset == 'housing':
        #ks = [103, 72, 67, 61, 59, 54, 53, 50, 143, 5, 74, 88, 100, 114, 129, 141, 155, 170, 184, 199, 211, 225, 240, 252,
        #    266]
        ks = [103, 72, 67, 61, 59, 54, 53, 50, 143, 5, 74, 88, 100, 114, 129, 141, 155, 170, 184, 199, 211, 225, 240,
              252, 266] + list(range(200, 6000, 300))
    else:
        raise RuntimeError('Unrecognised dataset "'+dataset+'"')

    all_original_scores = []
    all_suppressed_scores = []
    for k in ks:
        print("######## k:", k)
        original_score = compute_metric_scores(original_dataset, metrics, k, taxonomies, original_dataset)
        all_original_scores.append(original_score)
        suppressed_score = compute_metric_scores(suppressed_dataset, metrics, k, taxonomies, original_dataset)
        all_suppressed_scores.append(suppressed_score)
    original_scores = np.array(all_original_scores)
    suppressed_scores = np.array(all_suppressed_scores)

    original_scores_data = pd.DataFrame(original_scores, columns=metrics, index=ks)
    suppressed_scores_data = pd.DataFrame(suppressed_scores, columns=metrics, index=ks)

    original_scores_data.to_csv(
        path+'data/result/reference/'+dataset+'/original.csv')
    suppressed_scores_data.to_csv(
        path+'data/result/reference/'+dataset+'/suppressed.csv')


def add_reference_score(dataset, ks):
    all_original_scores = []
    all_suppressed_scores = []

    path_in = os.getcwd()
    pattern = '^.*/thesis-data-anonymisation/'
    path = re.search(pattern, path_in).group(0)

    original_dataset, attributes = read_data_path(
        path+'data/' + dataset + '/' + dataset + '.csv')

    suppressed_data = [['*'] * len(attributes)] * len(original_dataset.values)
    suppressed_dataset = pd.DataFrame(suppressed_data, columns=original_dataset.columns)

    taxonomies = [create_taxonomy(dataset, attr) for attr in attributes]
    add_semantic_distances_all(taxonomies)
    for taxonomy in taxonomies:
        taxonomy.add_boundary(compute_boundary(taxonomy))

    metrics = ['discernibility', 'entropy', 'groupsize', 'sse']

    outputpath_original = path+'data/result/reference/' + dataset + '/original.csv'
    outputpath_suppressed = path+'data/result/reference/' + dataset + '/suppressed.csv'

    prev_result_original = pd.read_csv(outputpath_original)
    prev_result_suppressed = pd.read_csv(outputpath_suppressed)

    for k in ks:
        print("######## k:", k)
        original_score = compute_metric_scores(original_dataset, metrics, k, taxonomies, original_dataset)
        all_original_scores.append(original_score)
        suppressed_score = compute_metric_scores(suppressed_dataset, metrics, k, taxonomies, original_dataset)
        all_suppressed_scores.append(suppressed_score)
    original_scores = np.array(all_original_scores)
    suppressed_scores = np.array(all_suppressed_scores)
    ks_add = np.array(ks).reshape((-1, 1))
    original_scores = np.concatenate((ks_add, original_scores), 1)
    suppressed_scores = np.concatenate((ks_add, suppressed_scores), 1)

    total_original = np.concatenate((prev_result_original.values, original_scores), 0)
    total_suppressed = np.concatenate((prev_result_suppressed.values, suppressed_scores), 0)

    original_scores_data = pd.DataFrame(total_original, columns=['k']+metrics)
    suppressed_scores_data = pd.DataFrame(total_suppressed, columns=['k']+metrics)


    original_scores_data.to_csv(
        path+'data/result/reference/' + dataset + '/original.csv')
    suppressed_scores_data.to_csv(
        path+'data/result/reference/' + dataset + '/suppressed.csv')


def add_reference_score_sub_attributes(dataset_name, ks, attributes):
    # TODO: NOT DONE!!!
    path_in = os.getcwd()
    pattern = '^.*/thesis-data-anonymisation/'
    path = re.search(pattern, path_in).group(0)

    datasets_path = path + 'data/'+dataset_name+'/attribute_subsets'
    metrics = ['discernibility', 'entropy', 'groupsize', 'sse']

    taxonomies = [create_taxonomy(dataset_name, attr) for attr in attributes]
    add_semantic_distances(taxonomies)
    for taxonomy in taxonomies:
        taxonomy.add_boundary(compute_boundary(taxonomy))

    attr_to_idx = {attr: n for n, attr in enumerate(attributes)}

    if dataset_name == 'adult':
        attr_range = range(2, 9)
    elif dataset_name == 'housing':
        attr_range = range(2, 10)
    elif dataset_name == 'musk':
        attr_range = range(2, 21)
    else:
        raise RuntimeError('Does not recognise dataset', dataset_name)


    for a in attr_range:
        files_path = datasets_path+'/'+str(a)
        files = os.listdir(files_path)
        for file in files:
            original_outputfile = path+'data/result/reference/'+dataset_name+'/'+str(a)+'/original_'+file
            suppressed_outputfile = path+'data/result/reference/'+dataset_name+'/'+str(a)+'/suppressed_'+file

            try:
                with open(original_outputfile) as file:
                    pass
                with open(suppressed_outputfile) as file:
                    pass
                print('File found')
            except FileNotFoundError:
                original_dataset, current_attrs = read_data_path(files_path+'/'+file)
                suppressed_data = [['*'] * len(current_attrs)] * len(original_dataset.values)
                suppressed_dataset = pd.DataFrame(suppressed_data, columns=original_dataset.columns)

                current_taxomomies = [taxonomies[attr_to_idx[attr]] for attr in current_attrs]

                all_original_scores = []
                all_suppressed_scores = []
                for k in ks:
                    original_score = compute_metric_scores(original_dataset, metrics, k, current_taxomomies, original_dataset)
                    all_original_scores.append([k]+original_score)
                    suppressed_score = compute_metric_scores(suppressed_dataset, metrics, k, current_taxomomies, original_dataset)
                    all_suppressed_scores.append([k]+suppressed_score)

                original_scores = np.array(all_original_scores)
                suppressed_scores = np.array(all_suppressed_scores)
                original_scores_data = pd.DataFrame(original_scores, columns=['k']+metrics)
                suppressed_scores_data = pd.DataFrame(suppressed_scores, columns=['k']+metrics)

                original_scores_data.to_csv(original_outputfile, index=False)
                suppressed_scores_data.to_csv(suppressed_outputfile, index=False)
        print('Attribute', a, 'done')


def compute_reference_scores_attribute_subsets(dataset_name, attributes):
    path_in = os.getcwd()
    pattern = '^.*/thesis-data-anonymisation/'
    path = re.search(pattern, path_in).group(0)

    datasets_path = path+'data/'+dataset_name+'/attribute_subsets'
    metrics = ['discernibility', 'entropy', 'groupsize', 'sse']

    if dataset_name == 'adult':
        ks = [59, 174]
        attr_range = range(2, 9)
    elif dataset_name == 'housing':
        ks = [59, 143]
        attr_range = range(2, 10)
    elif dataset_name == 'musk':
        ks = [45, 82]
        attr_range = range(2, 21)
    else:
        raise RuntimeError('Does not recognise dataset', dataset_name)

    taxonomies = [create_taxonomy(dataset_name, attr) for attr in attributes]
    add_semantic_distances(taxonomies)
    for taxonomy in taxonomies:
        taxonomy.add_boundary(compute_boundary(taxonomy))

    attr_to_idx = {attr: n for n, attr in enumerate(attributes)}

    for a in attr_range:
        files_path = datasets_path+'/'+str(a)
        files = os.listdir(files_path)
        for file in files:
            original_outputfile = path+'data/result/reference/'+dataset_name+'/'+str(a)+'/original_'+file
            suppressed_outputfile = path+'data/result/reference/'+dataset_name+'/'+str(a)+'/suppressed_'+file

            try:
                with open(original_outputfile) as file:
                    pass
                with open(suppressed_outputfile) as file:
                    pass
                print('File found')
            except FileNotFoundError:
                original_dataset, current_attrs = read_data_path(files_path+'/'+file)
                suppressed_data = [['*'] * len(current_attrs)] * len(original_dataset.values)
                suppressed_dataset = pd.DataFrame(suppressed_data, columns=original_dataset.columns)

                current_taxomomies = [taxonomies[attr_to_idx[attr]] for attr in current_attrs]

                all_original_scores = []
                all_suppressed_scores = []
                for k in ks:
                    original_score = compute_metric_scores(original_dataset, metrics, k, current_taxomomies, original_dataset)
                    all_original_scores.append([k]+original_score)
                    suppressed_score = compute_metric_scores(suppressed_dataset, metrics, k, current_taxomomies, original_dataset)
                    all_suppressed_scores.append([k]+suppressed_score)

                original_scores = np.array(all_original_scores)
                suppressed_scores = np.array(all_suppressed_scores)
                original_scores_data = pd.DataFrame(original_scores, columns=['k']+metrics)
                suppressed_scores_data = pd.DataFrame(suppressed_scores, columns=['k']+metrics)

                original_scores_data.to_csv(original_outputfile, index=False)
                suppressed_scores_data.to_csv(suppressed_outputfile, index=False)
        print('Attribute', a, 'done')


def compute_mean_var(result_path, file_names):
    vals = ['mean', 'std']
    for file_name in file_names:
        dataset = pd.read_csv(result_path+'/'+file_name)
        mean = np.mean(dataset.values, axis=0).reshape(1, -1)
        std = np.std(dataset.values, axis=0).reshape(1, -1)
        concat = np.concatenate((mean, std), axis=0)
        df = pd.DataFrame(concat, index=vals, columns=dataset.columns)
        try:
            df.to_csv(result_path+'/mean_std/mean_std_'+file_name)
        except FileNotFoundError:
            os.mkdir(result_path+'/mean_std')
            df.to_csv(result_path + '/mean_std/mean_std_' + file_name)
    return


def compute_count(dataset, counts, iterations):
    attributes = dataset.columns

    for attribute in attributes:
        attr_count = dataset[attribute].value_counts()
        for v, count in enumerate(attr_count):
            val = str(attr_count.index.values[v])
            try:
                counts[attribute][val] += (count / iterations)
            except KeyError:
                counts[attribute][val] = (count / iterations)

    return counts


def compute_hist_vals(dataset_name, model, filename_parts, iterations):
    path_in = os.getcwd()
    pattern = '^.*/thesis-data-anonymisation/'
    path = re.search(pattern, path_in).group(0)
    try:
        dataset = pd.read_csv(path+'data/'+dataset_name+'/'+dataset_name+
                              '_cropped.csv')
    except FileNotFoundError:
        dataset = pd.read_csv(
            path+'data/' + dataset_name + '/' + dataset_name + '.csv')

    attributes = dataset.columns

    model_path = path+'data/result/'+model+'_test/1_1/'+dataset_name
    dataset_path = model_path+'/datasets'

    filename_combos = list(itertools.product(*filename_parts))

    for filename_combo in filename_combos:
        filename = "".join(filename_combo)
        counts = {attribute: {} for attribute in attributes}
        if iterations > 1:
            for i in range(iterations):
                dataset = pd.read_csv(dataset_path+'/'+filename+'_'+str(i+1)+'.csv')
                counts = compute_count(dataset, counts, iterations)
        else:
            dataset = pd.read_csv(dataset_path + '/' + filename+'.csv')
            counts = compute_count(dataset, counts, iterations)

        with open(model_path+'/histograms/hist_'+filename+'.pkl', 'wb') as file:
            pickle.dump(counts, file)
    return


def generate_filename_combos(prefix, postfix, filename_parts, iterations=1):

    combinations = list(itertools.product(*filename_parts))

    filenames = [prefix + '_'.join(file_part) for file_part in combinations]
    if iterations > 1:
        file_combinations = [[filename+"_"+str(i+1)+postfix for i in range(iterations)] for filename in filenames]
    else:
        file_combinations = [filename+postfix for filename in filenames]

    return file_combinations


if __name__ == '__main__':
    compute_reference_scores('housing')
    housing_attributes = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
                  "population", "households", "median_income", "median_house_value"]

    musk_attributes = [str(i+1) for i in range(20)]

    #compute_reference_scores_attribute_subsets('musk', musk_attributes)


    #result_root = path+'data/result'
    #epsilons = [2.0, 1.5, 1.25, 1.0986122886681098, 1.0, 0.75, 0.6931471805599453, 0.5, 0.1, 0.01]
    #metrics = ['granularity', 'intensity', 'discernibility', 'entropy', 'groupsize']
    #deltas = ['1.0E-5', '1.0E-6', '1.0E-7', '1.0E-8', '1.0E-9', '1.0E-10', '1.0E-11', '1.0E-12', '1.0E-13',
    #                  '1.0E-14', '1.0E-15', '1.0E-16', '1.0E-17', '1.0E-18', '1.0E-19', '1.0E-20']

    #ks = [str(k) for k in list(range(150, 301, 10))]
    #filenames = ['num_suppressed_attributes_'+metric+'.csv' for metric in metrics]

    #dataset, attributes = read_data_path(path+'data/adult/adult_cropped2.csv')
    #
    #taxonomies = [create_taxonomy('adult', attribute) for attribute in attributes]
    #add_semantic_distances(taxonomies)
    #for taxonomy in taxonomies:
    #    taxonomy.add_boundary(compute_boundary(taxonomy))
    #
    #record1 = dataset.values[0]
    #
    #dist_suppressed = distance(record1, ['*']*len(taxonomies), taxonomies)
    #dist_semi_suppressed1 = distance(record1, ['36-55','gov','*','Never-married','*','White','Male','N.America'], taxonomies)
    #dist_semi_suppressed2 = distance(record1, ['16-55','gov','*','Never-married2','*','White','Male','N.America'], taxonomies)
    #dist_semi_suppressed3 = distance(record1, ['16-55','*','13-16','Never-married2','*','*','Male','*'], taxonomies)
    #dist_semi_suppressed4 = distance(record1, ['16-55','*','*','*','*','*','Male','*'], taxonomies)
    #dist_semi_suppressed5 = distance(record1, ['16-55','*','*','*','*','*','*','*'], taxonomies)

    #record1 = dataset.values[0]
    #record2 = dataset.values[1]
    #record3 = dataset.values[2]
#
    #add_semantic_distances(taxonomies)
    #for taxonomy in taxonomies:
    #    taxonomy.add_boundary(compute_boundary(taxonomy))
#
    #dist1 = distance(record1, record2, taxonomies)
    #dist2 = distance(record2, record3, taxonomies)
#
    #node_dist1 = node_distance(taxonomies[2].nodes['35.0-42.0'], taxonomies[2].nodes['35.0'])
    #node_dist2 = node_distance(taxonomies[2].nodes['35.0'], taxonomies[2].nodes['35.2'])
    pass
