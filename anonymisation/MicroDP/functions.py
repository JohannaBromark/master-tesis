import numpy as np
import math
import itertools
from anonymisation.utils import *
from anonymisation.distance_computations import *
import time
import random
import os
import re


# --------- Function for computing e-diff private centroids for clusters with categorical attributes


def save_clusters(filename, clusters):

    with open(filename, 'w') as file:
        for cluster in clusters:
            cluster_s = [str(n) for n in cluster]
            cluster_string = ''
            cluster_string += ','.join(cluster_s)+'\n'
            file.write(cluster_string)


def read_clusters(filename):
    clusters = []
    with open(filename, 'r') as file:
        file_lines = file.readlines()

    for line in file_lines:
        l = line.split(',')
        l_int = [int(n) for n in l]
        clusters.append(l_int)

    return clusters


def quality_criterion(dataset, attr_val, attr_idx, taxonomy):
    """
    The quality criterion defined in Algorithm 4 in Soria-Comas 2014.
    :param dataset: ndarray
    :param attr_val:
    :param attr_idx:
    :param taxonomy:
    :return:
    """

    attr_values = set(dataset[:, attr_idx])
    return - marginality(attr_val, taxonomy, attr_values)


def exponential_mechanism(cluster, epsilon, n, k, attr_idx, taxonomy):
    boundary = taxonomy.boundary
    sensitivity = node_distance(taxonomy.nodes[boundary[0]], taxonomy.nodes[boundary[1]])
    distribution = []
    for attr_val in taxonomy.get_domain():
        q = quality_criterion(cluster, attr_val, attr_idx, taxonomy)
        distribution.append(math.exp((epsilon*q)/(2*(n/k)*sensitivity)))

    return distribution


# -------------- Microaggregation


def find_cluster_records(dataset, ref_point, k, taxonomies, verbose=False):
    distances = []
    start = time.time()
    for i in range(len(dataset)-1):
        #print(i)
        distances.append(distance(dataset[i], ref_point, taxonomies))
    end = time.time()
    if verbose:
        print('Time: ', end-start)
    k_smallest = np.argpartition(distances, k)[:k]
    return k_smallest


def hamming_distance(x1, x2):
    return sum(abs(i-j) for i, j in zip(x1, x2))


def find_next_R(used_references, R):
    current_R = R.copy()
    new_r = ()
    prev = -1
    while len(new_r) == 0:
        if prev >= -len(used_references):
            prev_r = used_references[prev]
            hamming_distances = np.array([hamming_distance(prev_r, a) for a in current_R])
            max_val = np.max(hamming_distances)
            max_idx = np.where(hamming_distances == max_val)[0]
            if len(max_idx) > 1:
                new_current_R = [current_R[idx] for idx in max_idx]
                current_R = new_current_R
                prev -= 1
            else:
                new_r = current_R[max_idx[0]]
        else:
            new_r = current_R[0]
    return new_r


def e_diff_category_centroid(cluster, taxonomy, eps, n, k):
    boundary = taxonomy.boundary
    marginal_sensitivity = node_distance(taxonomy.nodes[boundary[0]], taxonomy.nodes[boundary[1]])
    domain = taxonomy.get_domain()
    exp_mechanism_vals = []

    for attr_val in domain:
        marg = marginality(attr_val, taxonomy, cluster)
        p = np.exp((eps * -marg)/(2 * (n/k) * marginal_sensitivity))
        exp_mechanism_vals.append(p)

    sum_exp = sum(exp_mechanism_vals)
    weights = [val/sum_exp for val in exp_mechanism_vals]

    attribute = np.random.choice(domain, p=weights)  # np.random seem to work better than random

    return attribute


def find_centroid(cluster_values, taxonomies, eps, n, k, add_cluster_noise):
    centroid = []
    for i in range(len(taxonomies)):
        numeric = taxonomies[i].is_numeric()
        if numeric:
            try: # Need to solve floats ans specific case for temperatures
                mean = float(np.mean(cluster_values[:, i]))
            except TypeError:
                unique, counts = np.unique(cluster_values[:, i], return_counts=True)
                num_values = dict(zip(unique, counts))
                try:
                    num_ranges = num_values['35.0_42.0']
                    if num_ranges > len(cluster_values[:, i])/2:
                        mean = '35.0_42.0'
                    else:
                        temp_cluster = cluster_values[:, i].copy()
                        idx_to_remove = np.where(temp_cluster == '35.0_42.0')

                        temp_cluster = np.delete(temp_cluster, idx_to_remove).astype(float)
                        mean = round(np.mean(temp_cluster), 1)
                except KeyError:
                    mean = round(np.mean(cluster_values[:, i].astype(float)), 1)
        else:
            # TODO: Use Algorithm 4 to find categorical cluster values
            # TODO: Test idea that algorithm 4 should be performed on already selected centroids
            if add_cluster_noise:
                mean = e_diff_category_centroid(cluster_values[:, i], taxonomies[i], eps, n, k)
            else:
                attr_vals = list(set(cluster_values[:, i]))
                marginalities = []
                for attr in attr_vals:
                    marginalities.append(marginality(attr, taxonomies[i], cluster_values[:, i]))
                mean_idx = np.argmin(marginalities)
                mean = attr_vals[int(mean_idx)]
        centroid.append(mean)

    return centroid


def find_negation(combinations):
    combo_list = [list(combo) for combo in combinations]
    array = np.array(combo_list)
    negate_array = (array+1)%2

    negate_combos = [tuple(row) for row in negate_array]
    negate_combos.reverse()

    return negate_combos

def microaggregation(dataset, k, taxonomies, eps, read=False, add_cluster_noise=False, cluster_path=None):
    """
    Insensitive microaggregation as descriped in 4.3 in SC 2014
    :param dataset:
    :param k:
    :param taxonomies:
    :return: Clusters from microaggregation
    """

    # Add c a olumn with record_id so that they can be matched with those in the clusters.
    X = dataset.copy()
    X['record_id'] = range(len(dataset.values))

    boundaries = [tax.boundary for tax in taxonomies]
    used_references = []

    if cluster_path is None:
        cluster_path = 'clusters_adult_' + str(k) + '.csv'

    try:
        clusters = read_clusters(cluster_path)
        read = True
    except FileNotFoundError:
        read = False

    if not read:
        print("Finding clusters...")
        R = []
        clusters = []
        orig_size = len(dataset.values) // k + 1000
        current_end = orig_size
        combos = itertools.product([0, 1], repeat=len(taxonomies))
        R_orig = sorted(list(combos))

        #if len(taxonomies) < 15:
        #    R_orig = sorted(list(combos))
        #else:
        #    R_orig_start = []
        #    for i, combo in enumerate(combos):
        #        if i >= orig_size:
        #            break
        #        R_orig_start.append(combo)
#
        #    R_orig_end = find_negation(R_orig_start)
        #    R_orig = R_orig_start + R_orig_end

        while len(X.values) >= 2*k:
            if len(R) < 1:
                R = R_orig.copy()
                r = R[0]
                R.remove(r)
                current_end += orig_size
            #if len(R) < 1 and current_end == orig_size:
            #    R = R_orig.copy()
            #    r = R[0]
            #    R.remove(r)
            #    current_end += orig_size
            #elif len(R) < 1 and current_end > orig_size:
            #    print("Restarting R")
            #    R_start = []
            #    for i, combo in enumerate(combos):
            #        if i >= current_end+orig_size:
            #            break
            #        if i >= current_end:
            #            R_start.append(combo)
            #    current_end += orig_size
            #    R_end = find_negation(R_start)
#
            #    R = R_start + R_end
            #    r = R[0]
            #    R.remove(r)

            record = [boundaries[i][a] for i, a in zip(range(len(taxonomies)), r)]  # The record at the current boundary

            C_idx = find_cluster_records(X.values, record, k, taxonomies)  # Indices for the k closest record to reference
            correct_idx = X.values[C_idx, :][:, len(taxonomies)].astype('int64')  # The corresponding indices in dataset
            clusters.append(correct_idx)
            X = X.drop(correct_idx)

            # Update reference point
            used_references.append(r)
            r = find_next_R(used_references, R)
            R.remove(r)
        print("Clusters found")

        # Add the records that are left to a cluster
        clusters.append(X.values[:, len(taxonomies)].astype('int64'))
        save_clusters(cluster_path, clusters)
    else:
        clusters = read_clusters(cluster_path)

    X_bar_data = np.zeros(dataset.values.shape).astype('object')

    for cluster in clusters:
        centroid = np.array(find_centroid(dataset.values[cluster], taxonomies, eps, len(dataset), k, add_cluster_noise))
        for c in cluster:
            X_bar_data[c, :] = centroid

    X_bar = pd.DataFrame(X_bar_data, columns=dataset.columns)  # Some clusters get the same centroid

    #partitions = X_bar.groupby(list(dataset.columns)).groups
    #partition_length = [len(part) for part in partitions.values()]

    return X_bar


def sanitise(dataset, eps, k, taxonomies, float_vals, num_decimals):
    # Add noise to numerical data
    noise_data = dataset.copy()
    boundaries = [taxonomy.boundary for taxonomy in taxonomies]
    for i in range(len(taxonomies)):
        numeric = taxonomies[i].is_numeric()
        if numeric:
            range_exists = False
            if float_vals[i]:
                float_val = True
                try:
                    noise_data[:, i].astype('float')
                    range_idx = []
                except ValueError:
                    range_idx = np.where(noise_data[:, i] == '35.0_42.0')[0]
                    if len(range_idx) > 0:
                        noise_data[range_idx, i] = 0
                        range_exists = True
                    noise_data[:, i].astype(float)
            else:
                float_val = False
                range_idx = []

            if float_val:
                sensitivity = float(boundaries[i][1]) - float(boundaries[i][0])
            else:
                sensitivity = int(boundaries[i][1]) - int(boundaries[i][0])
            scale = sensitivity/(eps*k)
            noise = np.random.laplace(0, scale, len(dataset))

            for j in range(len(dataset)):
                if float_val:
                    new_val = round((noise[j] + float(noise_data[j, i])), num_decimals[i])
                    if new_val < float(boundaries[i][0]):
                        new_val = float(boundaries[i][0])
                    elif new_val > float(boundaries[i][1]):
                        new_val = float(boundaries[i][1])
                    noise_data[j, i] = new_val
                else:
                    new_val = int(round(float(noise[j]), 0) + round(float(noise_data[j, i]), 0))
                    # Restrict the value to be within the attribute range
                    if new_val < int(boundaries[i][0]):
                        new_val = int(boundaries[i][0])
                    elif new_val > int(boundaries[i][1]):
                        new_val = int(boundaries[i][1])
                    noise_data[j, i] = new_val
            if range_exists:
                noise_data[range_idx, i] = '35.0_42.0'

    return noise_data


def main():
    dataset_name = 'adult'
    path_in = os.getcwd()
    pattern = '^.*/thesis-data-anonymisation/'
    path = re.search(pattern, path_in).group(0)

    dataset_path = path + 'data/' + dataset_name + '/' + dataset_name + '.csv'

    dataset, attributes = read_data_path(dataset_path)
    taxonomies = [create_taxonomy(dataset, attr) for attr in attributes]
    add_semantic_distances(taxonomies)
    for taxonomy in taxonomies:
        taxonomy.add_boundary(compute_boundary(taxonomy))
    eps = 2.0
    k = 500
    cluster_path = path+'anonymisation/S_C/clusters/'+dataset_name+'/k_'+str(k)+'.csv'
    X_bar = microaggregation(dataset, k, taxonomies, eps, add_cluster_noise=True, cluster_path=cluster_path)
    anon_data = sanitise(X_bar.values, eps, k, taxonomies, [False]*8, [0]*8)

    #anonymised = pd.DataFrame(anon_data, columns=data.columns)



if __name__ == '__main__':
    main()
