import numpy as np
import pandas as pd
import os
import queue
import time
from anonymisation.Taxonomy import *
from anonymisation.utils import read_data_path, create_taxonomy
from anonymisation.distance_computations import add_semantic_distances_all, compute_boundary, marginality, node_distance


def find_numerical_ranks(X, Y):

    original_ranks = rank_to_val_numerical(X)
    anonymous_ranks = val_to_rank_numerical(Y)

    return original_ranks, anonymous_ranks


def rank_to_val_numerical(attribute_values):
    rank_to_val = {}
    sorted_values = sorted(attribute_values)

    if sorted_values[0] == '35.0_42.0':
        spec_idx = np.where(np.array(sorted_values) == '35.0_42.0')[0]
        other_idx = np.where(np.array(sorted_values) != '35.0_42.0')[0]
        spec_values = np.array(sorted_values)[spec_idx]
        other_values = np.array(sorted_values)[other_idx]
        sorted_values = np.concatenate((other_values, spec_values))

    for i in range(len(sorted_values)):
        val = str(sorted_values[i])
        rank = i+1
        rank_to_val[rank] = val

    return rank_to_val


def val_to_rank_numerical(attributes_values):
    val_to_rank = {}
    sorted_values = sorted(attributes_values)

    if sorted_values[0] == '35.0_42.0':
        spec_idx = np.where(np.array(sorted_values) == '35.0_42.0')[0]
        other_idx = np.where(np.array(sorted_values) != '35.0_42.0')[0]
        spec_values = np.array(sorted_values)[spec_idx]
        other_values = np.array(sorted_values)[other_idx]
        sorted_values = np.concatenate((other_values, spec_values))

    for i in range(len(sorted_values)):
        val = str(sorted_values[i])
        rank = i + 1
        try:
            val_to_rank[val].put(rank)
        except KeyError:
            val_to_rank[val] = queue.Queue()
            val_to_rank[val].put(rank)

    return val_to_rank


def find_categorical_ranks(X, Y, taxonomy):

    original_ranks = rank_to_val_category(X, taxonomy)
    anonymous_ranks = val_to_rank_category(Y, taxonomy)

    return original_ranks, anonymous_ranks


def rank_to_val_category(attribute_values, taxonomy):
    rank_to_val = {}
    val_to_marg, marg_to_val = marginality_mappings(attribute_values, taxonomy)

    sorted_values = sorted([val_to_marg[val] for val in attribute_values])

    for i in range(len(sorted_values)):
        val = marg_to_val[sorted_values[i]]
        rank = i+1
        rank_to_val[rank] = val

    return rank_to_val


def val_to_rank_category(attribute_values, taxonomy):
    val_to_rank = {}
    val_to_marg, marg_to_val = marginality_mappings(attribute_values, taxonomy)

    sorted_values = sorted([val_to_marg[val] for val in attribute_values])

    for i in range(len(sorted_values)):
        val = marg_to_val[sorted_values[i]]
        rank = i + 1
        try:
            val_to_rank[val].put(rank)
        except KeyError:
            val_to_rank[val] = queue.Queue()
            val_to_rank[val].put(rank)

    return val_to_rank


def marginality_mappings(attribute_values, taxonomy):
    unique = np.unique(attribute_values)
    val_to_marg = {val: marginality(val, taxonomy, attribute_values) for val in unique}

    marg_to_val = {}

    for val in val_to_marg:
        not_added = True
        marg = val_to_marg[val]
        while not_added:
            try:
                marg_to_val[marg]
                marg += 1
            except KeyError:
                marg_to_val[marg] = val
                val_to_marg[val] = marg
                not_added = False
    return val_to_marg, marg_to_val


def find_nearest_value_numeric(value, array):
    try:
        idx = np.argmin(np.abs(array.values.astype(float) - float(value)))
    except ValueError:
        # Handle the case with 35.0_42.0
        if value == '35.0_42.0':
            idxs = np.where(array.values == value)[0]
            if len(idxs) > 0:
                idx = idxs[0]
            else:
                # The dataset does not contain 35.0_42.0, find the value closest to center
                new_val = (42+35)/2
                idx = np.argmin(np.abs(array.values.astype(float) - float(new_val)))
        else:
            # The array contains 35.0_42.0
            strange_indices = np.where(array.values == '35.0_42.0')[0]
            new_array = array.values.copy()
            new_array[strange_indices] = 0
            idx = np.argmin(np.abs(new_array.astype(float) - float(value)))

    return array[idx]


def find_nearest_value_category(value, array, taxonomy):
    distances = []
    for val in array:
        dist = abs(node_distance(taxonomy.nodes[val], taxonomy.nodes[value]))
        distances.append(dist)
    idx = np.argmin(distances)
    return array[idx]


def find_record(anonymised_dataset, y_stars, val_to_rank, return_index=False, verbose=False):
    d = 0
    not_done = True
    found_records = []
    record_idx = []
    while not_done:
        current_min_rank = -1
        for idx, anon_record in enumerate(anonymised_dataset.values):
            condition_hold = 0
            for i in range(len(anon_record)):
                attr_rank = get_rank(val_to_rank[i], anon_record[i])
                if current_min_rank > -1:
                    if d < attr_rank < current_min_rank:
                        current_min_rank = attr_rank
                else:
                    if attr_rank > d:
                        current_min_rank = attr_rank
                if abs(attr_rank - get_rank(val_to_rank[i], y_stars[i])) <= d:
                    condition_hold += 1
            if condition_hold == len(anon_record):
                found_records.append(anon_record)
                record_idx.append(idx)
                not_done = False
        if (current_min_rank - d) > 1:
            if verbose:
                print('Skipping from', d, 'to', current_min_rank)
            d = current_min_rank
        else:
            d += 1
    if return_index:
        return found_records, record_idx
    else:
        return found_records


def get_rank(val_to_rank, value):
    if len(val_to_rank[value].queue) == 1:
        return val_to_rank[value].queue[0]
    else:
        return val_to_rank[value].get(False)


def reverse_mapping(original_dataset, anonymous_dataset, taxonomies):
    reversed_dataset_list = []

    attributes = original_dataset.columns
    for a, attr in enumerate(attributes):
        numerical = taxonomies[a].is_numeric()
        if numerical:
            original_ranks_to_vals, anonymous_vals_to_ranks = find_numerical_ranks(original_dataset[attr], anonymous_dataset[attr])
        else:
            original_ranks_to_vals, anonymous_vals_to_ranks = find_categorical_ranks(original_dataset[attr], anonymous_dataset[attr],
                                                                    taxonomies[a])
        reversed_attributes = []

        for anon_val in anonymous_dataset[attr]:
            reversed_attributes.append(original_ranks_to_vals[anonymous_vals_to_ranks[str(anon_val)].get(False)])
        reversed_dataset_list.append(reversed_attributes)

    reversed_dataset = pd.DataFrame(np.array(reversed_dataset_list).T, columns=attributes)

    return reversed_dataset


def permutation_distance(original_record, anonymous_dataset, taxonomies, attributes_rank=None, record_index=-1):

    y_stars = []
    if attributes_rank is None:
        attributes_rank = []
    attributes = anonymous_dataset.columns

    for j, attr in enumerate(attributes):
        x_val = original_record[j]
        Y_j = anonymous_dataset[attr]
        numerical = taxonomies[j].is_numeric()
        if numerical:
            y_star = find_nearest_value_numeric(x_val, Y_j)
            if len(attributes_rank) < len(attributes):
                val_to_rank = val_to_rank_numerical(Y_j)
                attributes_rank.append(val_to_rank)
        else:
            y_star = find_nearest_value_category(x_val, Y_j, taxonomies[j])
            if len(attributes_rank) < len(attributes):
                val_to_rank = val_to_rank_category(Y_j, taxonomies[j])
                attributes_rank.append(val_to_rank)
        y_stars.append(y_star)

    if record_index > -1:
        anon_records = [anonymous_dataset.values[record_index]]
        found_index = record_index
    else:
        #print('Finding records...')
        anon_records, found_index = find_record(anonymous_dataset, y_stars, attributes_rank, True)
        #print('Records found')
    d = []
    for j, val in enumerate(anon_records[0]):
        rank_p = get_rank(attributes_rank[j], val)
        rank_y = get_rank(attributes_rank[j], y_stars[j])
        d.append(abs(rank_p-rank_y))

    return d, found_index


def record_linkage(anonymous_dataset, original_dataset, taxonomies, attributes_rank=None):
    # do a revesed_mapping
    # Compute shortest permutation distance for each record in anonymous reverted to original dataset, can be several
    # Choose the record with the shortest (max(permutation_distance)), that is the linkage
    # A correct linkage the and only the correct record is linked to correct record.

    # TODO: Painfully slow
    Z = reverse_mapping(original_dataset, anonymous_dataset, taxonomies)
    #print('Reverse mapping found')
    permutation_distances = []
    matched_records = []

    if attributes_rank is None:
        attributes = original_dataset.columns
        attributes_rank = []
        for i, attr in enumerate(attributes):
            numeric = taxonomies[i].is_numeric()
            if numeric:
                attributes_rank.append(val_to_rank_numerical(original_dataset[attr]))
            else:
                attributes_rank.append(val_to_rank_category(original_dataset[attr], taxonomies[i]))

    for i, val in enumerate(original_dataset.values):
        distance, records = permutation_distance(val, Z, taxonomies, attributes_rank)
        permutation_distances.append(distance)
        matched_records.append(records)
        print('record', i, 'of', len(original_dataset.values))
    #print('Distances found')
    num_matched = 0
    for i, record_index in enumerate(matched_records):
        if len(record_index) == 1:
            if i in record_index:
                num_matched += 1

    return num_matched/len(original_dataset.values)


def main():
    data_path = os.getcwd() + '/../data'

    dataset_name = 'housing'
    original_dataset, attributes = read_data_path(data_path+'/'+dataset_name+'/'+dataset_name+'.csv')
    anonymised_dataset = pd.read_csv(data_path+'/result/sc_test/1_1/'+dataset_name+'/datasets/eps-1.0_1.csv')

    #dataset_postfix = '/test_dataset/test_dataset.csv'
    #anonymised_postfix = '/test_dataset/test_anonymised.csv'
    #original_dataset, attributes = read_data_path(data_path + dataset_postfix)
    #anonymised_dataset = pd.read_csv(data_path + anonymised_postfix)
    #taxonomies = [Taxonomy({'*': [TaxNode('*', None, True)]}, 1) for i in range(len(attributes))]

    cropped_dataset = original_dataset.values[:1000]
    original_dataset = pd.DataFrame(cropped_dataset, columns=attributes)

    cropped_dataset_a = anonymised_dataset.values[:1000]
    anonymised_dataset = pd.DataFrame(cropped_dataset_a, columns=attributes)

    taxonomies = [create_taxonomy(dataset_name, attr) for attr in attributes]
    add_semantic_distances_all(taxonomies)
    for taxonomy in taxonomies:
        taxonomy.add_boundary(compute_boundary(taxonomy))
    print('Taxonomies found')

    # reverse_mapping(original_dataset, anonymised_dataset, taxonomies)
    #permutation_distance(original_dataset.values[2], anonymised_dataset, taxonomies)
    start_time = time.time()
    record_linkages = record_linkage(anonymised_dataset, original_dataset, taxonomies)
    print('record linkage', record_linkages, 'time:', time.time()-start_time)
    return


if __name__ == '__main__':
    main()