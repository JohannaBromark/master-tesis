import pandas as pd
import numpy as np
import itertools
import os
import re

from anonymisation.Taxonomy import *


def create_taxonomy(dataset_name, attr, dataset=[]):
    """
    Creates hierarchy based on the corresponding file in the correct dataset folder.
    :param dataset_name: (str) The name of the dataset to use
    :param attr: (str) The name of the attribute to create the taxonomy for
    :return: (Taxonomy) The taxonomy
    """
    #path = os.getcwd()

    path_in = os.getcwd()
    pattern = '^.*/thesis-data-anonymisation/'
    path_top = re.search(pattern, path_in).group(0)

    path = path_top +'data'

    if len(dataset_name) > 0:
        prefix = '../data/'+dataset_name+'/hierarchy_'
    else:
        prefix = '../data/hierarchy_'

    postfix = '.csv'

    try:
        file = open(path + '/' + prefix + attr + postfix, 'r')
    except FileNotFoundError:
        if len(dataset_name) > 0:
            prefix = '/data/'+dataset_name+'/hierarchy_'
        else:
            prefix = '/data/hierarchy_'
        file = open(path+prefix + attr + postfix, 'r')

    taxonomy = {}
    #dataset_group = dataset.groupby(attr).groups

    lines_in = file.readlines()
    file.close()
    lines = [line.strip().split(';') for line in lines_in]
    max_height = max([len(line) for line in lines])
    try:
        float(lines[0][0])
        is_numeric = True
    except ValueError:
        is_numeric = False
    for line in lines:
        #try:
        #    if is_numeric:
        #        dataset_group[int(line[0])]
        #    else:
        #        dataset_group[line[0]]
        #except KeyError:
        #    continue
        line.reverse()
        for i, val in enumerate(line):
            is_leaf = False
            if val == '*':
                node = TaxNode(val, None, is_numeric, is_leaf)
            else:
                if i == len(line) - 1:
                    is_leaf = True

                node = TaxNode(val, taxonomy[line[i - 1]][-1], is_numeric, is_leaf)
            try:
                current_nodes = taxonomy[val]
                already_added = False
                for current_node in current_nodes:
                    if current_node.parent is None:
                        already_added = True
                    elif current_node.parent.value == node.parent.value:
                        already_added = True
                if not already_added:
                    taxonomy[val].append(node)
            except KeyError:
                taxonomy[val] = [node]  # Saves the nodes in a list in case of several parents (only valid for nodes with several parents!!!)
    hierarchy = Taxonomy(taxonomy, max_height)

    return hierarchy


def read_data(dataset_name, attributes_to_drop=None):
    filename = dataset_name+'.csv'

    path_in = os.getcwd()
    pattern = '^.*/thesis-data-anonymisation/'
    path = re.search(pattern, path_in).group(0)
    data_frame = pd.read_csv(path+'data/'+dataset_name+'/'+filename)

    if attributes_to_drop is not None:
        for attr in attributes_to_drop:
            data_frame = data_frame.drop(attr, 1)
    attributes = data_frame.axes[1]
    return data_frame, attributes


def read_data_path(filepath, attributes_to_drop=None):
    data_frame = pd.read_csv(filepath)
    if attributes_to_drop is not None:
        for attr in attributes_to_drop:
            data_frame = data_frame.drop(attr, 1)
    attributes = data_frame.axes[1]
    return data_frame, attributes



def in_range(val, str_range):
    the_range = str_range.split('_')
    try:
        val_range = val.split('_')
        return in_range(int(val_range[0]), str_range) and in_range(int(val_range[1]), str_range)
    except AttributeError:
        if len(the_range) > 1:
            return int(the_range[0]) <= val <= int(the_range[1])
        else:
            return int(val) == int(the_range[0])


def in_range_cat(val, node):
    try:
        node.cover[val]
        return True
    except KeyError:
        return False


def in_all_range(pos_vals, rec_vals, attributes, attr_hierarchy):
    for i in range(len(pos_vals)):
        node = attr_hierarchy[attributes[i]].nodes[pos_vals[i]]
        if not in_range_cat(str(rec_vals[i]), node):
            return False
    return True


def find_range(val, level, attr_hiearchy):
    for gen_node in attr_hiearchy.structure[level]:
        if in_range_cat(str(val), gen_node):
            return gen_node.value
    raise RuntimeError('Were not able to find range for ', val)


def normalise(val, min_val, max_val):
    diff = max_val - min_val
    return (val - min_val) / diff


# -------- Semantic distance and marginality presented in (#67)

def semantic_distance_log(node1, node2):
    try:
        ancestor_union = set(node1.ancestors).union(node2.ancestors)
        ancestors_common = set(node1.ancestors).intersection(node2.ancestors)
    except AttributeError:
        n1_ancestors = [node1[0]]
        n2_ancestors = [node2[0]]
        for n1 in node1:
            n1_ancestors += n1.ancestors
        for n2 in node2:
            n2_ancestors += n2.ancestors
        ancestor_union = set(n1_ancestors).union(n2_ancestors)
        ancestors_common = set(n1_ancestors).intersection(n2_ancestors)

    dist = np.log2(1 + ((len(ancestor_union) - len(ancestors_common))/len(ancestor_union)))

    return dist


def marginality(attr_val, attr_idx, dataset, taxonomy):
    """ marginality = sum of semantic distance from the specified attr to all other in the dataset"""
    marginality = 0
    distances = {}
    for attr_r in dataset[:, attr_idx]:
        if attr_r != attr_val:
            sem_dist = semantic_distance_log(taxonomy.nodes[attr_val], taxonomy.nodes[attr_r])
            distances[attr_r] = sem_dist
            marginality += sem_dist
    return marginality


def get_marginalities(dataset, taxonomy, tax_id):
    marginalities = {}
    for attr_val in taxonomy.nodes:
        if taxonomy.nodes[attr_val][0].is_leaf:
            marginalities[attr_val] = marginality(attr_val, tax_id, dataset, taxonomy)

    return marginalities


def calc_mean(marginalities):

    mean_idx = int(np.argmin(list(marginalities.values())))
    mean_val = list(marginalities.keys())[mean_idx]

    return mean_val

def calc_variance_marginality(dataset, marginality, attr_idx):
    marg_sum = 0
    for record in dataset:
        marg_sum += marginality[record[attr_idx]]

    variance = marg_sum/len(dataset)
    return variance


def find_boundary(marginalities, taxonomy):

    mean_val = calc_mean(marginalities)
    dist_to_mean = []
    for attr_val in marginalities.keys():
        dist_to_mean.append(semantic_distance_log(taxonomy.nodes[mean_val], taxonomy.nodes[attr_val]))

    max_idx = int(np.argmax(dist_to_mean))
    bound1 = list(marginalities.keys())[max_idx]

    dist_to_bound = []
    for attr_val in marginalities.keys():
        dist_to_bound.append(semantic_distance_log(taxonomy.nodes[bound1], taxonomy.nodes[attr_val]))

    bound2_idx = int(np.argmax(dist_to_bound))

    bound2 = list(marginalities.keys())[bound2_idx]

    return mean_val, bound1, bound2


def translate_marginality(val, marginalities):
    margin_vals = np.array(list(marginalities.values()))
    attr_vals = list(marginalities.keys())

    min_idx = np.abs(margin_vals - val).argmin()

    attr_val = attr_vals[min_idx]

    return attr_val


def add_category_noise(data, taxonomy, normalise=False, noise_scale=1):
    marginalities = get_marginalities(data, taxonomy, 0)
    if normalise:
        norm_marginalities = {}
        min_val = min(marginalities.values())
        range_val = max(marginalities.values()) - min_val
        for attr in marginalities:
            norm_marginalities[attr] = (marginalities[attr] - min_val)/range_val
        marginalities = norm_marginalities

    mean, b1, b2 = find_boundary(marginalities, taxonomy)
    max_diff = abs(marginalities[b1] - marginalities[b2])
    marginality_data = [marginalities[a[0]] for a in data]
    noise = np.random.laplace(0, max_diff*noise_scale, size=len(marginality_data))
    noisy_data = [marginality_data[i] + noise[i] for i in range(len(noise))]
    new_data = [translate_marginality(v, marginalities) for v in noisy_data]
    return new_data


def s_distance(record1, record2, dataset, marginalities):
    dist = 0
    for i in range(len(record1)):
        loc_var = calc_variance_marginality(np.array([record1, record2]), marginalities, i)
        tot_var = calc_variance_marginality(dataset, marginalities, i)
        dist += loc_var/tot_var

    return dist/len(record1)


# -------- Semantic distance and noise addition from

def find_least_common_subsumer(node1, node2):

    ancestors1 = [node1[0]]
    ancestors2 = [node2[0]]
    for subnode in node1:
        ancestors1 += subnode.ancestors
    for subnode in node2:
        ancestors2 += subnode.ancestors

    common_ancestors = set(ancestors1).intersection(set(ancestors2))
    return max(common_ancestors)


def calc_mean_sim(marginalities):

    mean_idx = int(np.argmin(list(marginalities.values())))
    mean_val = list(marginalities.keys())[mean_idx]

    return mean_val


def semantic_distance_similarity(node1, node2):
    lcs = find_least_common_subsumer(node1, node2)

    # TODO: This does not work if a node is on different heights in different paths
    similarity = (2*(lcs.height+1))/((node1[0].height+1) + (node2[0].height+1))

    return 1 - similarity


def marginality_sim(attr_val, attr_idx, dataset, taxonomy):
    """ marginality = sum of semantic distance from the specified attr to all other in the dataset"""
    marginality = 0
    distances = {}
    for attr_r in dataset[:, attr_idx]:
        if attr_r != attr_val:
            sem_dist = semantic_distance_similarity(taxonomy.nodes[attr_val], taxonomy.nodes[attr_r])
            distances[attr_r] = sem_dist
            marginality += sem_dist
    return marginality


def get_marginalities_sim(dataset, taxonomy, tax_id):
    marginalities = {}
    for attr_val in taxonomy.nodes:
        if taxonomy.nodes[attr_val][0].is_leaf:
            marginalities[attr_val] = marginality_sim(attr_val, tax_id, dataset, taxonomy)

    return marginalities


def get_marginalities_sim_all(dataset, taxonomy, attr_idx):
    marginalities = {}
    for attr_val in taxonomy.nodes:
        marginalities[attr_val] = marginality_sim(attr_val, attr_idx, dataset, taxonomy)
    return marginalities


def calc_variance_dist(dataset, taxonomy, attr_idx, mean_node):
    sq_sum = 0
    for record in dataset:
        sq_sum += (semantic_distance_similarity(taxonomy.nodes[record[attr_idx]], mean_node)) ** 2
    return sq_sum / len(dataset)


def add_semantic_noise(dataset, taxonomy, attr_idx, noise_level):

    def find_possible_nodes(condition, comp_node, other_nodes):
        # Find all nodes that pass the condition when the sem dist(node, record val) >= abs(e)
        # If no such node exists, find the node with the largest value
        possible_nodes = []
        for node in other_nodes:
            dist = semantic_distance_similarity(node, comp_node)
            if condition(dist):
                possible_nodes.append(node)
        return possible_nodes

    def find_most_distance_node(comp_node, other_nodes):
        distances = [semantic_distance_similarity(comp_node, node) for node in other_nodes]
        max_dist = 0
        max_nodes = []

        for i, dist in enumerate(distances):
            if dist == max_dist:
                max_nodes.append(i)
            elif dist > max_dist:
                max_nodes = [i]
                max_dist = dist

        max_idx = np.random.choice(max_nodes)
        return other_nodes[max_idx]

    def find_closest_node(comp_node, other_nodes):
        distances = [semantic_distance_similarity(comp_node, node) for node in other_nodes]
        min_dist = 2
        min_nodes = []

        for i, dist in enumerate(distances):
            if dist == min_dist:
                min_nodes.append(i)
            elif dist < min_dist:
                min_nodes = [i]
                min_dist = dist
        min_idx = np.random.choice(min_nodes)

        return other_nodes[min_idx]

    marginalities = get_marginalities_sim_all(dataset, taxonomy, attr_idx)
    mean_val = calc_mean(marginalities)
    variance = calc_variance_dist(dataset, taxonomy, attr_idx, taxonomy.nodes[mean_val])
    noise = np.random.laplace(0, variance*noise_level, len(dataset))

    noise_data = dataset.copy()

    nodes = [taxonomy.nodes[attr] for attr in marginalities if taxonomy.nodes[attr][0].is_leaf]

    count = 0
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0

    for i in range(len(dataset)):
        e = noise[i]
        current_node = taxonomy.nodes[dataset[i, attr_idx]]
        possible_nodes = find_possible_nodes(lambda x: x >= abs(e), current_node, nodes)
        mean_dist = semantic_distance_similarity(taxonomy.nodes[mean_val], current_node)

        if len(possible_nodes) > 0:
            if e < 0.01:  # Maybe add something like e<0.01 so that no noise is added for very small e
                continue
            elif dataset[i, attr_idx] == mean_val:
                min_node = find_closest_node(taxonomy.nodes[dataset[i, attr_idx]], possible_nodes)
                noise_data[i, attr_idx] = min_node[0].value
                if min_node[0].value == "Lung cancer":
                    count1 += 1
            elif e > 0:
                # The node with the smallest distance to the record value, when the sem dist >= abs(e) AND
                # the distance between the mean and the node > the distance between the record val and the mean
                pos_nodes = find_possible_nodes(lambda x: x > mean_dist, taxonomy.nodes[mean_val], possible_nodes)
                if len(pos_nodes) > 0:
                    min_node = find_closest_node(current_node, pos_nodes)
                    noise_data[i, attr_idx] = min_node[0].value
                    if min_node[0].value == "Lung cancer":
                        count2 += 1
                else:
                    # Find the node that is closest to the mean distance
                    approx_node = find_most_distance_node(current_node, possible_nodes)
                    noise_data[i, attr_idx] = approx_node[0].value
                    if approx_node[0].value == "Lung cancer":
                        count3 += 1

            elif e < 0:
                # The node with the smallest distance to the record value, when the sem dist >= abs(e) AND
                # the distance between the mean and the node < the distance between the record val and the mean
                pos_nodes = find_possible_nodes(lambda x: x < mean_dist, taxonomy.nodes[mean_val], possible_nodes)

                if len(pos_nodes) > 0:
                    min_node = find_closest_node(current_node, pos_nodes)
                    noise_data[i, attr_idx] = min_node[0].value
                    if min_node[0].value == "Lung cancer":
                        count4 += 1
                else:
                    approx_node = find_closest_node(current_node, possible_nodes)
                    noise_data[i, attr_idx] = approx_node[0].value
                    if approx_node[0].value == "Lung cancer":
                        count5 += 1
        else:
            approx_node = find_most_distance_node(taxonomy.nodes[dataset[i, attr_idx]], nodes)
            noise_data[i, attr_idx] = approx_node[0].value
            if approx_node[0].value == "Lung cancer":
                count += 1
            # TODO: Add some randomness to the greatest node, so that they don't always map the the same.

    return noise_data
