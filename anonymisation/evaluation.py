from anonymisation.distance_computations import node_distance, distance


# Evaluation computations when using the Taxonomy class to represent the taxonomies


def SSE(raw, anon, taxonomies):
    """
    Compute sum of squared errors (SSE) on an attribute level, as in soria-comas (2014)
    :param raw: The original dataset
    :param anon: The anonymised dataset
    :param taxonomies: List with the taxonomy for each attribute
    :return: The sum of squared error
    """
    sse = 0
    for r_o, r_a in zip(raw.values, anon.values):
        diff = distance(r_o, r_a, taxonomies)
        sse += diff ** 2
    return sse


def granularity(dataset, taxonomies, k):
    """
    Presented in Safepub 5.1. Measures the extent to which the values in the dataset covers the domains of respective
    attribute.
    :param dataset: (DataFrame) The dataset to compute the metric for.
    :param taxonomies: Definition for the taxonomies
    :param k: The k that has been used for anonymisation.
    :return: The value for the granularity, scaled by the sensitivity
    """
    partitions = dataset.groupby(list(dataset.columns)).groups
    suppressed_record = tuple(['*']*len(taxonomies))
    gran = 0
    domains = []
    for n in range(len(taxonomies)):
        domains.append(get_domain_size(taxonomies[n]))
    num_outliers = 0

    for record_val in partitions.keys():
        for i in range(len(taxonomies)):
            if record_val == suppressed_record:
                num_outliers += len(partitions[record_val])
            else:
                if record_val[i] == '*':
                    num_outliers += len(partitions[record_val])
                else:
                    leaves = num_leaves(record_val[i], taxonomies[i])
                    gran += ((leaves / domains[i]) * len(partitions[record_val]))

    gran += num_outliers
    if k > 1:
        gran /= (len(taxonomies) * (k-1))

    return gran


def granularity_score(dataset, taxonomies, k):
    """
    Multiplies with -1, so that larger values are better.
    """
    return - granularity(dataset, taxonomies, k)


def intensity(dataset, taxonomies, k):
    """
     Presented in Safepub 5.1. Sums the relative generalisation level of values in all cells.
    :param dataset: (DataFrame) The dataset to compute the metric for.
    :param taxonomies: Definition for the taxonomies
    :param k: The k that has been used for anonymisation.
    :return: The intensity, scaled by the sensitivity
    """
    partitions = dataset.groupby(list(dataset.columns)).groups
    suppressed_record = tuple(['*']*len(taxonomies))
    try:
        num_suppressed = len(partitions[suppressed_record])
    except KeyError:
        num_suppressed = 0
    num_non_suppressed = len(dataset) - num_suppressed

    heights = [tax.height-1 for tax in taxonomies]

    random_record = None
    for partition_val in partitions:
        if partition_val != suppressed_record:
            random_record = partition_val
            break
    if random_record is not None:
        generalisation_levels = []
        for i in range(len(taxonomies)):
            level = taxonomies[i].get_node(str(random_record[i])).height
            generalisation_levels.append(abs(level-heights[i])%(heights[i]+1))
    else:
        generalisation_levels = heights

    intens = 0
    for i in range(len(taxonomies)):
        gen = generalisation_levels[i]/heights[i]
        intens += (gen * num_non_suppressed) + num_suppressed

    if k > 1:
        intens /= ((k-1)*len(taxonomies))

    return intens


def intensity_score(dataset, taxonomies, k):
    """
    Multiplies with -1, so that larger values are better.
    """
    return -intensity(dataset, taxonomies, k)


def discernability(dataset, k):
    """
    Presented in Safepub 5.2. Penetalises records depending on the size of the equivalence class they belong to.
    :param dataset: (DataFrame) The dataset to compute the metric for.
    :param k: The k that has been used for anonymisation.
    :return: The value for the discernability, scaled by the sensitivity
    """

    grouper = list(dataset.columns)
    suppressed_val = tuple(['*'] * len(dataset.columns))

    partitions = dataset.groupby(grouper).groups
    tot_records = len(dataset)

    try:
        num_suppressed = len(partitions[suppressed_val])
        del partitions[suppressed_val]
    except KeyError:
        num_suppressed = 0
    partition_values = list(partitions.values())
    partition_sizes = [len(partition) for partition in partition_values]
    penalty_non_suppressed = 0
    for part_size in partition_sizes:
        penalty_non_suppressed += part_size ** 2

    penalty_suppressed = num_suppressed * tot_records

    tot_penalty = penalty_non_suppressed + penalty_suppressed
    if k == 1:
        sensitivity = 5
    else:
        sensitivity = (((k ** 2) / (k - 1)) + 1)

    disc = tot_penalty/(sensitivity*tot_records)

    return disc


def discernability_score(dataset, k):
    """
    Multiplies with -1, so that larger values are better.
    """
    return -discernability(dataset, k)


def non_uniform_entropy(dataset, k):
    """
    Presented in Safepub 5.3. Quantifies the amount of information that can be obtained about the input dataset by
    observing the output dataset. Information loss increase with increasing homogeneity.
    :param dataset: (DataFrame) The dataset to compute the metric for.
    :param k: The k that has been used for anonymisation.
    :return: The value for the entropy, scaled by the sensitivity
    """
    suppressed_penalty = 0
    non_suppressed_penalty = 0
    attributes = dataset.columns
    for i in range(len(attributes)):
        partitions = dataset.groupby(attributes[i]).groups
        for att_val in partitions:
            if att_val == '*':
                suppressed_penalty += (len(partitions[att_val]) * len(dataset))
            else:
                non_suppressed_penalty += (len(partitions[att_val]) * len(partitions[att_val]))
    tot_penalty = suppressed_penalty + non_suppressed_penalty

    if k == 1:
        sensitivity = len(dataset.columns)
    else:
        sensitivity = (((k ** 2) / (k - 1)) + 1) * len(dataset.columns)

    score = tot_penalty/(len(dataset) * sensitivity)

    return score


def non_uniform_entropy_score(dataset, k):
    """
    Multiplies with -1, so that larger values are better.
    """
    return -non_uniform_entropy(dataset, k)


def group_size(dataset):
    """
    Presented in Safepub 5.4. Measures the average size of the equivalence classes.
    :param dataset: (DataFrame) The dataset to compute the metric for.
    :return: The value for the group size.
    """
    return len(dataset.groupby(list(dataset.columns)).groups)


# -- Granularity help functions
def num_leaves(r_i, taxonomy):
    leaves = taxonomy.nodes[str(r_i)][0].width
    if leaves == 0:
        return 1
    else:
        return leaves


def get_domain_size(taxonomy):
    return taxonomy.width
