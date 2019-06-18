import numpy as np
from anonymisation.Taxonomy import TaxNode

def add_semantic_distances(taxonomies):
    for taxonomy in taxonomies:
        if not taxonomy.is_numeric():
            for node1 in taxonomy.get_domain():
                for node2 in taxonomy.get_domain():
                    for node in taxonomy.nodes[node1]:
                        node.add_distance(node2, semantic_distance(taxonomy.nodes[node1], taxonomy.nodes[node2]))


def add_semantic_distances_all(taxonomies):
    for taxonomy in taxonomies:
        if not taxonomy.is_numeric():
            for node1 in taxonomy.nodes:
                for node2 in taxonomy.nodes:
                    for node in taxonomy.nodes[node1]:
                        node.add_distance(node2, semantic_distance(taxonomy.nodes[node1], taxonomy.nodes[node2]))



def semantic_distance(node1, node2):
    """
    The semantic distance between two concepts in a taxonomy, presented in paper #67 TODO: add proper reference
    :param node1: Taxonomy node 1
    :param node2: Taxonomy node 2
    :return: The semantic distance between the nodes
    TODO: Could speed this up by saving the relationships between nodes
    """
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


def semantic_distance_quick(node1, node2):
    try:
        return node1[0].get_distance(node2[0].value)
    except KeyError:
        return semantic_distance(node1, node2)

    #if not node1[0].is_leaf or not node2[0].is_leaf:
    #    return semantic_distance(node1, node2)
    #return node1[0].get_distance(node2[0].value)


def numeric_distance(node1, node2):
    """
    Computes the difference between two nodes of numerical type.
    :param node1: Taxonomy node 1
    :param node2: Taxonomy node 2
    :return: The difference
    """
    if node1[0].is_leaf and node2[0].is_leaf:
        try: # Specific for body_temperature
            return float(node1[0].value) - float(node2[0].value)
        except ValueError:
            try:
                return float(node1[0].value) - float(node2[0].value)
            except ValueError:
                # One of the nodes is 35.0-42.0
                if node1[0].value == '35.0-42.0' and node2[0].value == '35.0-42.0':
                    return 0
                try:
                    float(node1[0].value)
                    boundary = node2[0].value.split('_')
                    return float(boundary[1]) - float(boundary[0])
                except ValueError:
                    boundary = node1[0].value.split('_')
                    return float(boundary[1]) - float(boundary[0])
    elif node1[0].is_leaf and not node2[0].is_leaf:
        boundary = node2[0].value.split('_')
        if len(boundary) > 1:
            return float(boundary[1]) - float(boundary[0])
        return float(node1[0].value) - float(boundary[0])
    elif not node1[0].is_leaf and node2.is_leaf:
        boundary = node1[0].value.split('_')
        if len(boundary) > 1:
            return float(boundary[1]) - float(boundary[0])
        return float(node2[0].value) - float(boundary[0])

    raise ValueError("More than one node is generalised")


def node_distance(node1, node2, boundary_dist=1, quick=True):
    """
    Computes the distance between two nodes, using semantic distance for categories and numeric distance for numeric
    values
    :param node1: Taxonomy node 1
    :param node2: Taxonomy node 2
    :param quick: Decides whether the quick version should be computed or not
    :return: The distance between the two nodes
    TODO: return absolute values? Since it is a distance and not difference..
    """
    if node1[0].value == '*' or node2[0].value == '*':
        return 1
    if node1[0].is_numeric and node2[0].is_numeric:
        return numeric_distance(node1, node2)/boundary_dist
    elif not node1[0].is_numeric and not node2[0].is_numeric:
        if quick:
            return semantic_distance_quick(node1, node2)/boundary_dist
        else:
            return semantic_distance(node1, node2)/boundary_dist
    else:
        raise TypeError("The nodes are not of the same type")


def marginality(attr_val, taxonomy, samples=None):
    """ marginality = sum of semantic distance from the specified attr to all other in the dataset"""
    marginality_val = 0
    if samples is not None:
        domain = samples
    else:
        domain = taxonomy.get_domain()

    unique, counts = np.unique(domain, return_counts=True)

    for n, attr_l in enumerate(unique):
        if attr_l != attr_val:
            sem_dist = semantic_distance_quick(taxonomy.nodes[attr_val], taxonomy.nodes[attr_l])
            marginality_val += (sem_dist * counts[n])
    return marginality_val


def compute_boundary(taxonomy):
    """
    Find the smallest and largest value for the domain. For categories the boundary is defined as the node with the
    largest marginality and the node with the largest distance to that node
    :param taxonomy: The taxonomy
    :return: (tuple) (min, max)
    """
    domain = taxonomy.get_domain()
    if taxonomy.is_numeric():
        try:
            num_domain = [int(n) for n in domain]
        except ValueError:
            # This is specifically for the body_temperature case
            num_domain = []
            for n in domain:
                try:
                    num = float(n)
                    num_domain.append(num)
                except ValueError:
                    continue
        min_val = str(min(num_domain))
        max_val = str(max(num_domain))
    else:
        marginality_values = [marginality(attr, taxonomy) for attr in taxonomy.get_domain()]
        max_val = domain[np.argmax(marginality_values)]
        min_val = domain[np.argmax([semantic_distance_quick(taxonomy.nodes[attr], taxonomy.nodes[max_val])
                                    for attr in domain])]
    return min_val, max_val


def distance(record1, record2, taxonomies):
    """
    Based on distance with numerical and nominal attributes from SC 4.6
    :param record1: A record from the dataset
    :param record2: Another record from the dataset
    :param taxonomies: Taxonomy
    :return: The distance
    """
    boundaries = [taxonomy.boundary for taxonomy in taxonomies]
    tot_dist = 0
    for i in range(len(taxonomies)):
        try:
            node1 = taxonomies[i].nodes[str(record1[i]).strip('\n')]
            node2 = taxonomies[i].nodes[str(record2[i]).strip('\n')]
        except KeyError:
            node1 = [TaxNode(record1[i], None, True, True)]
            node2 = [TaxNode(record2[i], None, True, True)]
        nodeb = taxonomies[i].nodes[boundaries[i][0]]
        nodet = taxonomies[i].nodes[boundaries[i][1]]

        if record1[i] == '*':
            tot_dist += 1
        elif record2[i] == '*':
            tot_dist += 1
        else:
            tot_dist += ((node_distance(node1, node2))**2)/((node_distance(nodeb, nodet))**2)

    return tot_dist**(1/2)

