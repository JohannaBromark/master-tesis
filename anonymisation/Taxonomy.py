
# TODO: Add method for getting a node instead of getting the list of possibly several nodes, when that is not needed

class TaxNode(object):
    """Assumes each node has at most one parent"""
    def __init__(self, value, parent, is_numeric=False, is_leaf=False):
        self.value = value
        self.parent = parent
        self.ancestors = []
        self.height = 0
        self.width = 0
        self.covers = {value: self}
        self.is_leaf = is_leaf
        self.is_numeric = is_numeric
        self.distances = {}

        if parent is not None:
            parent.covers[self.value] = self
            #self.ancestors.append(self)
            self.ancestors.append(parent)
            self.ancestors += parent.ancestors[:]
            self.height = parent.height + 1
            for ancestor in self.ancestors:
                ancestor.covers[self.value] = self
                if self.is_leaf:
                    ancestor.width += 1

    def __lt__(self, other):
        return self.height < other.height

    def __gt__(self, other):
        return self.height > other.height

    def add_distance(self, node_val, distance):
        self.distances[node_val] = distance

    def get_distance(self, node_val):
        return self.distances[node_val]


class Taxonomy(object):
    def __init__(self, tax_nodes, height):
        self.nodes = tax_nodes
        self.height = height
        self.level_structure = self.create_level_structure()
        self.width = self.nodes['*'][0].width
        self.boundary = None

    def create_level_structure(self):
        levels = []
        for i in range(self.height):
            levels.append([])
        for node in list(self.nodes.values()):
            try:
                levels[node.height].append(node)
            except AttributeError:
                for n in node:
                    levels[n.height].append(node)
        return levels

    def get_node(self, value):
        """
        Return a representative of a node for that value. It does not matter what ancestors it has.
        :param value: The value of the node
        :return: The first node with that value
        """
        return self.nodes[value][0]

    def get_domain(self):
        return [node[0].value for node in self.level_structure[-1]]

    def is_numeric(self):
        #try:
        #    return self.nodes['*'][0].is_numeric
        #except KeyError: # TODO remove this case, only for testing
        #    return True
        return self.nodes['*'][0].is_numeric

    def add_boundary(self, boundary):
        self.boundary = boundary
