import ex_2_3_tree_helper as tree_helper
import pickle

class Tree(tree_helper.Tree):
    """
    Inherit from the tree helper
    """
    def visit(self, print_sample = False):
        """
        An example of visit
        """
        curr_layer = [self.root]
        while curr_layer != []:
            string = ''
            next_layer = []
            for elem in curr_layer:
                string = string + elem.name  + ' '
                if (print_sample and elem.sample != None):
                    string = string[:-1] + ':' + str(elem.sample)  + ' '
                for child in elem.descendants:
                    next_layer.append(child)
            print(string)
            curr_layer = next_layer

    # def pre_order(self):
    #     """
    #     Pre order traversal with stack(list)
    #     """
    #     node_stack = list()

    #     # add root
    #     node_stack.append(self.root)
    #     while node_stack:  # while the stack is not empty
    #         node = node_stack.pop()

    #         # visiting
    #         print("Node Name:", node.name)
    #         print("Node Cat:", node.cat)
    #         print("Node Sample:", node.sample)

    #         # add childrens to the stack
    #         for c in reversed(node.descendants):
    #             node_stack.append(c)

    def pre_order(self):
        """
        Preorder in recursion way
        """

if __name__ == '__main__':
    t = Tree()
    my_data_path = ''
    with open(my_data_path + 'tree_params.pickle', 'rb') as handle:
        params = pickle.load(handle)
    key = params.keys()[0]
    #   Load params into tree
    t.load_params(params[key])
    t.print_tree()

    t.pre_order()
