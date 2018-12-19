import numpy as np
import pickle
import ex_2_3
import ex_2_3_tree_helper as tree_helper

class Tree(tree_helper.Tree):
    """
    Inherit from the tree helper
    """

    def load_sample(self, sample):
        ex_2_3.load_sample(self.root, sample)

    # def print_tree(self, print_sample = False):
    #     """
    #     An example of visit
    #     """
    #     curr_layer = [self.root]
    #     while curr_layer != []:
    #         string = ''
    #         next_layer = []
    #         for elem in curr_layer:
    #             string = string + elem.name  + ' '
    #             if (print_sample and elem.sample != None):
    #                 string = string[:-1] + ':' + str(elem.sample)  + ' '
    #             for child in elem.descendants:
    #                 next_layer.append(child)
    #         print(string)
    #         curr_layer = next_layer

    # # def pre_order(self):
    # #     """
    # #     Pre order traversal with stack(list)
    # #     """
    # #     node_stack = list()

    # #     # add root
    # #     node_stack.append(self.root)
    # #     while node_stack:  # while the stack is not empty
    # #         node = node_stack.pop()

    # #         # visiting
    # #         print("Node Name:", node.name)
    # #         print("Node Cat:", node.cat)
    # #         print("Node Sample:", node.sample)

    # #         # add childrens to the stack
    # #         for c in reversed(node.descendants):
    # #             node_stack.append(c)

    # def pre_order(self):
    #     """
    #     Preorder in recursion way
    #     """

if __name__ == '__main__':

    # build a tree from parameters
    t = Tree()
    params = np.load("./tree_params.npy").tolist()
    key = params.keys()[0]
    #   Load params into tree
    t.load_params(params[key])
    t.print_tree()

    # ========================================================
    # set a sample to a tree
    samples = np.load("./tree_samples.npy").tolist()
    sample = samples[key + '_sample_1']
    t.load_sample(sample)

    t.print_tree(print_sample=True)
    # t.pre_order()

