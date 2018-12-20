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

    def pre_order(self, node=None):

        if node is None:
            node = self.root

        # visiting
        print("--------------------------")
        print("Node Name:", node.name)
        print("Node Cat:", node.cat)
        print("Node Sample:", node.sample)
        print("Is leaf:", self.is_leaf(node))
        print("--------------------------")

        for c in node.descendants:
            self.pre_order(c)

    def s_fun(self, node, value):
        """
        TODO: rename this function
        """
        if self.is_leaf(node):
            if node.sample == value:
                return 1
            else:
                return 0

        # for each child of this node, consider all its possible values
        ret = 1
        for c in node.descendants:
            tmp = 0
            for i, w in enumerate(c.cat[value][:]):
                # print(w)
                tmp += w * self.s_fun(node=c, value=i)

            ret *= tmp
            # print("=====================================")

        return ret

    @staticmethod
    def is_leaf(node):
        if not node.descendants:
            return True

        return False

    def get_obs_prob(self):
        # consider all posible value of the root

        # prob: the probability of the root has that i-th catagory
        #    i: the label of a catagory

        ret = 0
        for i, prob in enumerate(self.root.cat[0]):
            ret += prob * self.s_fun(self.root, i)

        return ret




if __name__ == '__main__':

    # build a tree from parameters
    t = Tree()
    params = np.load("./tree_params.npy").tolist()
    key = params.keys()[0]
    #   Load params into tree
    t.load_params(params[key])
    # t.print_tree()

    # ========================================================
    # set a sample to a tree
    samples = np.load("./tree_samples.npy").tolist()
    sample = samples[key + '_sample_1']
    t.load_sample(sample)

    t.print_tree(print_sample=True)
    t.pre_order()

    print(t.s_fun(t.root, 0))
    # print(t.get_obs_prob())


