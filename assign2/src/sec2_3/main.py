import numpy as np
import pickle
import ex_2_3
import ex_2_3_tree_helper as tree_helper

class Tree(tree_helper.Tree):
    """
    Inherit from the tree helper
    TODO: class documentation
    """


    def load_sample(self, sample):
        self.s_values = dict()
        ex_2_3.load_sample(self.root, sample)

    def s_fun(self, node, value):
        """
        TODO: rename this function
        """
        # print("s_fun({}, {})".format(node.name, value))
        if self.is_leaf(node):
            if node.sample == value:
                return 1
            else:
                return 0

        # for each child of this node, consider all its possible values
        ret = 1
        for c in node.descendants:
            key = "({},{})".format(c.name, value)
            if key in self.s_values:
                node_evidence = self.s_values[key]
            else:
                node_evidence = 0    # the sum of the blanket
                for i, w in enumerate(c.cat[value][:]):
                    node_evidence += w * self.s_fun(node=c, value=i)
                # save the node_evidence
                self.s_values[key] = node_evidence

            ret *= node_evidence
        return ret

    @staticmethod
    def is_leaf(node):
        """
        Test if the node is a leaf node

        Return:
            return true if the input node is a leaf node, vice verse.
        """
        if not node.descendants:
            return True

        return False

    def get_obs_prob(self):
        """
        Get the probability of the obserations (samples) set by
        the instance method: load_sample

        Return:
            return the probability of the set observations (samples)
        """
        self.s_values = dict()
        # consider all posible value of the root
        # prob: the probability of the root has that i-th catagory
        #    i: the label of a catagory
        ret = 0
        for i, prob in enumerate(self.root.cat[0]):
            ret += prob * self.s_fun(self.root, i)

        return ret


if __name__ == '__main__':

    # Load parameters and samples
    params = np.load("./tree_params.npy").tolist()
    samples = np.load("./tree_samples.npy").tolist()

    for param_k, param_v in params.items():
        t = Tree()
        t.load_params(param_v)
        for s in range(1, 4):
            sample_name = "{}_sample_{}".format(param_k, s)
            t.load_sample(samples[sample_name])
            print(sample_name)
            print(t.get_obs_prob())

