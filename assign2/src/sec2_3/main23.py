import numpy as np
import pickle
import ex_2_3
import ex_2_3_tree_helper as tree_helper
import pandas as pd
import csv

class Tree(tree_helper.Tree):
    """
    Inherit from the tree helper
    """
    st_evids = dict()  # subtree evidences dictionary

    def load_sample(self, sample):
        # clean out the saved s(uv, i) results
        self.st_evids = dict()
        # use the provided function
        ex_2_3.load_sample(self.root, sample)

    def s_fun(self, node, value):
        """
        The s(u, v) in mentioned in the problem
        Args:
            node (Node): the node
            value (number): the value of this node
        """
        if self.is_leaf(node):
            if node.sample == value:
                return 1
            else:
                return 0

        # for each child of this node, consider all its possible values
        ret = 1
        for c in node.descendants:
            key = "{},{},{}".format(node.name, c.name, value) # (u,v,i)

            if key in self.st_evids:
                st_evid = self.st_evids[key]
            else:
                st_evid = 0    # the sum of the blanket
                # calculate the result recursively
                for i, w in enumerate(c.cat[value][:]):
                    st_evid += w * self.s_fun(node=c, value=i)
                # save the node_evidence
                self.st_evids[key] = st_evid

            ret *= st_evid
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
        self.st_evids = dict()
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

    result_dict = dict()
    for param_k, param_v in params.items():
        t = Tree()
        t.load_params(param_v)
        # for each tree, create a dictionary to save the result
        result_dict[param_k] = dict()
        for s in range(1, 4):

            sample_name = "{}_sample_{}".format(param_k, s)
            t.load_sample(samples[sample_name])
            p_obs_nodes = t.get_obs_prob()
            print(sample_name)
            print(p_obs_nodes)
            result_dict[param_k]["Sample{}".format(s)] = p_obs_nodes

    # transform the result to csv
    df = pd.DataFrame.from_dict(result_dict, orient="index")
    df.index.name = "Tree"
    # reorder the columns
    df = df[["Sample{}".format(i) for i in range(1, 4)]]
    # rename index
    idx_map = dict()
    for idx in df.index:
        tree_name = idx.split("_alpha_")[0]  # drop alpha values
        tree_name = tree_name.replace("_", " ")
        idx_map[idx] = tree_name

    df = df.rename(index=idx_map)

    df.to_csv("result_sec2_3.csv", float_format="%.3e", quoting=csv.QUOTE_NONE)
