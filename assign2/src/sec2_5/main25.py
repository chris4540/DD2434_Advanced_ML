import numpy as np
import pickle
import ex_2_3
import ex_2_3_tree_helper as tree_helper

class Tree(tree_helper.Tree):
    """
    Inherit from the tree helper
    """

    s_fun = None
    st_evids = None


    def load_sample(self, sample):
        self.st_evids = dict()
        self.s_fun = dict()
        ex_2_3.load_sample(self.root, sample)

    def get_s_fun_value(self, node, value):
        if (node.name, value) in self.s_fun:
            return self.s_fun[(node.name, value)]

        if self.is_leaf(node):
            if node.sample == value:
                ret = 1
            else:
                ret = 0
            # store s_fun value
            self.s_fun[(node.name, value)] = ret

            return ret

        # for each child of this node, consider all its possible values
        ret = 1
        for c in node.descendants:
            key = "{},{},{}".format(node.name, c.name, value) # for saving s(uv,i)

            if key in self.st_evids:
                st_evid = self.st_evids[key]
            else:
                st_evid = 0    # the sum of the blanket
                for i, w in enumerate(c.cat[value][:]):
                    st_evid += w * self.get_s_fun_value(node=c, value=i)
                # save the node_evidence
                self.st_evids[key] = st_evid

            ret *= st_evid

        # store s_fun value
        self.s_fun[(node.name, value)] = ret
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
        # consider all posible value of the root
        # prob: the probability of the root has that i-th catagory
        #    i: the label of a catagory
        ret = 0
        for i, prob in enumerate(self.root.cat[0]):
            ret += prob * self.get_s_fun_value(self.root, i)

        return ret

    def _sample(self, node):
        """
        sample from a catagorial distribution given that the parant has a sampled
        value (except root)
        """
        if self.is_leaf(node):
            return

        if node == self.root:
            theta = self.root.cat[0]
        else:
            parent_val = node.ancestor.sample
            theta = node.cat[parent_val]

        post_theta = np.zeros(len(theta))  # posterior catagorial distribution
        for i in range(len(theta)):
            post_theta[i] = theta[i] * self.s_fun[(node.name, i)]

        post_theta = self.normalize_p_vec(post_theta)

        sample = self.get_sample_from_cat(post_theta)
        # write sample to this node
        node.sample = sample

        print("{},{},{}".format(node.name, sample, post_theta[sample]))

        for c in node.descendants:
            self._sample(c)

    def perform_sampling(self):
        self._sample(self.root)

    @staticmethod
    def get_sample_from_cat(theta):
        """
        Sample from categorical distribution
        Args:
            theta (np.array/list): categorical parameter for each category
        """
        return np.random.choice([i for i in range(len(theta))], p=theta)

    @staticmethod
    def normalize_p_vec(p_vec):
        """
        normalize a probability vector to sum to 1
        """
        return p_vec / np.sum(p_vec)


def load_pickle(fname):
    with open(fname, 'rb') as f:
        ret = pickle.load(f)
    return ret

if __name__ == '__main__':

    cpd = load_pickle("tree_with_CPD.pkl")
    leaves = load_pickle("tree_with_leaf_samples.pkl")

    t = Tree()
    t.load_params(cpd)
    t.load_sample(leaves)
    t.print_tree(True)

    # throught calculating the prob of observation, get all s(u,i) values
    p = t.get_obs_prob()
    t.perform_sampling()

    t.print_tree(True)

