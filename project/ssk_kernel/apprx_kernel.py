"""
A wrapper of the exact implementation.
Provide the support of storing data and saving entries to json file
"""
import os
import json
import ssk_impl
import ssk_impl.ssk_kernel_approx_c as kernel
import numpy as np

class SSKKernelApprox:

    def __init__(self, k, decay, save_entry=False):
        self.k = k
        self.decay = decay
        self.save_entry = save_entry

        self.kernel_entry_json = "./ssk_kernel_data/ssk_aprx_k{}_lbda{}.json".format(k, decay)

        try:
            with open(self.kernel_entry_json, "r") as f:
                self.kernel_entry = json.load(f)
        except:
            self.kernel_entry = dict()

        basis_json = "./ssk_kernel_data/most_freq_contss_k{}.json".format(k)
        with open(basis_json, "r") as f:
            self.basis = json.load(f)

        self.d2idx_file = "./ssk_kernel_data/data2idx.json"
        with open(self.d2idx_file, "r") as f:
            self.data2idx = json.load(f)


    def __call__(self, doc1, doc2):

        d1_idx = 0
        d2_idx = 0

        if doc1 in self.data2idx:
            d1_idx = self.data2idx[doc1]

        if doc2 in self.data2idx:
            d2_idx = self.data2idx[doc2]

        if d1_idx == 0 or d2_idx == 0:
            # cannot find entry in my own data
            ret = self.ssk_value(doc1, doc2)
        else:
            key = "{}:{}".format(d1_idx, d2_idx)
            if key in self.kernel_entry:
                ret = self.kernel_entry[key]
                return ret
            else:
                ret = self.ssk_value(doc1, doc2)

        # save entry
        if self.save_entry:
            self.set_kernel_entry(doc1, doc2, ret)
        return ret

    def add_doc_to_map(self, doc):
        if doc not in self.data2idx:
            self.data2idx[doc] = len(self.data2idx)

    def save_kernel_to_json(self):
        with open(self.kernel_entry_json, "w") as f:
            json.dump(self.kernel_entry, f, indent=1, sort_keys=True)

        with open(self.d2idx_file, "w") as f:
            json.dump(self.data2idx, f, indent=1, sort_keys=True)

    def set_kernel_entry(self, doc1, doc2, val):

        self.add_doc_to_map(doc1)
        self.add_doc_to_map(doc2)

        d1_idx = self.data2idx[doc1]
        d2_idx = self.data2idx[doc2]
        key = "{}:{}".format(d1_idx, d2_idx)
        key_rev = "{}:{}".format(d2_idx, d1_idx)
        # save the entry
        self.kernel_entry[key] = val
        self.kernel_entry[key_rev] = val


    def ssk_value(self, doc1, doc2):
        st = kernel.ssk_kernel_approx(doc1, doc2, self.k, self.decay, self.basis)
        ss = kernel.ssk_kernel_approx(doc1, doc1, self.k, self.decay, self.basis)
        tt = kernel.ssk_kernel_approx(doc2, doc2, self.k, self.decay, self.basis)

        # normalize the return value
        ret = st / np.sqrt(ss * tt)

        if np.isnan(ret):
            ret = 1e-20
        return ret

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.save_kernel_to_json()
