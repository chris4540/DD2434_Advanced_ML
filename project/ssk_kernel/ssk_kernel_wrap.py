import os
import json
import ssk_kernel.ssk_kernel_c as ssk_kernel_c

class SSKKernel:

    def __init__(self, k, decay):
        self.k = k
        self.decay = decay

        self.kernel_entry_json = "./ssk_kernel_data/ssk_k{}_lbda{}.json".format(k, decay)

        try:
            with open(self.kernel_entry_json, "r") as f:
                self.kernel_entry = json.load(f)
        except:
            self.kernel_entry = dict()

    def __call__(self, doc1, doc2):
        key = "{}:{}".format(doc1, doc2)
        key_rev = "{}:{}".format(doc2, doc1)

        if key in self.kernel_entry:
            return self.kernel_entry[key]

        if key_rev in self.kernel_entry:
            return self.kernel_entry[key]
        # =================================
        # cannot find entry in my own data
        ret = ssk_kernel_c.ssk_kernel(doc1, doc2, self.k, self.decay)

        # save the entry
        self.kernel_entry[key] = ret
        self.kernel_entry[key_rev] = ret

        return ret

    def save_kernel_entry(self):
        with open(self.kernel_entry_json, "w") as f:
            json.dump(self.kernel_entry, f)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.save_kernel_entry()

if __name__ == '__main__':
    with SSKKernel(2, 0.5) as k:
        ret = k("car", "cat")
        print(ret)
    # k = SSKKernel(2, 0.5)
    # ret = k("car", "cat")
    # k.save_kernel_entry()
    # print(ret)
