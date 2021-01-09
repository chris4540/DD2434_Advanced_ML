import json

if __name__ == '__main__':
    k = 5
    decay = 0.8
    kernel_entry_json = "./ssk_kernel_data/ssk_k{}_lbda{}.json".format(k, decay)

    ret = dict()
    with open(kernel_entry_json, "r") as f:
        kernel_entry = json.load(f)

    with open("./ssk_kernel_data/data2idx.json", "r") as f:
        data2idx = json.load(f)

    cnt = 0
    for k, v in kernel_entry.items():
        doc1, doc2 = k.split(":")

        if doc1 in data2idx:
            d1_idx = data2idx[doc1]
        else:
            d1_idx = len(data2idx)
            data2idx[doc1] = d1_idx

        if doc2 in data2idx:
            d2_idx = data2idx[doc2]
        else:
            d2_idx = len(data2idx)
            data2idx[doc2] = d2_idx

        k_new = "{}:{}".format(d1_idx, d2_idx)

        ret[k_new] = v
        cnt += 1
        print("Processed {} data".format(cnt))

    with open(kernel_entry_json, "w") as f:
        json.dump(ret, f, indent=1)

    with open("./ssk_kernel_data/data2idx.json", "w") as f:
        json.dump(data2idx, f, indent=1)
