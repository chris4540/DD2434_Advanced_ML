import json

if __name__ == '__main__':
    with open("ssk_aprx_k14_lbda0.5.json", "r") as f:
        data = f.readlines()[0]

    docs = data.split(",")

    # remove the last one
    docs.pop(-1)
    docs[0] = docs[0][1:]

    res = dict()
    for doc in docs:
        json_str = "{" + doc + "}"
        d = json.loads(json_str)
        for k, v in d.items():
            res[k] = v

    with open("ssk_aprx_k14_lbda0.5.json", "w") as f:
        json.dump(res, f, indent=1)

