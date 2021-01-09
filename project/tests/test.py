import ssk_kernel
from ssk_kernel.ssk_kernel_approx_wrap import SSKKernelApprox

if __name__ == '__main__':
    # with SSKKernel(5, 0.5) as k:
    #     ret = k("car", "cat")
    #     print(ret)
    k = SSKKernelApprox(5, 0.5)
    docs = []
    for key in k.data2idx.keys():
        print(key)
        docs.append(key)

    ret = k(docs[-2], docs[-1])
    # ret = k("car", "cat")
    print(ret)
    # k.save_kernel_to_json()

