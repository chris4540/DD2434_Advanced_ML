from ssk_kernel.apprx_kernel import SSKKernelApprox

if __name__ == '__main__':
    kernel = SSKKernelApprox(5, 0.5, True)


    # get all docuements
    docs = list(kernel.data2idx.keys())
    doc1 = docs[-1]
    doc2 = docs[-2]
    ret = kernel(doc1, doc2)

    kernel.save_kernel_to_json()
