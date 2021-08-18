# coding: utf-8
# 2021/8/14 @ tongshiwei


def assign_features(src: list, tar: list, node_indexes: list):
    """

    Parameters
    ----------
    src: list of tensor
    tar: list of tensor
    node_indexes: list of tensor

    Returns
    -------
    src: list of tensor

    """
    for _src, _tar, node_index in zip(src, tar, node_indexes):
        src[node_index] = tar
    return src


if __name__ == '__main__':
    def while_yield():
        i = 0
        while i < 5:
            yield i
            i += 1
        print("hello world")


    for j in while_yield():
        print(j)
