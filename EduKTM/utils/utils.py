# coding: utf-8
# 2021/5/24 @ tongshiwei


__all__ = ["as_list"]


def as_list(obj) -> list:
    r"""A utility function that converts the argument to a list
    if it is not already.

    Parameters
    ----------
    obj : object
        argument to be converted to a list

    Returns
    -------
    list_obj: list
        If `obj` is a list or tuple, return it. Otherwise,
        return `[obj]` as a single-element list.

    Examples
    --------
    >>> as_list(1)
    [1]
    >>> as_list([1])
    [1]
    >>> as_list((1, 2))
    [1, 2]
    """
    if isinstance(obj, list):
        return obj
    elif isinstance(obj, tuple):
        return list(obj)
    else:
        return [obj]
