# coding: utf-8
# 2021/7/12 @ tongshiwei

__all__ = ["PadSequence", "pad_sequence"]


class PadSequence(object):
    """Pad the sequence.

    Pad the sequence to the given `length` by inserting `pad_val`. If `clip` is set,
    sequence that has length larger than `length` will be clipped.

    Parameters
    ----------
    length : int
        The maximum length to pad/clip the sequence
    pad_val : number
        The pad value. Default 0
    clip : bool
    """

    def __init__(self, length, pad_val=0, clip=True):
        self._length = length
        self._pad_val = pad_val
        self._clip = clip

    def __call__(self, sample: list):
        """

        Parameters
        ----------
        sample : list of number

        Returns
        -------
        ret : list of number
        """
        sample_length = len(sample)
        if sample_length >= self._length:
            if self._clip and sample_length > self._length:
                return sample[:self._length]
            else:
                return sample
        else:
            return sample + [
                self._pad_val for _ in range(self._length - sample_length)
            ]


def pad_sequence(sequence: list, max_length=None, pad_val=0, clip=True):
    """

    Parameters
    ----------
    sequence
    max_length
    pad_val
    clip

    Returns
    -------

    Examples
    --------
    >>> seq = [[4, 3, 3], [2], [3, 3, 2]]
    >>> pad_sequence(seq)
    [[4, 3, 3], [2, 0, 0], [3, 3, 2]]
    >>> pad_sequence(seq, pad_val=1)
    [[4, 3, 3], [2, 1, 1], [3, 3, 2]]
    >>> pad_sequence(seq, max_length=2)
    [[4, 3], [2, 0], [3, 3]]
    >>> pad_sequence(seq, max_length=2, clip=False)
    [[4, 3, 3], [2, 0], [3, 3, 2]]
    """
    padder = PadSequence(max([len(seq) for seq in sequence]) if max_length is None else max_length, pad_val, clip)
    return [padder(seq) for seq in sequence]
