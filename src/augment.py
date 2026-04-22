import numpy as np


# The 8 members of the dihedral group D4: all combos of flips + 90-degree rotations.
# Applied identically to (lr, hr) pairs so spatial alignment is preserved.
_AUGMENTATIONS = [
    lambda x: x,
    lambda x: np.rot90(x, 1),
    lambda x: np.rot90(x, 2),
    lambda x: np.rot90(x, 3),
    lambda x: np.fliplr(x),
    lambda x: np.rot90(np.fliplr(x), 1),
    lambda x: np.rot90(np.fliplr(x), 2),
    lambda x: np.rot90(np.fliplr(x), 3),
]


def augment_pair(lr, hr, idx=None):
    """Apply one of the 8 D4 symmetry transforms to an (lr, hr) pair.

    Args:
        lr: numpy array (H, W, C)
        hr: numpy array (H, W, C)
        idx: which transform to apply (0-7). Random if None.

    Returns:
        (lr_aug, hr_aug) as contiguous uint8 arrays.
    """
    if idx is None:
        idx = np.random.randint(8)
    fn = _AUGMENTATIONS[idx]
    return np.ascontiguousarray(fn(lr)), np.ascontiguousarray(fn(hr))
