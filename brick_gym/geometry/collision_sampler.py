import numpy as np
from numpy.linalg import inv
from itertools import product, chain


def get_all_snap_rotations(snap):
    cloned_snaps = []
    for theta in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
        # Rotate around the y axis
        # From https://en.wikipedia.org/wiki/Rotation_matrix
        rotation = np.array([
            [np.cos(theta), 0, np.sin(theta), 0],
            [0, 1, 0, 0],
            [-np.sin(theta), 0, np.cos(theta), 0],
            [0, 0, 0, 1],
        ])
        transform = snap.transform @ rotation @ inv(snap.transform)
        cloned_snaps.append(snap.transformed_copy(transform))
    return cloned_snaps


def get_all_transformed_snaps(snaps):
    """
    Returns a list of all male snaps with quarter rotations about y
    and return a list of female snaps (no rotations applied) because
    only one piece needs to rotate
    """
    male = [s for s in snaps if s.gender.lower() == 'm']
    all_male = []
    for snap in male:
        all_male.extend(get_all_snap_rotations(snap))
    female = [s for s in snaps if s.gender.lower() == 'f']
    return {
        'male': all_male,
        'female': female,
    }


def get_all_transformed_snap_pairs(instance1_snaps, instance2_snaps):
    snaps1 = get_all_transformed_snaps(instance1_snaps)
    snaps2 = get_all_transformed_snaps(instance2_snaps)
    return chain(
        product(snaps1['male'], snaps2['female']),
        product(snaps1['female'], snaps2['male']),
    )
