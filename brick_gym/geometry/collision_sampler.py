from numpy import np
from numpy.linalg import inv
from itertools import product, chain


def get_all_snap_rotations(snap):
    cloned_snaps = []
    for theta in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
        # Rotate around the y axis
        # From https://en.wikipedia.org/wiki/Rotation_matrix
        rotation = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ])
        transform = snap.transform @ rotation @ inv(snap.transform)
        cloned_snaps.append(snap.transformed_copy(transform))
    return cloned_snaps


def get_all_transformed_snaps(snaps):
    male = [s for s in snaps if s.gender.lower() == 'm']
    all_male = []
    for snap in male:
        all_male.extend(get_all_snap_rotations(snap))
    female = [s for s in snaps if s.gender.lower() == 'f']
    all_female = []
    for snap in female:
        all_female.extend(get_all_snap_rotations(snap))
    return {
        'male': all_male,
        'female': all_female,
    }


def get_all_transformed_snap_pairs(instance1, instance2):
    snaps1 = get_all_transformed_snaps(instance1.get_snaps())
    snaps2 = get_all_transformed_snaps(instance2.get_snaps())
    return chain(
        product(snaps1['male'], snaps2['female']),
        product(snaps1['female'], snaps1['male']),
    )
