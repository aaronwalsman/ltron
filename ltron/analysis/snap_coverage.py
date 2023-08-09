import tqdm

from ltron.ldraw.parts import LDRAW_PARTS, LDRAW_BLACKLIST_ALL
from ltron.bricks.brick_shape import BrickShape
from ltron.bricks.snap import (
    UnsupportedSnap, UnsupportedCylinderSnap, UnsupportedFingerSnap
)

complete = []
incomplete = []

unsupported_cylinders = []
unsupported_cylinder_types = {}
unsupported_fingers = []
unsupported_other = []

iterate = tqdm.tqdm(LDRAW_PARTS)

for part in iterate:
    try:
        shape = BrickShape(part)
    except Exception as ex:
        if isinstance(ex, KeyboardInterrupt):
            raise
        else:
            continue
    
    is_complete = True
    for snap in shape.snaps:
        if isinstance(snap, UnsupportedSnap):
            is_complete = False
            if isinstance(snap, UnsupportedCylinderSnap):
                unsupported_cylinders.append(snap)
                flags = snap.command.flags
                key = (flags['gender'], flags['caps'], flags['secs'])
                if key not in unsupported_cylinder_types:
                    unsupported_cylinder_types[key] = []
                unsupported_cylinder_types[key].append((snap, shape))
            elif isinstance(snap, UnsupportedFingerSnap):
                unsupported_fingers.append(snap)
            else:
                unsupported_other.append(snap)
    
    if is_complete:
        complete.append(part)
    else:
        incomplete.append(part)
        print(part)
    
    iterate.set_description(
        '%.04f'%(len(complete) / (len(complete) + len(incomplete))))

print('Cylinders:', len(unsupported_cylinders))
print('Fingers:', len(unsupported_fingers))
print('Other:', len(unsupported_other))

unsupported_counts = {k:len(v) for k,v in unsupported_cylinder_types.items()}
sorted_unsupported = sorted((v,k) for k,v in unsupported_counts.items())

for v,k in reversed(sorted_unsupported):
    print(k, ':', v)

breakpoint()
