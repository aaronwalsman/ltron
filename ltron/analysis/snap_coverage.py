import tqdm

from ltron.ldraw.parts import LDRAW_PARTS, LDRAW_BLACKLIST_ALL
from ltron.bricks.brick_shape import BrickShape
from ltron.bricks.snap import UnsupportedSnap

complete = []
incomplete = []

iterate = tqdm.tqdm(LDRAW_PARTS)

for part in iterate:
    try:
        shape = BrickShape(part)
    except Exception as ex:
        if isinstance(ex, KeyboardInterrupt):
            raise
        else:
            continue
    
    if any(isinstance(snap, UnsupportedSnap) for snap in shape.snaps):
        incomplete.append(part)
        print(part)
    else:
        complete.append(part)
    
    iterate.set_description(
        '%.04f'%(len(complete) / (len(complete) + len(incomplete))))
