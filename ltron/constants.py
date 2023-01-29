# revisit this once we have all snaps accounted for
MAX_SNAPS_PER_BRICK = 256 #4096
MAX_INSTANCES_PER_SCENE = 16384

# default bounds on the scene in LDU
# 20 LDU = 1 stud + brick width, so 100000 = 5000 studs
DEFAULT_WORLD_BBOX = ((-100000,-100000,-100000),(100000,100000,100000))
