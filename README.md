# LDraw Gym Environment

## Dependencies:
- python 3 (recommend using [anaconda](http://www.anaconda.com))
- [pytorch](http://www.pytorch.org)
- [blender 2.90+](http://www.blender.org) with [Import LDraw Addon](https://github.com/TobyLobster/ImportLDraw) (optional for building brick obj files)

### LDraw
run `bin/install_ldraw`

### Instructions for building objs from scratch
First open blender and make sure the [Import LDraw Addon](https://github.com/TobyLobster/ImportLDraw) is installed.  Then go into the script console and run:
```import sys
sys.path.append('path/to/brick-gym')
import brick_gym.blender.blender_export_obj as blender_export_obj
blender_export_obj.export_all()
```

## Units:
All 3D units are in [LDraw LDUs](http://www.ldraw.org/article/218.html).  One LDU is approximately 0.4 mm, so the physical extents of these scenes can be quite large.
