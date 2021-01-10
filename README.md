# LDraw Gym Environment

## Dependencies:
### Standalone
- [python 3](http://www.python.org) We recommend using [anaconda](http://www.anaconda.com))
- [pytorch](http://www.pytorch.org) This is optional and only necessary for running the training code.  It is not necessary for running the gym environments.
- [blender 2.90+](http://www.blender.org) with [Import LDraw Addon](https://github.com/TobyLobster/ImportLDraw) This is optional and only necessary for building brick obj files.

### Should be automatically installed when this package is installed
- [gym](https://github.com/openai/gym) We recommend installing gym using pip/anaconda.
- [tqdm](https://github.com/tqdm/tqdm) Progress bar library.

### Need from github (for now, should pip-ify this)
- [renderpy](https://gitlab.cs.washington.edu/awalsman/renderpy)

### Assets:
- [LDraw](https://www.ldraw.org/) is a format and repository for describing Lego bricks and models.  We have 
run `bin/install_ldraw`
- [LDCad](http://www.melkert.net/LDCad) is a Lego CAD software that uses the LDraw data type.
- [Open Model Repository](https://omr.ldraw.org/) provided by LDraw.  This contains files representing official Lego sets that have been reverse engineered by the online Lego community.  We have conveniently scraped this and packed it into a dataset.
  - Get ```OMR.zip``` from Aaron.
  - Unzip it in the ```brick-gym/data``` directory so that you have ```brick-gym/data/OMR/ldraw/...```

### Renderpy obj files
- Get ```renderpy_assets.zip``` from Aaron.
- Unzip it in the ```brick-gym/data``` directory so that you have ```brick-gym/data/renderpy/meshes...```

#### Instructions for building objs from scratch
First open blender and make sure the [Import LDraw Addon](https://github.com/TobyLobster/ImportLDraw) is installed.  Then go into the script console and run:
```
import sys
sys.path.append('path/to/brick-gym')
import brick_gym.blender.export_obj as export_obj
export_obj.export_all(**export_obj.medium_settings)
```

## Units:
All 3D units are in [LDraw LDUs](http://www.ldraw.org/article/218.html).  One LDU is approximately 0.4 mm, so the physical extents of these scenes can be quite large.
