# LDraw Gym Environment

## Installation:
### Standalone
- Install [python 3](http://www.python.org) We recommend using [anaconda](http://www.anaconda.com))
- Optionally Install [pytorch](http://www.pytorch.org) This is optional and only necessary for running the training code.  It is not necessary for running the gym environments.
- Optionally Install [blender 2.90+](http://www.blender.org) with [Import LDraw Addon](https://github.com/TobyLobster/ImportLDraw) This is optional and only necessary for building brick obj files.
- Clone this repo somewhere
- Go to the top directory of the cloned repo and run ```pip install -e .```  This should automatically install:
  - [gym](https://github.com/openai/gym) We recommend installing gym using pip/anaconda.
  - [tqdm](https://github.com/tqdm/tqdm) Progress bar library.
- Install [renderpy](https://gitlab.cs.washington.edu/awalsman/renderpy) by cloning it from gitlab.  TODO: This should be moved to github and pip-ified.
  - Run ```pip install -e .``` inside the top renderpy directory to install it with python.
- Install [LDraw](https://www.ldraw.org/) using our script to download this and put it in the right place.  Just run `bin/install_ldraw`
- Install the [LDCad](http://www.melkert.net/LDCad) shadow library.  LDCad is a Lego CAD software that uses the LDraw data type.
- Install the [Open Model Repository](https://omr.ldraw.org/).  This contains files representing official Lego sets that have been reverse engineered by the online Lego community.  We have conveniently scraped this and packed it into a dataset.
  - Get ```OMR.zip``` from Aaron.
  - Unzip it in the ```brick-gym/data``` directory so that you have ```brick-gym/data/OMR/ldraw/...```
- Renderpy obj files
  - Get ```renderpy_assets.zip``` from Aaron.
  - Unzip it in the ```brick-gym/data``` directory so that you have ```brick-gym/data/renderpy/meshes...```

### Instructions for building objs from scratch
First open blender and make sure the [Import LDraw Addon](https://github.com/TobyLobster/ImportLDraw) is installed.  Then go into the script console and run:
```
import sys
sys.path.append('path/to/brick-gym')
import brick_gym.blender.export_obj as export_obj
export_obj.export_all(**export_obj.medium_settings)
```

## Units:
All 3D units are in [LDraw LDUs](http://www.ldraw.org/article/218.html).  One LDU is approximately 0.4 mm, so the physical extents of these scenes can be quite large.
