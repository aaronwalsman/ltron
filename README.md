# LDraw Gym Environment

## Installation:
- Install [python 3](http://www.python.org) We recommend using [anaconda](http://www.anaconda.com))
- Optionally Install [pytorch](http://www.pytorch.org) This is optional and only necessary for running the training code.  It is not necessary for running the gym environments.
- Optionally Install [blender 2.90+](http://www.blender.org) with [Import LDraw Addon](https://github.com/TobyLobster/ImportLDraw) This is optional and only necessary for building brick obj files.
- Clone this repo somewhere
- Go to the top directory of the cloned repo and run `pip install -e .`  This should automatically also install:
  - [numpy](https://numpy.org/)
  - [gym](https://github.com/openai/gym)
  - [tqdm](https://github.com/tqdm/tqdm)
- Install [renderpy](https://gitlab.cs.washington.edu/awalsman/renderpy) by cloning it from gitlab.  TODO: This should be moved to github and pip-ified.
  - Run `pip install -e .` inside the top renderpy directory to install it with python.
- Install [LDraw](https://www.ldraw.org/) using our script to download this and put it in the right place.  Just run `bin/install_ldraw`
- Install the [LDCad](http://www.melkert.net/LDCad) shadow library.  TODO: Incorporate this into the `install_ldraw` script.
  - Get `shadow.zip` from Aaron.
  - Unzip it in the `brick-gym/data` directory so that you have `brick-gym/shadow/offLibShadow/...`
- Install the [Open Model Repository](https://omr.ldraw.org/).  This contains files representing official Lego sets that have been reverse engineered by the online Lego community.  We have conveniently scraped this and packed it into a dataset.
  - Get `OMR.zip` from Aaron.
  - Unzip it in the `brick-gym/data` directory so that you have `brick-gym/data/OMR/ldraw/...`
- Renderpy obj files
  - Get `renderpy_assets.zip` from Aaron.
  - Unzip it in the `brick-gym/data` directory so that you have `brick-gym/data/renderpy/meshes...`

### Instructions for building objs from scratch
If you'd rather First open blender and make sure the [Import LDraw Addon](https://github.com/TobyLobster/ImportLDraw) is installed.  Then go into the script console and run:
```
import sys
sys.path.append('path/to/brick-gym')
import brick_gym.blender.export_obj as export_obj
export_obj.export_all(**export_obj.medium_settings)
```

## Testing Things Out:
Once installed you should be able to run `brick_viewer` to interactively inspect a lego model.  For example from the top directory you can run:

`brick_viewer "data/OMR/ldraw/8661-1 - Carbon Star.mpd"`

Or:

`brick_viewer "data/OMR/ldraw/75050-1 - B-Wing.mpd"`

Or:

`brick_viewer "data/OMR/ldraw/75060 - Slave I.mpd"`

You can interact with the model by clicking on it and dragging the mouse around (warning dragging on the background causes things to fly around I'm working on it).  LMB - Orbit.  RMB - Pan.  Scroll - Zoom.  There are a few keys you can press `h` to hide the brick you are hovering over, `v` to show all hidden bricks and `m` to switch back and forth between mask mode and regular rendering.  See other options in `brick_gym/visualization/brick_viewer.py` TODO: Document.

## Project Structure
- brick-gym/setup.cfg: A configuration file that lets you change the locations of datasets and other directories.  The paths in this file can be accessed using the `brick_gym.config` module.
- brick-gym/assets.cfg: A configuration file that points to the renderpy assets (meshes) necessary for rendering these scenes.
- brick-gym/bin: All stand-alone python scripts that can be run from the command line should be here.
- brick-gym/brick_gym: Package directory containing all the libraries that can be imported by these scripts.  There shouldn't be any stand-alone scripts in here, although there are some at the moment that need to be cleaned up.
  - brick-gym/brick_gym/blender: modules used by blender for converting ldraw files to objs
  - brick-gym/brick_gym/bricks: the primary API used to represent scenes
  - brick-gym/brick_gym/dataset: modules for loading different datasets
  - brick-gym/brick_gym/gym: the gym RL interface.  Uses `brick_gym.bricks` to interface with a scene.
  - brick-gym/brick_gym/ldraw: modules to parse the LDraw and LDCad data formats
  - brick-gym/brick_gym/random_stack: old (out of date) modules for running experiments with the random_stack dataset.  TODO: Remove this once all the new training scripts are online
  - brick-gym/brick_gym/torch: pytorch models and training code.  No torch code should be anywhere outside this directory.
- brick-gym/data: Data files that are too large to be committed to the repo.

## BrickGym Structure
We have tried to make the BrickGym structure modular so that it can be customized for a variety of observation types, action spaces and tasks.  

## Further Notes:
All 3D units are in [LDraw LDUs](http://www.ldraw.org/article/218.html).  One LDU is approximately 0.4 mm, so the physical extents of these scenes can be quite large.
