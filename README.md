![LTRON](assets/ltron_logo.png?raw=true "LTRON")

# Interactive Lego Machine Learning Environment

## Installation:
To install LTRON run:
```
pip install ltron
ltron_asset_installer
```

The first line will download LTRON from pypi and install it in your python path. The second line will download and install a set of models and part files that are necessary for LTRON.  By default, these assets are installed to `~/.cache/ltron` and `~/.cache/splendor`.  This will take around 3GB of space.

Release Versions:
 - 0.0.X : [Embodied AI Workshop, CVPR 2021](https://embodied-ai.org/papers/LegoTron.pdf).
 - 1.0.X : ECCV 2022

The code in this repo is under active development and the API/interfaces are not stable.  The pypi versions listed above belong to different publications.  The github branch `v1.0.0` corresponds to the ECCV 2022 version.

## Testing Things Out:
Once installed you should be able to run `ltron_viewer` to interactively inspect a lego model.  For example from the top directory you can run:

`ltron_viewer "~/.cache/ltron/collections/omr/ldraw/8661-1 - Carbon Star.mpd"`

Or:

`ltron_viewer "~/.cache/ltron/collections/omr/ldraw/75050-1 - B-Wing.mpd"`

Or:

`ltron_viewer "~/.cache/ltron/collections/omr/ldraw/75060 - Slave I.mpd"`

You can interact with the model by clicking on it and dragging the mouse around.  LMB - Orbit.  RMB - Pan.  Scroll - Zoom.  There are a few keys you can press `h` to hide the brick you are hovering over, `v` to show all hidden bricks and `m` to switch back and forth between mask mode and regular rendering.  See other options in `ltron/visualization/ltron_viewer.py`.

## Requirements:
```
gym
numpy
pyquaternion
gdown
tqdm
splendor-render
```
The splendor-render package only works on Ubuntu at the moment, and requires OpenGL 4.6.  As long as you have a modern GPU with recent drivers you should be fine.

You may need to install freeglut:
```
sudo apt-get install freeglut3-dev
```

## File Formats
LTRON uses the [LDraw file formats](https://www.ldraw.org/article/218) (.dat, .ldr, .mpd) to describe bricks and models.  We convert all bricks to obj files for use in splendor-render.  These are installed to `~/.cache/splendor/ltron_assets_low` or `~/.cache/splendor/ltron_assets_high`.

The Open Model Repository files are installed to `~/.cache/ltron/collections/omr/ldraw`.

### Instructions for building objs from scratch
We bundle brick meshes and install them with `ltron_install_assets` but it is also possible (but time-consuming) to build them from scratch. First open blender and make sure the [Import LDraw Addon](https://github.com/TobyLobster/ImportLDraw) is installed.  Then go into the script console and run:
```
import sys
sys.path.append('path/to/ltron')
import ltron.blender.export_obj as export_obj
ltron.export_all(**export_obj.medium_settings)
```

## Further Notes:
All 3D units are in [LDraw LDUs](http://www.ldraw.org/article/218.html).  One LDU is approximately 0.4 mm, so the physical extents of these scenes can be quite large.
