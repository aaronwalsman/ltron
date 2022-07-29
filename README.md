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
 - [Embodied AI Workshop, CVPR 2021:](https://embodied-ai.org/papers/LegoTron.pdf) (pypi 0.0.X) (branch TODO).
 - ECCV 2022: (pypi 1.0.X) (branch v1.0.0)

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

## Data Layout:
This section is current as of v1.0.X, but may be streamlined in future releases:

As noted above, the LTRON assets are placed by default in `~/.cache/ltron`.  This location can be changed by setting the `$LTRON_HOME` environment variable.  Additionally, some rendering assets are placed in `~/.cache/splendor`.  This location can be changed by setting the $SPLENDOR_HOME` environment variable.

All LTRON datasets are registered in `$LTRON_HOME/settings.cfg`.  In LTRON each dataset contains a json file that describes where to find the files associated with that dataset, and various metadata.  These are listed in the `[datasets]` header in the `settings.cfg` file.  These json files usually refer to a particular "collection" of ldraw files.  You can think of a collection as a root file path where several ldraw files and/or episode zip files live.  These locations are specified in the `[collections]` header in the `settings.cfg` file.

Looking the `$LTRON_HOME/collections/random_construction_6_6/rc_6_6.json` file we can see what kind of data it contains:
- `splits`: a set of names corresponding to a block of files used for training or testing.  `rc_6_6.json` contains nine splits: a `train_N`, `test_N` and `train_episodes_N` for N=2,4,8.  The path to each of these splits starts with `{random_construction_6_6}/...` which tells the system to look for these files inside the `random_construction_6_6` collection.
- `max_instances_per_scene`: an upper bound on number of instances that exist in any of the ldraw files used in the dataset.
- `max_edges_per_scene`: an upper bound on the number of connections between bricks that exist in any of the ldraw files used in this dataset.
- `shape_ids`: a class label for each brick shape used in this dataset
- `class_ids`: a separate class label for each brick color used in this dataset

Using this structure provides access to the following commands:
- `ltron.dataset.get_dataset_info('dataset_name')` returns the contents of the json dataset file
- `ltron.dataset.get_dataset_paths('dataset_name', 'split_name', subset=None, rank=0, size=1)` returns a list of paths belonging to a particular split.
- `ltron.dataset.get_zip_paths('dataset_name', 'split_name', subset=None, rank=0, size=1)` returns a zipfile object and a list of its contents for a particular split.


## LtronEnv Structure
We have tried to make the LtronEnv structure modular so that it can be customized for a variety of observation types, action spaces and tasks.  Therefore the LtronEnv gym environment is a container for multiple components, each of which specify their own action/observation spaces and state variables.  When LtronEnv's reset/step/etc. method is called, it executes each of its component's reset/step/etc. methods in order and accumulating the result into a dictionary based on the component's name.  Components do not need to implement every one of these functions.  For example the DatasetPathComponent (`ltron/gym/components/dataset.py`) only has a reset method which picks a file path from a known datset and stores it as a state variable that is visible to other components.  It does not have a step method because it doesn't need to change anything on a step-by-step basis.  Other components can read this path variable if the dataset component was passed to them when they were constructed.  To illustrate it is worth looking at an example.

Find the `segmentation_supervision_env` in `brick-gym/brick_gym/gym/standard_envs.py`  It is designed to generate data to train a segmentation model.  Note that it has the following components:
- DatasetPathComponent: loads paths from a dataset.  At the start of each new episode (reset) it picks one of these paths and stores it in the scene_path state variable.
- SceneComponent: this component contains the API for the current scene.  It takes the DatasetPathComponent as a constructor argument, so that it can read the scene_path variable.  When the episode resets, this component will read the scene_path and load that file so that other components can interact with it.  Note however that a DatasetPathComponent is not necessary.  You can instead construct this component with a single path that will always be loaded at reset if you are only ever interacting with one scene, or even specify nothing at all if you want to reset to a blank scene at each episode.
- Fixed/RandomizedAzimuthalViewpointComponent: this controls the viewpoint for each episode.  RandomizedAzimuthalViewpointComponent randomizes the viewpoint at the start of each episode.  It takes the SceneComponent as an argument so that it can manipulate the camera information in the scene as necessary.
- RandomFloatingBricks: adds randomized floating bricks to the scene for domain randomization.
- MaxEpisodeLengthComponent: ends the episode after a fixed number of frames.
- RandomizeColorsComponent: randomizes the colors of the bricks at the beginning of each episode for domain randomization.
- InstanceVisibilityComponent: provides an action space that allows an agent to hide bricks by specifying an instance id at each step.
- ColorRenderComponent: provides a render of the scene as a component of the observation space.
- SegmentationRenderComponent: provides a ground-truth segmentation mask of the scene as a component of the observation space.
- InstanceLabelComponent: provides a ground-truth labelling of the class id of each instance in the scene.

All of these components come together to form a single gym environment that picks a random path from the dataset at each episode reset, then loads that path and chooses a random viewpoint.  It then provides a render of the scene, with a ground-truth segmentation mask and node labelling at each frame, and provides an interface to hide bricks interactively from step to step.  

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
