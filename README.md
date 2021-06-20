# LTRON: Lego Interactive Machine Learning Environment

## Installation:
- Install [python 3](http://www.python.org) We recommend using [anaconda](http://www.anaconda.com))
- Optionally Install [pytorch](http://www.pytorch.org) This is optional and only necessary for running the training code.  It is not necessary for running the gym environments.
  - Also install [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) and its dependencies.
- Optionally Install [blender 2.90+](http://www.blender.org) with [Import LDraw Addon](https://github.com/TobyLobster/ImportLDraw) This is optional and only necessary for building brick obj files.
- Clone this repo somewhere
- Go to the top directory of the cloned repo and run `pip install -e .`  This should automatically also install:
  - [numpy](https://numpy.org/)
  - [gym](https://github.com/openai/gym)
  - [tqdm](https://github.com/tqdm/tqdm)
- Install [splendor-render](https://gitlab.cs.washington.edu/awalsman/splendor-render) by cloning it from gitlab.  TODO: This should be moved to github and pip-ified.
  - Run `pip install -e .` inside the top splendor-render directory to install it with python.
- Install [LDraw](https://www.ldraw.org/) using our script to download this and put it in the right place.  Just run `bin/install_ldraw`
- Install the [LDCad](http://www.melkert.net/LDCad) shadow library.  TODO: Incorporate this into the `install_ldraw` script.
  - Get `shadow.zip` from Aaron.
  - Unzip it in the `brick-gym/data` directory so that you have `brick-gym/shadow/offLib/offLibShadow/...`
- Install the [Open Model Repository](https://omr.ldraw.org/).  This contains files representing official Lego sets that have been reverse engineered by the online Lego community.  We have conveniently scraped this and packed it into a dataset.
  - Get `OMR.zip` from Aaron.
  - Unzip it in the `brick-gym/data` directory so that you have `brick-gym/data/OMR/ldraw/...`
- Renderpy obj files
  - Get `splendor_assets.zip` from Aaron.
  - Unzip it in the `brick-gym/data` directory so that you have `brick-gym/data/splendor/meshes...`

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
brick-gym
- setup.cfg: A configuration file that lets you change the locations of datasets and other directories.  The paths in this file can be accessed using the `brick_gym.config` module.
- splendor_assets.cfg: A configuration file that points to the splendor-render assets (meshes) necessary for rendering these scenes.
- bin: All stand-alone python scripts that can be run from the command line should be here.
- brick_gym: Package directory containing all the libraries that can be imported by these scripts.  There shouldn't be any stand-alone scripts in here, although there are some at the moment that need to be cleaned up.
  - blender: modules used by blender for converting ldraw files to objs
  - bricks: the primary API used to represent scenes
  - dataset: modules for loading different datasets
  - gym: the gym RL interface.  Uses `brick_gym.bricks` to interface with a scene.
  - ldraw: modules to parse the LDraw and LDCad data formats
  - random_stack: old (out of date) modules for running experiments with the random_stack dataset.  TODO: Remove this once all the new training scripts are online
  - torch: pytorch models and training code.  No torch code should be anywhere outside this directory.
- data: Data files that are too large to be committed to the repo.

## BrickGym Structure
We have tried to make the BrickGym structure modular so that it can be customized for a variety of observation types, action spaces and tasks.  Therefore the BrickGym gym environment is a container for multiple components, each of which specify their own action/observation spaces and state variables.  The way this works is that each component has their own reset/step/etc. methods.  When BrickGym's reset/step/etc. method is called, it executes each of its component's reset/step/etc. methods and accumulating the result into a dictionary based on the component's name.  Components do not need to implement every one of these functions.  For example the DatasetPathComponent only has a reset method which picks a file path from a known datset and stores it as a state variable that is visible to other components that may load that file path or do whatever else they want with it.  It does not have a step method because it doesn't need to change anything on a step-by-step basis.  Other components can read this path variable if the dataset component was passed to them when they were constructed.  To illustrate it is worth looking at an example.

Find the `segmentation_supervision_env` in `brick-gym/brick_gym/gym/standard_envs.py`  It is designed to generate data to train a segmentation model.  Note that it has the following components:
- DatasetPathComponent: loads paths from a dataset.  At the start of each new episode (reset) it picks one of these paths and stores it in the scene_path state variable.
- SceneComponent: this component contains the API for the current scene.  It takes the DatasetPathComponent as a constructor argument, so that it can read the scene_path variable.  When the episode resets, this component will read the scene_path and load that file so that other components can interact with it.  Note however that a DatasetPathComponent is not necessary.  You can instead construct this component with a single path that will always be loaded at reset if you are only ever interacting with one scene, or even specify nothing at all if you want to reset to a blank scene at each episode.
- RandomizedAzimuthalViewpointComponent: this randomizes the viewpoint at the start of each episode.  It takes the SceneComponent as an argument so that it can manipulate the camera information in the scene as necessary.
- InstanceVisibilityComponent: provides an action space that allows an agent to hide bricks by specifying an instance id at each step.
- ColorRenderComponent: provides a render of the scene as a component of the observation space.
- SegmentationRenderComponent: provides a ground-truth segmentation mask of the scene as a component of the observation space.
- InstanceLabelComponent: provides a ground-truth labelling of the class id of each instance in the scene.

All of these components come together to form a single gym environment that picks a random path from the dataset at each episode reset, then loads that path and chooses a random viewpoint.  It then provides a render of the scene, with a ground-truth segmentation mask and node labelling at each frame, and provides an interface to hide bricks interactively from step to step.  Looking in `brick-gym/brick_gym/experiments/image_generation.py` we can see a function that uses this gym environment to build a dataset for learning segmentation.  At each step it picks a random instance to hide, then gets a new observation and saves the images to disk.  This is not a complete example yet (it should probably save the node ids as a json file or something as well, and put the data somewhere sensible rather than just dumping everything to the current directory, but you get the idea.

## Further Notes:
All 3D units are in [LDraw LDUs](http://www.ldraw.org/article/218.html).  One LDU is approximately 0.4 mm, so the physical extents of these scenes can be quite large.
