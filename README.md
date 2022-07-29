![LTRON](assets/ltron_logo.png?raw=true "LTRON")

# Interactive Lego Machine Learning Environment

LTRON is an environment for interactive machine learning assembly problems using Lego bricks.  It is based on [LDraw](https://ldraw.org/) the [Open Model Repository](https://omr.ldraw.org/), and uses additional data from [LDCAD](http://www.melkert.net/LDCad).

If you use LTRON in acadmic work, please cite our [ECCV 2022 Paper](https://arxiv.org/abs/2207.13738):
```
@inproceedings{walsman2022break,
  author = {Aaron Walsman, Muru Zhang, Klemen Kotar, Karthik Desingh, Ali Farhadi, Dieter Fox},
  title = {Break and Make: Interactive Structural Understanding Using LEGO Bricks},
  booktitle = {European Conference on Computer Vision},
  year={2022}
}
```

### Examples
8661-1 - Carbon Star.mpd
![carbon_star](assets/carbon_star.png?raw=true)

31025 - Mountain Hut - Mountain Hut.mpd
![house](assets/house.png?raw=true)

10030-1 - Imperial Star Destroyer - UCS.mpd
![destroyer](assets/destroyer.png?raw=true)

## Installation:
To install LTRON run:
```
pip install ltron
ltron_asset_installer
```

The first line will download LTRON from pypi and install it in your python path. The second line will download and install a set of models and part files that are necessary for LTRON.  By default, these assets are installed to `~/.cache/ltron` and `~/.cache/splendor`.  This will take around 3GB of space.

Release Versions:
 - [Embodied AI Workshop, CVPR 2021:](https://embodied-ai.org/papers/LegoTron.pdf) (pypi 0.0.X) (branch TODO).
 - [ECCV 2022:](https://arxiv.org/abs/2207.13738) (pypi 1.0.X) (branch v1.0.0)
 - Ongoing Work (not in pypi) (branch master)

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
We have tried to make the LtronEnv structure modular so that it can be customized for a variety of observation types, action spaces and tasks.  Therefore the LtronEnv [gym](https://github.com/openai/gym) environment is a container for multiple components, each of which specify their own action/observation spaces and state variables.  When LtronEnv's reset/step/etc. method is called, it executes each of its component's reset/step/etc. methods in order and accumulating the result into a dictionary based on the component's name.  Currently only the Break and Make (ltron/gym/envs/break_and_make_env.py in v1.0.0) is stable.

## Further Notes:
All 3D units are in [LDraw LDUs](http://www.ldraw.org/article/218.html).  One LDU is approximately 0.4 mm, so the physical extents of these scenes can be quite large.

