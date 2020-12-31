# LDraw Gym Environment

## Dependencies:
- python 3 (recommend using [anaconda](http://www.anaconda.com))
- [openai gym](https://gym.openai.com/)
- [ldraw](https://www.ldraw.org/) (see `install_ldraw` script)
- [pytorch](http://www.pytorch.org) (optional for training using pytorch)
- [blender 2.90+](http://www.blender.org) with [Import LDraw Addon](https://github.com/TobyLobster/ImportLDraw) (optional for building brick obj files)

### LDraw
run `bin/install_ldraw`

## Units:
All 3D units are in [LDraw LDUs](http://www.ldraw.org/article/218.html).  One LDU is approximately 0.4 mm, so the physical extents of these scenes can be quite large.
