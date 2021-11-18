import sys
sys.path.append('/media/awalsman/LEGODrive/ltron')
sys.path.append('/home/awalsman/Development/splendor-render')

import ltron.blender.export_obj as export_obj
export_obj.export_brick('3006.dat', directory='/home/awalsman/Documents/tmp', overwrite=True)
