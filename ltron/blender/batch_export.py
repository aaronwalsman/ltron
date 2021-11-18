import os
import argparse
from subprocess import Popen, PIPE, STDOUT

import tqdm

import splendor
import splendor.assets as assets
import ltron
from ltron.dataset.parts import all_ldraw_parts

parser = argparse.ArgumentParser()
parser.add_argument('blender', type=str)
parser.add_argument('--output-path', type=str, default=None)
parser.add_argument('--bricks', type=str, default=None)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--quality', type=str, default='medium')

def main():
    
    args = parser.parse_args()
    if args.bricks is None:
        bricks = all_ldraw_parts()
    else:
        bricks = [b.strip() for b in args.bricks.split(',')]
    
    if args.output_path is None:
        ltron_assets = assets.AssetLibrary('ltron_assets')
        output_path = ltron_assets['meshes'].paths[0]
    else:
        output_path = args.output_path
    
    exported = []
    skipped = []
    for brick in tqdm.tqdm(bricks):
        brick_dest = os.path.join(output_path, brick.replace('.dat', '.obj'))
        if os.path.exists(brick_dest) and not args.overwrite:
            skipped.append(brick)
            continue
        
        p = Popen(
            [args.blender, '--background', '--python-console'],
            stdout=PIPE,
            stdin=PIPE,
            stderr=PIPE,
        )
        splendor_path = os.path.split(os.path.split(splendor.__file__)[0])[0]
        ltron_path = os.path.split(os.path.split(ltron.__file__)[0])[0]
        commands = [
            b'import sys',
            b'sys.path.append("%s")'%str.encode(splendor_path),
            b'sys.path.append("%s")'%str.encode(ltron_path),
            b'import ltron.blender.export_obj as export_obj',
            b'export_obj.export_brick('
                b'"%s",'
                b'directory="%s",'
                b'overwrite=True,'
                b'**export_obj.%s_settings'
            b')'%(
                str.encode(brick),
                str.encode(output_path),
                str.encode(args.quality),
            )
        ]
        stdout = p.communicate(input=b';'.join(commands))
        if args.debug:
            print(stdout)
        
        exported.append(brick)
    
    print('Skipped: %s'%(','.join(skipped)))
    print('Exported: %s'%(','.join(exported)))

if __name__ == '__main__':
    main()
