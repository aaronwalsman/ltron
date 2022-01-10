from pathlib import Path

from ltron.bricks.brick_scene import BrickScene
from ltron.dataset.submodel_extraction import blacklist_computation
import json
import os

import tqdm

def blacklist_out(
    directory, dest, blacklist_dest, threshold=400, blacklist=None):
    
    if blacklist is None:
        blacklist = []
    
    blacklist_path = os.path.join(blacklist_dest, 'blacklist_%i.json'%threshold)
    if os.path.exists(blacklist_path):
        with open(blacklist_path) as f:
            blacklist = blacklist + json.load(f)
    else:
        blacklist = blacklist + blacklist_computation(threshold)
    with open(blacklist_path, 'w') as f:
        json.dump(blacklist, f)
    path = Path(directory).expanduser()
    modelList = list(path.rglob('*'))
    print('-'*80)
    print('Removing Blacklisted Bricks From Scenes')
    for model in tqdm.tqdm(modelList):
        model = str(model)
        try:
            scene = BrickScene(track_snaps=False)
            scene.import_ldraw(model)
        except:
            print("Can't open: " + model + " during blacklisting")
            continue

        model_name = model.split("/")[-1]
        keep = [i+1 for i in range(len(scene.instances)) if scene.instances.instances[i+1].brick_shape.reference_name not in blacklist]
        # for i in range(len(scene.instances)):
        #     try:
        #         if scene.instances.instances[i+1].brick_shape.reference_name in blacklist:
        #             scene.remove_instance(i+1)
        #     except:
        #         print(scene.instances.instances)
        #         print(i)
        #         print(scene.instances.instances[i+1].brick_shape)
        scene.export_ldraw(dest + model_name, instances=keep)
