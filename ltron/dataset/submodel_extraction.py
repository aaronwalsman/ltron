import json
from ltron.ldraw.documents import LDrawMPDMainFile
from ltron.bricks.brick_scene import BrickScene
from pathlib import Path
from ltron.gym.components.scene import SceneComponent
from ltron.gym.components.episode import MaxEpisodeLengthComponent
from ltron.gym.components.render import ColorRenderComponent
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from ltron.gym.ltron_env import LtronEnv
from ltron.gym.components.viewpoint import (
        ControlledAzimuthalViewpointComponent,
        RandomizedAzimuthalViewpointComponent,
        FixedAzimuthalViewpointComponent)
import copy
import collections
import math
import os

def add_brick(limit, cur_mod, comp_list, instance_id, scene):
    instance = scene.instances.instances[instance_id]
    connections = scene.get_all_snap_connections([instance])[str(instance_id)]
    if len(connections) == 0: return False
    for conn in connections:
        target = int(conn[0])
        if target in cur_mod:
            continue
        cur_mod.append(target)
        if len(cur_mod) >= limit:
            comp_list.append(cur_mod)
            return True
        else:
            add_brick(limit, cur_mod, comp_list, target, scene)
            return True

# limit - number of bricks in each subcomponent
# num_comp - number of subcomponents want to generate(won't run over this number)
def subcomponent_extraction(limit, num_comp):

    global_count = 0
    folder_name = "subcomponents" + str(limit) + "/"
    try:
        os.stat(folder_name)
    except:
        os.mkdir(folder_name)
    path = Path("~/.cache/ltron/collections/omr/ldraw").expanduser()
    mpdlist = path.rglob('*mpd') # 1432
    ldrlist = path.rglob('*ldr') # 62
    stat = {"error" : []}

    for mpd in mpdlist:
        print(mpd)
        mpd = str(mpd)
        try:
            mpdfile = LDrawMPDMainFile(mpd)
            scene = BrickScene(track_snaps=True)
            scene.import_ldraw(mpd)
        except:
            stat['error'].append(mpd)
            continue

        num_instances = len(scene.instances.instances)
        components = []
        for i in range(num_instances):
            cur_list = []
            add_brick(limit, [], cur_list, i+1, scene)
            for subcomp in cur_list:
                if global_count == num_comp:
                    break
                subcomp.sort()
                if subcomp not in components:
                    components.append(subcomp)
                    global_count += 1
            if global_count == num_comp:
                break

        count = 1
        modelname = mpd.split("/")[-1][:-4]
        for comp in components:
            temp_scene = BrickScene()
            temp_scene.import_ldraw(mpd)
            instances = copy.deepcopy(temp_scene.instances.instances)
            for k, v in instances.items():
                if k not in comp:
                    temp_scene.remove_instance(v)

            temp_scene.export_ldraw(folder_name + modelname + "_"
                                                            + str(count) + ".mpd")
            count += 1

        if global_count == num_comp:
            break

# Render a .mpd file
def render(filepath):
    components = collections.OrderedDict()
    components['scene'] = SceneComponent(dataset_component=None,
                                         initial_scene_path=filepath,
                                         track_snaps=True)
    components['episode'] = MaxEpisodeLengthComponent(1000)
    components['camera'] = FixedAzimuthalViewpointComponent(
        components['scene'],
        azimuth=math.radians(-135.),
        elevation=math.radians(-30.),
        aspect_ratio=1,
        distance=(2, 2))
    components['render'] = ColorRenderComponent(512, 512, components['scene'])

    env = LtronEnv(components)
    obs = env.reset()
    imshow(obs['render'])
    plt.show()

def main():
    subcomponent_extraction(8, 50)

if __name__ == '__main__' :
    main()

