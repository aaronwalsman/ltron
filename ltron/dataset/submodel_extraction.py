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
import numpy
import random


def compute_boxsize(instances, scene):
    instance_tran = {}
    for k in instances:
        instance_tran[k] = scene.instances.instances[k].transform

    instance_pos = {}
    for k in instances:
        instance_pos[k] = scene.instances.instances[k].brick_type.bbox

    point = []
    for ins, bbox in instance_pos.items():
        minb = bbox[0]
        maxb = bbox[1]
        point.append((numpy.matmul(instance_tran[ins], numpy.array([minb[0], minb[1], minb[2], 1])))[:3])
        point.append((numpy.matmul(instance_tran[ins], numpy.array([maxb[0], minb[1], minb[2], 1])))[:3])
        point.append((numpy.matmul(instance_tran[ins], numpy.array([minb[0], maxb[1], minb[2], 1])))[:3])
        point.append((numpy.matmul(instance_tran[ins], numpy.array([minb[0], minb[1], maxb[2], 1])))[:3])
        point.append((numpy.matmul(instance_tran[ins], numpy.array([maxb[0], maxb[1], minb[2], 1])))[:3])
        point.append((numpy.matmul(instance_tran[ins], numpy.array([minb[0], maxb[1], maxb[2], 1])))[:3])
        point.append((numpy.matmul(instance_tran[ins], numpy.array([maxb[0], minb[1], maxb[2], 1])))[:3])
        point.append((numpy.matmul(instance_tran[ins], numpy.array([maxb[0], maxb[1], maxb[2], 1])))[:3])

    min_y = 100000
    max_y = -1000000
    min_x = 100000
    max_x = -100000
    min_z = 100000
    max_z = -1000000
    for p in point:
        if p[1] > max_y:
            max_y = p[1]
        if p[1] < min_y:
            min_y = p[1]
        if p[0] > max_x:
            max_x = p[0]
        if p[0] < min_x:
            min_x = p[0]
        if p[2] > max_z:
            max_z = p[2]
        if p[2] < max_z:
            min_z = p[2]

    # if abs(max_y - min_y) - 35 > 0: return 10000
    # else: return -1000
    return max(abs(max_y - min_y), abs(max_x - min_x), abs(max_z - min_z))

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

def add_brick_box(limit, cur_mod, comp_list, instance_id, scene, used_brick = [], blacklist=[], debug=False):
    instance = scene.instances.instances[instance_id]
    connections = scene.get_all_snap_connections([instance])[str(instance_id)]
    if len(connections) == 0:
        if debug:
            print("no connection")
        return False
    box_map = {}
    for conn in connections:
        temp = copy.deepcopy(cur_mod)
        target = int(conn[0])
        if target in blacklist:
            continue
        if target in cur_mod:
            if debug:
                print("current")
            continue
        if target in used_brick:
            if debug:
                print("used")
            continue
        temp.append(target)
        box_map[target] = compute_boxsize(temp, scene)

    if len(box_map) == 0:
        if debug:
            print("no map")
        return False
    best_target = min(box_map, key=box_map.get)
    cur_mod.append(best_target)
    # used_brick.append(best_target)
    if len(cur_mod) >= limit:
        comp_list.append(cur_mod)
        return True
    else:
        status = add_brick_box(limit, cur_mod, comp_list, best_target, scene, used_brick)
        return status

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

# The method in use
def subcomponent_nonoverlap_extraction(limit, num_comp):
    global_count = 0
    folder_name = "subcomponents" + str(limit) + "/"
    try:
        os.stat(folder_name)
    except:
        os.mkdir(folder_name)
    path = Path("~/.cache/ltron/collections/omr/ldraw").expanduser()
    mpdlist = path.rglob('*mpd') # 1432
    stat = {"error" : [], "total_count" : 0, "models" : {}}

    # Iterate through mpd files
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
        # print(num_instances)
        components = []
        used_brick = []
        for i in range(num_instances):
            cur_list = []
            debug = False
            # if mpd == "/home/nanami/.cache/ltron/collections/omr/ldraw/42038-1 - Arctic Truck.mpd":
            #     debug = True
            status = add_brick_box(limit, [], cur_list, i+1, scene, used_brick, debug=debug)
            # print(status)
            # if mpd == "/home/nanami/.cache/ltron/collections/omr/ldraw/42038-1 - Arctic Truck.mpd":
            #     print(i)
            #     print(status)
            for subcomp in cur_list:
                if global_count == num_comp:
                    break
                subcomp.sort()
                if subcomp not in components:
                    decider = True
                    for ins in subcomp:
                        if ins in used_brick:
                            decider = False
                            break
                        used_brick.append(ins)
                    if decider:
                        components.append(subcomp)
                        global_count += 1

            if global_count == num_comp:
                break

        # print(global_count)

        count = 1
        modelname = mpd.split("/")[-1][:-4]
        for comp in components:
            temp_scene = BrickScene()
            temp_scene.import_ldraw(mpd)
            instances = copy.deepcopy(temp_scene.instances.instances)
            for k, v in instances.items():
                if k not in comp:
                    temp_scene.remove_instance(v)

            # temp_scene.export_ldraw(folder_name + modelname + "_"
            #                                                 + str(count) + ".mpd")
            count += 1

        stat['models'][modelname] = [num_instances, count - 1]

        if global_count == num_comp:
            break

    stat['total_count'] = global_count

    with open(folder_name + "stat.json", "w") as f:
        json.dump(stat, f)

def subcomponent_partition_extractoin(limit, num_comp):
    global_count = 0
    global_list = []
    folder_name = "subcomponents" + str(limit) + "/"
    try:
        os.stat(folder_name)
    except:
        os.mkdir(folder_name)
    path = Path("~/.cache/ltron/collections/omr/ldraw").expanduser()
    mpdlist = path.rglob('*mpd')  # 1432
    stat = {"error": []}

    while global_count < num_comp:

        # Iterate through mpd files
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
            used_brick = []

            ini_brick = random.randint(1, num_instances)
            cur_list = []
            add_brick(limit, [], cur_list, ini_brick, scene)

            for subcomp in cur_list:
                if global_count == num_comp:
                    break
                subcomp.sort()
                if subcomp not in components:
                    decider = True
                    for ins in subcomp:
                        if ins in used_brick:
                            decider = False
                            break
                        used_brick.append(ins)
                    if decider:
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
    # print(compute_boxsize(components['scene'].brick_scene.instances.instances.keys(), components['scene'].brick_scene))

def main():
    subcomponent_nonoverlap_extraction(8, 40000000)
    render("subcomponents8/6954-1 - Renegade_1.mpd")
    render("subcomponents8/6954-1 - Renegade_2.mpd")

if __name__ == '__main__' :
    main()

