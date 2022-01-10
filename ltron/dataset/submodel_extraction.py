import time
import json
import tqdm
from ltron.ldraw.documents import LDrawMPDMainFile
from ltron.bricks.brick_scene import BrickScene
from pathlib import Path
# from ltron.gym.components.scene import SceneComponent
from ltron.gym.components.episode import MaxEpisodeLengthComponent
from ltron.gym.components.render import ColorRenderComponent
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
# from ltron.gym.ltron_env import LtronEnv
from ltron.gym.components.viewpoint import (
        ControlledAzimuthalViewpointComponent,
        RandomizedAzimuthalViewpointComponent,
        FixedAzimuthalViewpointComponent)
from ltron.bricks.brick_shape import BrickShape
import copy
import collections
import math
import os
import numpy
import random
import tqdm

def compute_boxsize_old(instances, scene):
    instance_tran = {}
    for k in instances:
        instance_tran[k] = scene.instances.instances[k].transform

    instance_pos = {}
    for k in instances:
        instance_pos[k] = scene.instances.instances[k].brick_shape.bbox

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
        if p[2] < min_z:
            min_z = p[2]

    # if abs(max_y - min_y) - 35 > 0: return 10000
    # else: return -1000
    return max(abs(max_y - min_y), abs(max_x - min_x), abs(max_z - min_z))

# The complete parameter is somehow not working
def compute_boxsize(instance, scene, complete=False):
    if complete:
        all_vertices = numpy.concatenate([scene.instances[ins].bbox_vertices for ins in scene.instances])
    else:
        all_vertices = numpy.concatenate([
            scene.instances[i].bbox_vertices() for i in instance], axis=1)
    bbox_min = numpy.min(all_vertices[:3], axis=1)
    bbox_max = numpy.max(all_vertices[:3], axis=1)
    offset = bbox_max - bbox_min
    return max(offset)

def blacklist_computation(threshold):
    path = Path("~/.cache/ltron/ldraw/parts").expanduser()
    partlist = list(path.glob("*.dat"))

    blacklist = []
    print('-'*80)
    print('Finding Large Bricks to Blacklist')
    for part in tqdm.tqdm(partlist):
        part = str(part)
        if "30520.dat" in part:
            continue
        bshape = BrickShape(part)
        max_dim = numpy.max(bshape.bbox[1] - bshape.bbox[0])
        if max_dim > threshold:
            blacklist.append(bshape.reference_name)

    blacklist.append("30520.dat")
    return blacklist

def add_brick(limit, cur_mod, comp_list, instance_id, scene):
    instance = scene.instances.instances[instance_id]
    connections = scene.get_all_snap_connections([instance])[str(instance_id)]
    if len(connections) == 0: return False
    for conn in connections:
        target = int(conn[0])
        if conn[0].reference_name in cur_mod:
            continue
        cur_mod.append(target)
        if len(cur_mod) >= limit:
            comp_list.append(cur_mod)
            return True
        else:
            add_brick(limit, cur_mod, comp_list, target, scene)
            return True

def add_brick_box(limit, cur_mod, comp_list, instance_id, scene, used_brick = [], blacklist=[], min_size=100000, max_size=100000, min_brick=3, debug=False, connections={}):
    instance = scene.instances.instances[instance_id]
    #connections = scene.get_all_snap_connections(cur_mod)
    box_map = {}
    
    #for instance_name, connection in connections.items():
    #    if int(instance_name) not in cur_mod:
    #        continue
    for instance_id in cur_mod:
        connection = connections[str(instance_id)]
        for conn in connection:
            temp = copy.deepcopy(cur_mod)
            target = int(conn[0])
            if scene.instances.instances[target].brick_shape.reference_name in blacklist:
                continue
            if target in cur_mod:
                continue
            if target in used_brick:
                continue
            if target in box_map.keys():
                continue
            temp.append(target)
            box_map[target] = compute_boxsize(temp, scene)

    if len(box_map) == 0:
        size = compute_boxsize(cur_mod, scene)
        if min_size <= size <= max_size and len(cur_mod) >= min_brick:
            comp_list.append(cur_mod)
            return True
        #print('bail box')
        return False
    best_target = min(box_map, key=box_map.get)
    cur_mod.append(best_target)
    # used_brick.append(best_target)
    if len(cur_mod) >= limit:
        if compute_boxsize(cur_mod, scene) >= max_size:
            #print('bail big')
            return False
        comp_list.append(cur_mod)
        return True
    else:
        status = add_brick_box(limit, cur_mod, comp_list, best_target, scene, used_brick, connections=connections)
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
    path = Path("~/.cache/ltron/collections/omr/omrtest").expanduser()
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
        if num_instances < limit:
            continue
        
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
def subcomponent_nonoverlap_extraction(src, limit, num_comp, blacklist, min_size=100000, max_size=100000, folder_name = None):
    global_count = 0
    if folder_name is None:
        folder_name = "subcomponents" + str(limit) + "/"
    try:
        os.stat(folder_name)
    except:
        os.mkdir(folder_name)
    path = Path(src).expanduser()
    mpdlist = list(path.rglob('*'))
    stat = {"error" : [], "blacklist" : blacklist, "total_count" : 0, "models" : {}}

    # Iterate through mpd files
    cnt = 0
    iterate = tqdm.tqdm(mpdlist)
    for mpd in iterate:
        description = os.path.basename(str(mpd))
        if len(description) > 30:
            description = description[:27] + '...'
        elif len(description) < 30:
            description = description + ' ' * (30 - len(description))
        iterate.set_description(description)
        # t_start = time.time()
        # t_add_total = 0.
        # t_for_total = 0.
        # num_adds = 0
        # cnt += 1
        # if cnt == 5: break
        # print(mpd)
        mpd = str(mpd)
        # if mpd != "/home/nanami/.cache/ltron/collections/omr/ldraw/30190 - Ferrari 150deg Italia.mpd":
        #     continue
        try:
            mpdfile = LDrawMPDMainFile(mpd)
            scene = BrickScene(track_snaps=True)
            scene.import_ldraw(mpd)
        except:
            stat['error'].append(mpd)
            print("can't load file during extraction {}".format(mpd))
            continue

        num_instances = len(scene.instances.instances)
        if num_instances < limit:
            continue
        # print(num_instances)
        components = []
        used_brick = []

        not_terminate = True
        sad_instance = []
    
        connections = scene.get_all_snap_connections()
        while not_terminate:

            not_terminate = False
            for i in range(num_instances):
                if i in sad_instance:
                    continue
                if scene.instances.instances[i+1].brick_shape.reference_name in blacklist:
                    continue
                #if i in used_brick:
                #    print('used, skipping')
                #    continue
                cur_list = []
                debug = False
                
                # t_add_start = time.time()
                status = add_brick_box(limit, [i+1], cur_list, i+1, scene, used_brick,
                                       min_size=min_size, max_size=max_size, blacklist=blacklist, debug=debug, connections=connections)
                # t_add_end = time.time()
                # num_adds += 1
                # t_add_total += (t_add_end - t_add_start)
                #print('add elapsed: %f'%(t_add_end-t_add_start))
                # if not status:
                #     sad_instance.append(i)
                # not_terminate = not_terminate or status
                
                # t_for_start = time.time()
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
                    not_terminate = False
                    break
        #         t_for_end = time.time()
        #         t_for_total += t_for_end - t_for_start
        #
        # t_end_start = time.time()
        count = 1
        modelname = mpd.split("/")[-1][:-4]
        for comp in components:
            #temp_scene = BrickScene()
            #temp_scene.import_ldraw(mpd)
            #instances = copy.deepcopy(temp_scene.instances.instances)
            #for k, v in instances.items():
            #    if k not in comp:
            #        temp_scene.remove_instance(v)

            #temp_scene.export_ldraw(folder_name + modelname + "_"
            #                                                + str(count) + ".mpd")
            
            _, ext = os.path.splitext(mpd)
            model_file_name = '%s_%i_%i%s'%(modelname, limit, count, ext)
            model_path = os.path.join(folder_name, model_file_name)
            scene.export_ldraw(model_path, instances=comp)
            #scene.export_ldraw(folder_name + modelname + "_" + str(limit) + "_" + str(count) + "." + mpd.split(".")[-1], instances=comp)
            
            count += 1
        # t_end_end = time.time()
        
        stat['models'][modelname] = [num_instances, count - 1]
        # t_end = time.time()
        # print('elapsed: %f'%(t_end-t_start))
        # print('add total: %f'%(t_add_total))
        # print('for total: %f'%(t_for_total))
        # print('end total: %f'%(t_end_end - t_end_start))
        # print('num adds: %i'%(num_adds))

        if global_count == num_comp:
            break

    stat['total_count'] = global_count

    with open(folder_name + "stat_blacklist70_min200_max300_b.json", "w") as f:
        json.dump(stat, f)

def subcomponent_minmax_extraction(src, limit, min_size, max_size, num_comp, blacklist):
    global_count = 0
    folder_name = "subcomponents" + str(limit) + "/"
    try:
        os.stat(folder_name)
    except:
        os.mkdir(folder_name)
    path = Path(src).expanduser()
    mpdlist = path.rglob('*')
    stat = {"error" : [], "blacklist" : blacklist, "total_count" : 0, "models" : {}}

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

        not_terminate = True
        sad_instance = []
        while not_terminate:
            print("l")
            not_terminate = False
            for i in range(num_instances):
                if i in sad_instance:
                    continue
                cur_list = []
                debug = False
                # if mpd == "/home/nanami/.cache/ltron/collections/omr/ldraw/42038-1 - Arctic Truck.mpd":
                #     debug = True
                status = add_brick_box(limit, [], cur_list, i+1, scene, used_brick, blacklist, debug=debug)
                if not status:
                    sad_instance.append(i)
                not_terminate = not_terminate or status
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
                    not_terminate = False
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

            temp_scene.export_ldraw(folder_name + modelname + "_"
                                                            + str(count) + ".mpd")
            count += 1

        stat['models'][modelname] = [num_instances, count - 1]

        if global_count == num_comp:
            break

    stat['total_count'] = global_count

    with open(folder_name + "stat_blacklist.json", "w") as f:
        json.dump(stat, f)

# Render a .mpd file
def render(filepath):
    components = collections.OrderedDict()
    # components['scene'] = SceneComponent(dataset_component=None,
    #                                      initial_scene_path=filepath,
    #                                      track_snaps=True)
    components['episode'] = MaxEpisodeLengthComponent(1000)
    components['camera'] = FixedAzimuthalViewpointComponent(
        components['scene'],
        azimuth=math.radians(-135.),
        elevation=math.radians(-30.),
        aspect_ratio=1,
        distance=(2, 2))
    components['render'] = ColorRenderComponent(512, 512, components['scene'])

    # env = LtronEnv(components)
    obs = env.reset()
    imshow(obs['render'])
    plt.show()
    # print(compute_boxsize(components['scene'].brick_scene.instances.instances.keys(), components['scene'].brick_scene))

def main():
    # blacklist = blacklist_computation(70)
    # # print(blacklist)
    # f = open('subcomponents8/stat_blacklist70_min200_max300.json')
    # blacklist = json.load(f)['blacklist']
    subcomponent_nonoverlap_extraction("omr_clean/whole/", 8, 40000, blacklist=[], min_size=200, max_size=400)
    #render("subcomponents8/6954-1 - Renegade_1.mpd")
    #render("subcomponents8/6954-1 - Renegade_2.mpd")

if __name__ == '__main__' :
    main()

