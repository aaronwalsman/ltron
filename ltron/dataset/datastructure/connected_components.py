import time
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
from ltron.bricks.brick_type import BrickType
import copy
import collections
import math
import os
import numpy
import random
import queue

def BFS(instance, connections, visited):
    cum_list = [instance]
    visited[instance-1] = True
    que = queue.Queue()
    que.put(instance)
    while not que.empty():
        connection = connections[str(que.get())]
        for conn in connection:
            target = int(conn[0])
            if not visited[target-1]:
                que.put(target)
                visited[target-1] = True
                cum_list.append(target)
    return cum_list

def partition(scene):
    components = {}
    connections = scene.get_all_snap_connections()
    instance_num = len(scene.instances.instances)
    visited = [False] * instance_num

    comp_num = 1
    for i in range(instance_num):
        if visited[i]:
            continue
        else:
            cur_list = BFS(i+1, connections, visited)
            components[comp_num] = cur_list
            comp_num += 1

    return components

def partition_omr(directory, outdir=None):
    path = Path(directory).expanduser()
    modelList = path.rglob('*')

    # Iterate through mpd files
    cnt = 0
    for model in modelList:
        model = str(model)
        try:
            cur_model = LDrawMPDMainFile(model)
            scene = BrickScene(track_snaps=True)
            scene.import_ldraw(model)
        except:
            print("Can't open: " + model + " during connected components partition")
            continue

        components = partition(scene)
        if outdir is None:
            folder_name = "conn_comps/"
        else:
            folder_name = outdir
        modelname = model.split("/")[-1][:-4]
        for idx, comp in components.items():
            # temp_scene = BrickScene()
            # temp_scene.import_ldraw(mpd)
            # instances = copy.deepcopy(temp_scene.instances.instances)
            # for k, v in instances.items():
            #    if k not in comp:
            #        temp_scene.remove_instance(v)

            # temp_scene.export_ldraw(folder_name + modelname + "_"
            #                                                + str(count) + ".mpd")

            scene.export_ldraw(folder_name + modelname + "@" + str(idx) + "." + model.split(".")[-1], instances=comp)

def main():
    direc = "~/.cache/ltron/collections/omr/ldraw"
    partition_omr(direc)

if __name__ == '__main__' :
    main()
