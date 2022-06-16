import time
import os
import json
from ltron.ldraw.documents import LDrawMPDMainFile
from ltron.bricks.brick_scene import BrickScene
from pathlib import Path
import queue
import tqdm

def BFS(instance, connections, visited):
    cum_list = [instance]
    visited[instance-1] = True
    que = queue.Queue()
    que.put(instance)
    while not que.empty():
        connection = connections[int(que.get())]
        for conn in connection:
            target = int(conn[1].brick_instance)
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

# remove all components that contains less than or equal to the threshold
def partition_omr(src_directory, dest_directory=None, remove_threshold=0):
    src_directory = Path(src_directory).expanduser()
    model_list = list(src_directory.rglob('*'))
    #model_list = [os.path.join(src_directory, '7657-1 - AT-ST.mpd')]
    #model_list = [os.path.join(src_directory, '6832-1 - Super Nova II.mpd')]
    #model_list = [os.path.join(src_directory, '7140-1 - X-wing Fighter.mpd')]

    # Iterate through mpd files
    cnt = 0
    scene = BrickScene(track_snaps=True)
    for model in tqdm.tqdm(model_list):
        model = str(model)
        print(model)
        scene.clear_instances()
        try:
            scene.import_ldraw(model)
        except:
            print("Can't open: " + model + " during connected components partition")
            continue

        components = partition(scene)
        if dest_directory is None:
            folder_name = "conn_comps/"
        else:
            folder_name = dest_directory
        #modelname = model.split("/")[-1][:-4]
        modelname, ext = os.path.splitext(os.path.split(model)[-1])
        idx = 1
        for _, comp in components.items():
            if len(comp) <= remove_threshold:
                continue
            out_path = os.path.join(
                folder_name, '%s__%i%s'%(modelname, idx, ext))
            scene.export_ldraw(out_path, instances=comp)
            #scene.export_ldraw(
            #    folder_name + modelname + "@" + str(idx) +
            #"." + model.split(".")[-1], instances=comp)
            idx += 1

def main():
    direc = "~/.cache/ltron/collections/omr/ldraw"
    partition_omr(direc)

if __name__ == '__main__' :
    main()
