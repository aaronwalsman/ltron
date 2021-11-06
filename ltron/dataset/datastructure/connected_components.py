import time
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

# remove all components that contains less than or equal to the threshold
def partition_omr(directory, outdir=None, remove_thre = 0):
    path = Path(directory).expanduser()
    modelList = list(path.rglob('*'))

    # Iterate through mpd files
    cnt = 0
    for model in tqdm.tqdm(modelList):
        model = str(model)
        try:
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
        idx = 1
        for _, comp in components.items():
            if len(comp) <= remove_thre: continue
            scene.export_ldraw(folder_name + modelname + "@" + str(idx) + "." + model.split(".")[-1], instances=comp)
            idx += 1

def main():
    direc = "~/.cache/ltron/collections/omr/ldraw"
    partition_omr(direc)

if __name__ == '__main__' :
    main()
