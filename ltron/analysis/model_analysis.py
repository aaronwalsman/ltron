import json
from ltron.ldraw.documents import LDrawMPDMainFile
from ltron.bricks.brick_scene import BrickScene
from pathlib import Path

def count_gen():
    path = Path("~/.cache/ltron/collections/omr/ldraw").expanduser()
    mpdlist = path.rglob('*mpd') # 1432
    ldrlist = path.rglob('*ldr') # 62
    stat = {"error" : []}

    for mpd in mpdlist:
        mpd = str(mpd)
        try:
            mpdfile = LDrawMPDMainFile(mpd)
            scene = BrickScene()
            scene.import_ldraw(mpd)
        except:
            stat['error'].append(mpd)
            continue
        count = len(scene.instances)
        filename = mpd.split('/')[-1]
        stat[filename] = {'count' : count, 'submodel' : {}}
        internal = mpdfile.internal_files

        for sub in internal:
            ref = sub.reference_name
            scene = BrickScene()
            try:
                scene.import_ldraw(mpd + "#" + ref)
            except:
                stat['error'].append(mpd + "#" + ref)
                continue
            count = len(scene.instances)
            stat[filename]['submodel'][ref] = count

    for ldr in ldrlist:
        ldr = str(ldr)
        scene = BrickScene()
        scene.import_ldraw(ldr)
        count = len(scene.instances)
        filename = ldr.split('/')[-1]
        stat[filename] = {'count': count, 'submodel': {}}

    with open('model_stat_full.json', 'w') as f:
        json.dump(stat, f)

def count_read(filename, subname = None): # assume model_stat.json is already in the same directory
    f = open('model_stat_full.json', 'w')
    stat = json.load(f)
    if subname is None:
        return stat[filename]['count']
    else:
        return stat[filename]['submodel'][subname]

def threshold_count(thre): # return the number of models with at least thre number of bricks
    f = open('model_stat_full.json')
    stat = json.load(f)
    model_list = []

    for k in stat.keys():
        if k == "error": continue
        count = stat[k]['count']
        if count < thre:
            continue
        model_list.append(k)

        submodel = stat[k]['submodel']
        for sub in submodel:
            if stat[k]['submodel'][sub] >= thre:
                model_list.append(k + "#" + sub)

    print("Number of models meet the threshold: ", len(model_list))
    return model_list, len(model_list)