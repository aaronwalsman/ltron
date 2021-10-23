import json

from ltron.bricks.brick_scene import BrickScene
from ltron.ldraw.documents import LDrawMPDMainFile
from pathlib import Path
from collections import OrderedDict
import random

def traintest_splitter(path, train_ratio=0.8, mode = "raw"):

    if mode == "raw":
        # This file should be within the same directory
        error = []
        with open("theme_map.json", 'r') as f:
            stat = json.load(f)

        theme_count = stat['theme_meta']
        annot = {"train" : [], "test": []}
        theme_train = {}
        theme_cum = {}
        for theme, count in theme_count.items():
            theme_train[theme] = int(count*train_ratio)
            theme_cum[theme] = 0

        path = Path(path).expanduser()
        random.seed(810975)
        modelList = list(path.rglob('*'))
        random.shuffle(modelList)

        print(len(modelList))
        # Iterate through models
        for model in modelList:
            model = str(model)
            model_name = model.split("/")[-1]

            # 0 is Train, 1 is Test
            try:
                cur_theme = stat['model_map'][model_name]
            except:
                error.append(model_name)
                continue
            if theme_cum[cur_theme] >= theme_train[cur_theme]:
                annot['test'].append(model)
            else:
                theme_cum[cur_theme] += 1
                annot['train'].append(model)

        with open("notFoundModel_raw.json", 'w') as f:
            json.dump(error, f)

    else:
        # This file should be within the same directory
        error = []
        with open("omr_raw.json", 'r') as f:
            stat = json.load(f)

        annot = {"train": [], "test": []}
        train = stat['split']['train']

        # Extract true model name
        simp_train = []
        for t in train:
            simp_train.append(t.split("/")[-1])
        path = Path(path).expanduser()
        modelList = list(path.rglob('*'))

        print(len(modelList))
        # Iterate through models
        for model in modelList:
            model = str(model)
            model_name = model.split("/")[-1]

            # Only care about the model name
            checker = model_name.split("@")
            part_we_care = checker[0] + "." + model_name.split(".")[-1]
            # print(part_we_care)
            # print(simp_train[0])
            # exit(0)
            # 0 is Train, 1 is Test
            if part_we_care not in simp_train:
                annot['test'].append(model)
            else:
                annot['train'].append(model)

        with open("notFoundModel_clean.json", 'w') as f:
            json.dump(error, f)

    return annot

def category_splitter(path, mode='raw'):

    if mode == 'raw':
        with open("theme_map.json", 'r') as f:
            stat = json.load(f)

        annot = {}
        for theme, _ in stat['theme_meta'].items():
            annot[theme] = []

        path = Path(path).expanduser()
        modelList = path.rglob('*')

        # Iterate through models
        for model in modelList:
            model = str(model)
            model_name = model.split("/")[-1]

            # 0 is Train, 1 is Test
            try:
                cur_theme = stat['model_map'][model_name]
            except:
                continue

            annot[cur_theme].append(model)

        return annot

    else:
        with open("theme_map.json", 'r') as f:
            stat = json.load(f)

        annot = {}
        for theme, _ in stat['theme_meta'].items():
            annot[theme] = []

        path = Path(path).expanduser()
        modelList = path.rglob('*')

        # Iterate through models
        for model in modelList:
            model = str(model)
            model_name = model.split("/")[-1].split("@")[0] + "." + model.split(".")[-1]

            # 0 is Train, 1 is Test
            try:
                cur_theme = stat['model_map'][model_name]
            except:
                continue

            annot[cur_theme].append(model)

        return annot

def size_splitter(path, size_map):

    annot = {}
    for size, _ in size_map.items():
        annot[size] = []

    path = Path(path).expanduser()
    modelList = path.rglob('*')

    # Iterate through models
    for model in modelList:
        model = str(model)

        try:
            scene = BrickScene(track_snaps=False)
            scene.import_ldraw(model)
        except:
            continue

        for size, bound in size_map.items():
            if len(scene.instances) <= bound:
                annot[size].append(model)
                break

    return annot

def extract_stat(path):
    path = Path(path).expanduser()
    modelList = path.rglob('*')

    # Iterate through models
    for model in modelList:
        model = str(model)
        try:
            cur_model = LDrawMPDMainFile(model)
            scene = BrickScene(track_snaps=True)
            scene.import_ldraw(model)
        except:
            print("Can't open: " + model + " during annotation generation")
            continue

# Any raw model shouldn't contain "@"
# ALl model name shouldn't contain "/"
# The real model name should be extracted by model_name.split("/")[-1].split("@")[0] given its path modle_name
def generate_json(model_path, dest_path, train_ratio=0.8, mode='raw'):

    annot = {"split": {}}
    annot['split'] = {**traintest_splitter(model_path, train_ratio, mode), **annot['split']}
    annot['split'] = {**category_splitter(model_path, mode), **annot['split']}

    # Size map should be an ordered dict ranked with the upperbound increasingly, value is the upperbound
    size_map = {'Pico' : 2, 'Micro' : 8, 'Mini' : 32, "Small" : 128, "Medium" : 512, "large" : float("inf")}
    annot['split'] = {**size_splitter(model_path, size_map), **annot['split']}

    if mode == "raw":
        with open(dest_path + "omr_raw.json", 'w') as f:
            json.dump(annot, f)
    else:
        with open(dest_path + "omr_clean.json", 'w') as f:
            json.dump(annot, f)

    print("Annotation Generation Done")

if __name__ == '__main__':
    generate_json("~/.cache/ltron/collections/omr/ldraw", "")