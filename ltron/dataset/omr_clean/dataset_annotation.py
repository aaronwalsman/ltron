import json

import tqdm

from ltron.bricks.brick_scene import BrickScene
from ltron.ldraw.documents import LDrawMPDMainFile
from pathlib import Path
from collections import OrderedDict
import random
import os
import glob
from ltron.dataset.submodel_extraction import compute_boxsize
import ltron.settings as settings

def traintest_splitter(path, train_ratio=0.8, mode = "raw"):

    if mode == "raw":
        # This file should be within the same directory
        error = []
        theme_path = os.path.join(
            settings.collections['omr'], 'theme_map.json')
        with open(theme_path, 'r') as f:
            stat = json.load(f)

        theme_count = stat['theme_meta']
        annot = {"train" : {'mpd' : []}, "test": {'mpd' : []}}
        theme_train = {}
        theme_cum = {}
        print('-')
        print('Generating Theme Train Counts')
        for theme, count in tqdm.tqdm(theme_count.items()):
            theme_train[theme] = int(count*train_ratio)
            theme_cum[theme] = 0

        path = Path(path).expanduser()
        random.seed(810975)
        modelList = list(path.rglob('*'))
        random.shuffle(modelList)

        print(len(modelList))
        # Iterate through models
        print('-'*80)
        print('Splitting %s train/test %.02f/%.02f'%(
            mode, train_ratio, 1. - train_ratio))
        iterate = tqdm.tqdm(modelList)
        for model in iterate:
            model = str(model)
            model_name = model.split("/")[-1]

            # 0 is Train, 1 is Test
            try:
                cur_theme = stat['model_map'][model_name]
            except:
                error.append(model_name)
                continue
            if theme_cum[cur_theme] >= theme_train[cur_theme]:
                path_key = "{omr_raw}/"
                if mode == "clean":
                    path_key = "{omr_clean}/"
                model = model.split("/")[-2] + "/" + model.split("/")[-1]
                annot['test']['mpd'].append(path_key + model)
            else:
                theme_cum[cur_theme] += 1
                path_key = "{omr_raw}/"
                if mode == "clean":
                    path_key = "{omr_clean}/"
                model = model.split("/")[-2] + "/" + model.split("/")[-1]
                annot['train']['mpd'].append(path_key + model)

        with open("notFoundModel_raw.json", 'w') as f:
            json.dump(error, f)

    else:
        # This file should be within the same directory
        error = []
        with open("omr_raw.json", 'r') as f:
            stat = json.load(f)

        annot = {"train": {'mpd' : []}, "test": {'mpd' : []}}
        train = stat['splits']['train']['mpd']

        # Extract true model name
        simp_train = []
        print('-'*80)
        print('Getting Train Files')
        for t in tqdm.tqdm(train):
            simp_train.append(t.split("/")[-1])
        path = Path(path).expanduser()
        modelList = list(path.rglob('*'))
        
        print('-'*80)
        print('Allocating Into Train/Test')
        # Iterate through models
        for model in tqdm.tqdm(modelList):
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
                path_key = "{omr_raw}/"
                if mode == "clean":
                    path_key = "{omr_clean}/"
                model = model.split("/")[-2] + "/" + model.split("/")[-1]
                annot['test']['mpd'].append(path_key + model)
            else:
                path_key = "{omr_raw}/"
                if mode == "clean":
                    path_key = "{omr_clean}/"
                model = model.split("/")[-2] + "/" + model.split("/")[-1]
                annot['train']['mpd'].append(path_key + model)

        with open("notFoundModel_clean.json", 'w') as f:
            json.dump(error, f)

    return annot

def category_splitter(path, mode='raw'):

    if mode == 'raw':
        theme_path = os.path.join(
            settings.collections['omr'], 'theme_map.json')
        with open(theme_path, 'r') as f:
            stat = json.load(f)

        annot = {}
        for theme, _ in stat['theme_meta'].items():
            annot[theme.lower()] = {}
            annot[theme.lower()]['mpd'] = []

        path = Path(path).expanduser()
        modelList = list(path.rglob('*'))

        # Iterate through models
        print('-'*80)
        print('Splitting Into Themes')
        for model in tqdm.tqdm(modelList):
            model = str(model)
            model_name = model.split("/")[-1]

            # 0 is Train, 1 is Test
            try:
                cur_theme = stat['model_map'][model_name]
            except:
                continue

            path_key = "{omr_raw}/"
            if mode == "clean":
                path_key = "{omr_clean}/"
            model = model.split("/")[-2] + "/" + model.split("/")[-1]
            annot[cur_theme.lower()]['mpd'].append(path_key + model)

        return annot

    else:
        theme_path = os.path.join(
            settings.collections['omr'], 'theme_map.json')
        with open(theme_path, 'r') as f:
            stat = json.load(f)

        annot = {}
        for theme, _ in stat['theme_meta'].items():
            annot[theme.lower()] = {}
            annot[theme.lower()]['mpd'] = []

        path = Path(path).expanduser()
        modelList = list(path.rglob('*'))

        # Iterate through models
        print('-'*80)
        print('Splitting Into Themes')
        for model in tqdm.tqdm(modelList):
            model = str(model)
            model_name = (
                model.split("/")[-1].split("@")[0] + "." + model.split(".")[-1])

            # 0 is Train, 1 is Test
            try:
                cur_theme = stat['model_map'][model_name]
            except:
                continue

            path_key = "{omr_raw}/"
            if mode == "clean":
                path_key = "{omr_clean}/"
            model = model.split("/")[-2] + "/" + model.split("/")[-1]
            annot[cur_theme.lower()]['mpd'].append(path_key + model)

        return annot

def size_splitter(path, size_map, mode="raw"):

    annot = {}
    for size, _ in size_map.items():
        annot[size.lower()] = {}
        annot[size.lower()]['mpd'] = []

    path = Path(path).expanduser()
    modelList = list(path.rglob('*'))

    # Iterate through models
    print('-'*80)
    print('Splitting Into Size Categories')
    for model in tqdm.tqdm(modelList):
        model = str(model)

        try:
            scene = BrickScene(track_snaps=False)
            scene.import_ldraw(model)
        except:
            continue

        modelSize = compute_boxsize(scene.instances, scene, complete = False);
        for size, bound in size_map.items():
            if size.lower() != 'medium' or size.lower() != 'large':
                if modelSize >= 400:
                    continue
            elif size.lower() == 'medium':
                    if modelSize >= 800:
                        continue
            if len(scene.instances) <= bound:
                path_key = "{omr_raw}/"
                if mode == "clean":
                    path_key = "{omr_clean}/"
                model = model.split("/")[-2] + "/" + model.split("/")[-1]
                annot[size.lower()]['mpd'].append(path_key + model)
                break

    return annot


def build_metadata(path_root):
    metadata = {}
    mpds = list(glob.glob(os.path.join(path_root, '*')))

    max_instances_per_scene = 0
    max_edges_per_scene = 0
    all_brick_names = set()
    all_color_names = set()
    print('-'*80)
    print('Building Metadata')
    for mpd in tqdm.tqdm(mpds):
        scene = BrickScene(track_snaps=True)
        scene.import_ldraw(mpd)
        brick_names = set(scene.shape_library.keys())
        all_brick_names |= brick_names
        color_names = set(scene.color_library.keys())
        all_color_names |= color_names

        num_instances = len(scene.instances)
        max_instances_per_scene = max(num_instances, max_instances_per_scene)

        edges = scene.get_assembly_edges(unidirectional=False)
        num_edges = edges.shape[1]
        max_edges_per_scene = max(num_edges, max_edges_per_scene)

    metadata['max_instances_per_scene'] = max_instances_per_scene
    metadata['max_edges_per_scene'] = max_edges_per_scene
    metadata['shape_ids'] = {
        brick_name: i
        for i, brick_name in enumerate(all_brick_names, start=1)
    }
    metadata['color_ids'] = {
        color_name: i
        for i, color_name in enumerate(all_color_names, start=0)
    }

    return metadata

def extract_stat(path):
    path = Path(path).expanduser()
    modelList = list(path.rglob('*'))

    # Iterate through models
    print('-'*80)
    print('Extracting Stats')
    for model in tqdm.tqdm(modelList):
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
# The real model name should be extracted by model_name.split("/")[-1].split("@")[0] given its path model_name
def generate_json(model_path, dest_path, train_ratio=0.8, mode='raw'):

    annot = {"splits": {}}
    annot['splits'] = {**traintest_splitter(model_path, train_ratio, mode), **annot['splits']}
    annot['splits'] = {**category_splitter(model_path, mode), **annot['splits']}
    annot = {**build_metadata(model_path), **annot}

    # Size map should be an ordered dict ranked with the upperbound increasingly, value is the upperbound
    size_map = {'pico' :
                    2, 'micro' : 8, 'mini' : 32, "small" : 128, "medium" : 512, "large" : float("inf")}
    annot['splits'] = {**size_splitter(model_path, size_map), **annot['splits']}

    if mode == "raw":
        with open(dest_path + "omr_raw.json", 'w') as f:
            json.dump(annot, f, indent=2)
    else:
        with open(dest_path + "omr_clean.json", 'w') as f:
            json.dump(annot, f, indent=2)

    print("Annotation Generation Done")

if __name__ == '__main__':
    # generate_json("~/.cache/ltron/collections/omr/ldraw", "")
    generate_json("ldraw", "temp_json/")
