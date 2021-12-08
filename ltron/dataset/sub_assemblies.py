#!/usr/bin/env python
import os

import tqdm

import ltron.settings as settings
from ltron.bricks.brick_scene import BrickScene
from ltron.ldraw.documents import LDrawDocument, LDrawMPDMainFile
from ltron.ldraw.commands import LDrawImportCommand
#from ltron.ldraw.paths import LDRAW_FILES
from ltron.ldraw.parts import LDRAW_PATHS

omr_directory = os.path.join(settings.paths['omr'], 'ldraw')
model_files = os.listdir(omr_directory)

min_parts = 1

good_internal_files = 0

for model_file in tqdm.tqdm(model_files):
    model_path = os.path.join(omr_directory, model_file)
    try:
        document = LDrawDocument.parse_document(model_path)
    except:
        print('bad document: %s'%model_file)
        continue
    if isinstance(document, LDrawMPDMainFile):
        for internal_file in document.internal_files:
            part_references = 0
            for command in internal_file.commands:
                if isinstance(command, LDrawImportCommand):
                    if command.reference_name in LDRAW_PATHS:
                        part_references += 1
                        if part_references >= min_parts:
                            good_internal_files += 1
                            break

print(good_internal_files)
