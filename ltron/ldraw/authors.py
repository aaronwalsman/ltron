#!/usr/bin/env python
import tqdm

from ltron.ldraw.paths import LDRAW_FILES, SHADOW_FILES
from ltron.ldraw.documents import LDrawDocument
from ltron.ldraw.commands import LDrawAuthorComment

all_authors = {}

for part in tqdm.tqdm(SHADOW_FILES.values()):
    try:
        doc = LDrawDocument.parse_document(part)
    except ValueError:
        print('skipping: %s'%part)
        continue
    for command in doc.commands:
        if isinstance(command, LDrawAuthorComment):
            if command.author not in all_authors:
                all_authors[command.author] = []
            all_authors[command.author].append(part)
            break
    
    else:
        print('No author found for: %s'%part)
    
for author, parts in sorted(
        all_authors.items(), key=lambda x : len(x[1]), reverse=True):
    print(author.replace('_', '\\_') + ',')
