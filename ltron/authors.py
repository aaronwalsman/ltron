import tqdm

from ltron.ldraw.parts import LDRAW_PATHS, SHADOW_PATHS
from ltron.ldraw.documents import LDrawDocument
from ltron.ldraw.commands import LDrawAuthorComment

def get_all_ldraw_authorship():
    library_authors = {}
    for library_name, part_files in (
        ('shadow', SHADOW_PATHS.keys()),
        ('ldraw', LDRAW_PATHS.keys()),
    ):
        library_authors[library_name] = get_ldraw_authorship(part_files)
    
    return library_authors

def get_ldraw_authorship(part_files):
    all_authors = {'UNKNOWN':[]}
    
    for part in tqdm.tqdm(part_files):
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
            all_authors['UNKNOWN'].append(part)
    
    return all_authors
