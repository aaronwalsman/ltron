import os

import ltron.settings as settings

def all_ldraw_parts():
    part_directory = os.path.join(settings.PATHS['ldraw'], 'parts')
    parts = [
        part for part in os.listdir(part_directory)
        if os.path.splitext(part)[-1].lower() == '.dat'
    ]

    return parts
