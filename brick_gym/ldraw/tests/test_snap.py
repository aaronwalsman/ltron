#!/usr/bin/env python
import os

import brick_gym.config as config
import brick_gym.ldraw.documents as documents
import brick_gym.ldraw.snap as snap

doc1 = documents.LDrawDocument.parse_document(
        os.path.join(config.paths['omr'], '8661-1 - Carbon Star.mpd'))

parts = doc1.get_parts()
print('num carbon star parts: %i'%len(parts))

doc2 = parts[0][0]
snap_points2 = snap.snap_points_from_part_document(doc2)
print('num snap points for part %s: %i'%(doc2.clean_name, len(snap_points2)))

doc3 = documents.LDrawDocument.parse_document('3003.dat')
snap_points3 = snap.snap_points_from_part_document(doc3)
print('num snap points for part %s: %i'%(doc3.clean_name, len(snap_points3)))

doc4 = documents.LDrawDocument.parse_document(
        os.path.join(config.paths['ldraw'], 'parts', '3003.dat'))
snap_points4 = snap.snap_points_from_part_document(doc4)
print('num snap points for part %s: %i'%(doc4.clean_name, len(snap_points4)))
