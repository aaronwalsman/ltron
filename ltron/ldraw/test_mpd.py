#!/usr/bin/env python
import os
import mpd

external_parts = os.listdir('../ldraw/parts')
with open('../OMD/8661-1 - Carbon Star.mpd') as f:
    flattened_parts = mpd.parts_from_mpd(f, external_parts)
    print(flattened_parts)
    print(len(flattened_parts))
