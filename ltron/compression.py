from functools import reduce

import numpy

def tile_frame(frame, tile_height, tile_width):
    h, w, *c = frame.shape
    assert h % tile_height == 0
    assert w % tile_width == 0
    hh = h // tile_height
    ww = w // tile_width
    
    frame = frame.reshape(hh, tile_height, ww, tile_width, *c)
    frame = numpy.moveaxis(frame, 2, 1)
    frame = frame.reshape(hh*ww, tile_height, tile_width, *c)
    
    return frame

def deduplicate_tiled_seq(frames, tile_height, tile_width, background=0):
    frames = [tile_frame(frame, tile_height, tile_width) for frame in frames]
    n, th, tw, *c = frames[0].shape
    
    seq = []
    coords = []
    
    previous_frame = numpy.zeros((n, th, tw, *c), dtype=frames[0].dtype)
    previous_frame[:,:,:] = background
    for i, frame in enumerate(frames):
        match = frame == previous_frame
        match = match.reshape(n, -1)
        modified_tiles = ~numpy.all(match, axis=-1)
        modified_tiles = 
        modified_coords = numpy.where(modified_tiles)[0]
        nn = modified_coords.shape[0]
        modified_coords = numpy.stack(
            (numpy.ones(nn, dtype=numpy.long)*i, modified_coords),
            axis=-1
        )
        seq.append(frame[modified_tiles])
        coords.append(modified_coords)
        
        previous_frame = frame
    
    seq = numpy.concatenate(seq, axis=0)
    coords = numpy.concatenate(coords, axis=0)
    
    return seq, coords

def batch_deduplicate_tiled_seqs(frame_seqs, *args, **kwargs):
    
    tile_seqs, tile_coords = zip(*[
        deduplicate_tiled_seq(seq, *args, **kwargs) for seq in frame_seqs
    ])
    
    max_len = max(len(seq) for seq in tile_seqs)
    padded_seqs = []
    padded_coords = []
    padding_mask = numpy.zeros(
        (max_len, len(frame_seqs)),
        dtype=numpy.bool
    )
    for i, (seq, coords) in enumerate(zip(tile_seqs, tile_coords)):
        seq_len = len(seq)
        pad_len = max_len - seq_len
        
        sz = numpy.zeros(
            (pad_len, *seq.shape[1:]), dtype=seq.dtype)
        seq = numpy.concatenate((seq, sz), axis=0)
        padded_seqs.append(seq)
        
        cz = numpy.zeros(
            (pad_len, 2), dtype=coords.dtype)
        coords = numpy.concatenate((coords, cz), axis=0)
        padded_coords.append(coords)
        
        padding_mask[:seq_len, i] = True
    
    padded_seqs = numpy.stack(padded_seqs, axis=1)
    padded_coords = numpy.stack(padded_coords, axis=1)
    
    return padded_seqs, padded_coords, padding_mask
