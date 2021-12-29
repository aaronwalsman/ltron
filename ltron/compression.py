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
    
    #previous_frame = numpy.zeros((n, th, tw, *c), dtype=frames[0].dtype)
    #previous_frame[:,:,:] = background
    previous_frame = background
    for i, frame in enumerate(frames):
        match = frame == previous_frame
        match = match.reshape(n, -1)
        modified_tiles = ~numpy.all(match, axis=-1)
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

def batch_deduplicate_tiled_seqs_old(seqs, *args, s_start = 0, **kwargs):
    
    #tile_seqs, tile_coords = zip(*[
    #    deduplicate_tiled_seq(seq, *args, **kwargs) for seq in frame_seqs
    #])
    b = seqs.shape[1]
    tile_seqs, tile_coords = zip(*[
        deduplicate_tiled_seq(seqs[:,i], *args, **kwargs) for i in range(b)
    ])
    
    max_len = max(len(seq) for seq in tile_seqs)
    padded_seqs = []
    padded_coords = []
    padding_mask = numpy.zeros(
        #(max_len, len(frame_seqs)),
        (max_len, b),
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

def batch_deduplicate_from_masks(
    frames,
    masks,
    t,
    pad,
):
    s, b, h, w, c = frames.shape
    sm, bm, hh, ww = masks.shape
    assert sm == s
    assert bm == b
    tile_height = h // hh
    tile_width = w // ww
    
    seq_tiles = frames.reshape(s, b, hh, tile_height, ww, tile_width, c)
    seq_tiles = seq_tiles.transpose(1, 0, 2, 4, 3, 5, 6)
    seq_tiles = seq_tiles.reshape(b, s, hh, ww, -1)
    
    nonstatic_tiles = masks.transpose(1, 0, 2, 3)
    
    # below copied from batch_deduplicate_tiles_seqs below
    
    # get indices of changing tiles
    b_coord, s_coord, h_coord, w_coord = numpy.where(nonstatic_tiles)
    max_lengths = pad[b_coord]
    nonpadded_indices = s_coord < max_lengths
    b_coord = b_coord[nonpadded_indices]
    s_coord = s_coord[nonpadded_indices]
    h_coord = h_coord[nonpadded_indices]
    w_coord = w_coord[nonpadded_indices]
    
    # extract the changing tiles
    compressed_len = len(b_coord)
    compressed_tiles = seq_tiles[b_coord, s_coord, h_coord, w_coord]
    compressed_tiles = compressed_tiles.reshape(
        compressed_len, tile_height, tile_width, c)
    
    # compute the coordinates for the tiles on the new padded grid
    # this time the padding is not based on the original frame sequences,
    # but on the lengths of the newly computed tile sequences
    # the reason why the batch dimension must come first above, is so that
    # b_coord will align properly with i_coord below
    batch_pad = numpy.bincount(b_coord, minlength=b)
    max_len = numpy.max(batch_pad)
    i_coord = numpy.concatenate([numpy.arange(cc) for cc in batch_pad])
    
    # place the tiles in the padded grid and swap the batch and time axes
    batch_padded_tiles = numpy.zeros(
        (b, max_len, tile_height, tile_width, c), dtype=compressed_tiles.dtype)
    batch_padded_tiles[b_coord, i_coord] = compressed_tiles
    batch_padded_tiles = batch_padded_tiles.transpose(1,0,2,3,4)
    
    # place the coordinates in a padded grid and swap the batch and time axes
    batch_padded_coords = numpy.zeros((b, max_len, 3), dtype=s_coord.dtype)
    t_coord = t[s_coord, b_coord]
    shw_coord = numpy.stack((t_coord, h_coord, w_coord), axis=-1)
    batch_padded_coords[b_coord, i_coord] = shw_coord
    #if s_start is not None:
    #    batch_padded_coords[:,:,0] += s_start.reshape(b, 1)
    batch_padded_coords = batch_padded_coords.transpose(1,0,2)
    
    return (
        batch_padded_tiles,
        batch_padded_coords,
        batch_pad,
    )
    

def batch_deduplicate_tiled_seqs(
    seqs,
    pad,
    tile_width,
    tile_height,
    background=0,
    s_start=None,
):
    s, b, h, w, c = seqs.shape
    assert h % tile_height == 0
    assert w % tile_width == 0
    hh = h // tile_height
    ww = w // tile_width
    
    # reshape to b x s x hh x ww x (tile_height*tile_width*c)
    # batch must come first because this makes it possible to map
    # the extracted tiles onto the compressed batch tensor later
    seq_tiles = seqs.reshape(s, b, hh, tile_height, ww, tile_width, c)
    seq_tiles = seq_tiles.transpose(1, 0, 2, 4, 3, 5, 6)
    seq_tiles = seq_tiles.reshape(b, s, hh, ww, -1)
    
    # make the background
    try:
        background = background.reshape(
            b, 1, hh, tile_height, ww, tile_width, c)
        background = background.transpose(0, 1, 2, 4, 3, 5, 6)
        background = background.reshape(b, 1, hh, ww, -1)
    except AttributeError:
        background = background * numpy.ones(
            (b, 1, hh, ww, tile_height*tile_width*c), dtype=seq_tiles.dtype)
    
    # compute tiles that change
    prev_tiles = numpy.concatenate((background, seq_tiles[:,:-1]), axis=1)
    nonstatic_tiles = numpy.any(seq_tiles != prev_tiles, axis=-1)
    
    # get indices of changing tiles
    b_coord, s_coord, h_coord, w_coord = numpy.where(nonstatic_tiles)
    max_lengths = pad[b_coord]
    nonpadded_indices = s_coord < max_lengths
    b_coord = b_coord[nonpadded_indices]
    s_coord = s_coord[nonpadded_indices]
    h_coord = h_coord[nonpadded_indices]
    w_coord = w_coord[nonpadded_indices]
    
    # extract the changing tiles
    compressed_len = len(b_coord)
    compressed_tiles = seq_tiles[b_coord, s_coord, h_coord, w_coord]
    compressed_tiles = compressed_tiles.reshape(
        compressed_len, tile_height, tile_width, c)
    
    # compute the coordinates for the tiles on the new padded grid
    # this time the padding is not based on the original frame sequences,
    # but on the lengths of the newly computed tile sequences
    # the reason why the batch dimension must come first above, is so that
    # b_coord will align properly with t_coord below
    batch_pad = numpy.bincount(b_coord, minlength=b)
    max_len = numpy.max(batch_pad)
    t_coord = numpy.concatenate([numpy.arange(cc) for cc in batch_pad])
    
    # place the tiles in the padded grid and swap the batch and time axes
    batch_padded_tiles = numpy.zeros(
        (b, max_len, tile_height, tile_width, c), dtype=compressed_tiles.dtype)
    batch_padded_tiles[b_coord, t_coord] = compressed_tiles
    batch_padded_tiles = batch_padded_tiles.transpose(1,0,2,3,4)
    
    # place the coordinates in a padded grid and swap the batch and time axes
    batch_padded_coords = numpy.zeros((b, max_len, 3), dtype=s_coord.dtype)
    shw_coord = numpy.stack((s_coord, h_coord, w_coord), axis=-1)
    batch_padded_coords[b_coord, t_coord] = shw_coord
    if s_start is not None:
        batch_padded_coords[:,:,0] += s_start.reshape(b, 1)
    batch_padded_coords = batch_padded_coords.transpose(1,0,2)
    
    return (
        batch_padded_tiles,
        batch_padded_coords,
        batch_pad,
    )
