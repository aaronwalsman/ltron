import numpy

import brick_gym.config as config
ldraw_directory = config.paths['ldraw']
ldraw_parts_directories = [
        os.path.join(ldraw_directory, parts_directory
        for parts_directory in 'parts', 'p', 'models']

def matrix_ldraw_to_numpy(elements):
    assert len(elements) == 12
    (x, y, z,
     xx, xy, xz,
     yx, yy, yz,
     zx, zy, zz) = elements
    x, y, z = float(x), float(y), float(z)
    xx, xy, xz = float(xx), float(xy), float(xz)
    yx, yy, yz = float(yx), float(yy), float(yz)
    zx, zy, zz = float(zx), float(zy), float(zz)
    return numpy.array([
            [xx, xy, xz, x],
            [yx, yy, yz, y],
            [zx, zy, zz, z],
            [ 0,  0,  0, 1]])

def vectors_ldraw_to_numpy(elements):
    assert len(elements)%3 == 0
    return numpy.array([
            elements[0::3],
            elements[1::3],
            elements[2::3],
            [1] * len(elements) // 3])

def vectors_numpy_to_ldraw(vectors):
    num_elements = 3 * vectors.shape[1]
    vectors = vectors[:3,:]
    vectors = tuple(vectors.T.reshape(-1).tolist())
    return ('_'.join(['%s']*num_elements))%vectors

class LDRAWReferenceNotFoundError(Exception):
    pass

def resolve_ldraw_part_filepath(file_name):
    if os.path.exists(file_name):
        return file_name
    
    for parts_directory in ldraw_parts_directories:
        candidate = os.path.join(parts_directory, file_name)
        if os.path.exists(candidate):
            return candidate
        #part_files = os.listdir(parts_directory)
        #if file_name in part_files:
        #    return os.path.join(parts_directory, file_name)
    
    raise LDRAWReferenceNotFoundError('Could not find part %s'%file_name)

def resolve_ldraw_references(
        ldraw_text,
        reference_color = default_brick_color,
        reference_complement_color = default_brick_complement_color,
        reference_transform = None):
    
    lines = ldraw_text.splitlines()
    resolved_lines = []
    for line in lines:
        line_contents = line.split(None, 1)
        if len(line_contents) != 2:
            resolved_lines.append(line)
            continue
        
        command, arguments = line_contents
        if command == '1':
            # file reference
            # extract color and transform, then recurse
            color, *matrix_elements, file_name = arguments.split(None, 13)
            color = ldraw_color_to_hex(color)
            complement_color = ldraw_color_to_complement_hex(color)
            transform = ldraw_to_numpy_matrix(matrix_elements)
            file_path = resolve_ldraw_part_filepath(file_name)
            with open(file_path, 'r') as f:
                sub_ldraw_text = f.read()
            subfile_contents = resolve_ldraw_references(
                    sub_ldraw_text,
                    reference_color = color,
                    reference_complement_color = complement_color,
                    reference_transform = transform)
            resolved_lines.append(subfile_contents)
        
        elif command in ('2', '3', '4', '5'):
            # update vertices and color based on reference transform and color
            color, *vector_elements = arguments.split()
            if color == '16':
                color = reference_color
            elif color == '24':
                color = reference_complement_color
            
            if reference_transform is not None:
                vectors = numpy.dot(
                        reference_transform,
                        ldraw_to_numpy_vectors(vector_elements))
                vectors = numpy_to_ldraw_vectors(vectors)
            
            resolved_lines.append(' '.join((command, color, vectors)))
        
        else:
            resolved_lines.append(line)
