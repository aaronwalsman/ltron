import copy
import numpy

def parts_from_mpd(mpd_data, external_parts):
    
    # if mpd_data is a file-like object, read all the data from it
    try:
        mpd_data = mpd_data.read()
    except AttributeError:
        pass
    
    mpd_lines = mpd_data.splitlines()
    
    # find all nested files
    nested_files = {}
    current_file = {}
    for i, line in enumerate(mpd_lines):
        if line[:7] == '0 FILE ':
            file_name = line[7:].strip()
            current_file = {'name':file_name, 'lines':[]}
            if not len(nested_files):
                main_file = current_file
            nested_files[file_name] = current_file
        
        if not len(current_file):
            continue    
        current_file['lines'].append(line)
    
    if not len(current_file):
        print("Warning, no file exist")
        return []

    complete = True
    # find the parts in all nested files
    for nested_name, nested_file in nested_files.items():
        nested_file['references'] = []
        nested_file['parts'] = []
        for line in nested_file['lines']:
            if line[:2] == '1 ':
                (color,
                 x, y, z,
                 xx, xy, xz,
                 yx, yy, yz,
                 zx, zy, zz,
                 file_name) = line[2:].split(None, 13)
                color = int(color,0)
                x, y, z = float(x), float(y), float(z)
                xx, xy, xz = float(xx), float(xy), float(xz)
                yx, yy, yz = float(yx), float(yy), float(yz)
                zx, zy, zz = float(zx), float(zy), float(zz)
                transform = numpy.array([
                        [xx, xy, xz, x],
                        [yx, yy, yz, y],
                        [zx, zy, zz, z],
                        [ 0,  0,  0, 1]]).tolist()
                file_name = file_name.strip()
                reference = {
                        'file_name' : file_name,
                        'color' : color,
                        'transform' : transform
                }
                if file_name in nested_files:
                    nested_file['references'].append(reference)
 
                else:
                    if file_name in external_parts:
                        nested_file['parts'].append(reference)
                    else:
                        complete = False
                        print('Warning!!! Missing part: %s'%file_name)

    def get_all_parts(nested_file):
        all_parts = []
        for reference in nested_file['references']:
            referenced_file = nested_files[reference['file_name']]
            referenced_parts = get_all_parts(referenced_file)
            for referenced_part in referenced_parts:
                flattened_part = copy.deepcopy(referenced_part)
                flattened_part['transform'] = numpy.dot(
                        reference['transform'],
                        referenced_part['transform']).tolist()
                all_parts.append(flattened_part)
        
        all_parts.extend(nested_file['parts'])
        return all_parts
 
    all_parts = get_all_parts(main_file)
    all_parts.append({'complete':complete})
    return all_parts
    
