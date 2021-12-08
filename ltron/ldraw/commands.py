import re

import numpy
from ltron.ldraw.exceptions import LDrawException
#import ltron.ldraw.paths as ldraw_paths
from ltron.ldraw.parts import get_reference_name

class InvalidLDrawCommand(LDrawException):
    pass

class BadMatrixException(LDrawException):
    pass

class BadVerticesException(LDrawException):
    pass

def filter_floats(elements):
    floats = []
    for element in elements:
        try:
            floats.append(float(element))
        except ValueError:
            pass
        
    return floats

def matrix_ldraw_to_numpy(elements):
    # it's stupid to have to filter these, but there are some files in the OMR
    # that have non-float garbage (e.g. //) at the end of the line
    elements = filter_floats(elements)
    if len(elements) != 12:
        raise BadMatrixException('ldraw matrix must have 12 elements')
    elements = [float(element) for element in elements]
    (x, y, z,
     xx, xy, xz,
     yx, yy, yz, 
     zx, zy, zz) = elements
    return numpy.array([
            [xx, xy, xz, x],
            [yx, yy, yz, y],
            [zx, zy, zz, z],
            [ 0,  0,  0, 1]])

def matrix_ldcad_to_numpy(flags):
    if 'pos' in flags:
        x, y, z = [float(xyz) for xyz in flags['pos'].split()]
    else:
        x, y, z = 0., 0., 0.
    if 'ori' in flags:
        xx, xy, xz, yx, yy, yz, zx, zy, zz = [
                float(xyz) for xyz in flags['ori'].split()]
    else:
        xx, xy, xz, yx, yy, yz, zx, zy, zz = 1., 0., 0., 0., 1., 0., 0., 0., 1.
    
    return numpy.array([
            [xx, xy, xz, x],
            [yx, yy, yz, y],
            [zx, zy, zz, z],
            [ 0,  0,  0, 1]])

def vertices_ldraw_to_numpy(elements):
    # it's stupid to have to filter these, but there are some files in the OMR
    # that have non-float garbage (e.g. //) at the end of the line
    elements = filter_floats(elements)
    if len(elements)%3 != 0:
        raise BadVerticesException(
                'LDraw vertex elements must be divisible by 3')
    #elements = [float(element) for element in elements]
    vertices = [elements[i::3] for i in range(3)]
    vertices.append([1] * (len(elements)//3))
    
    return numpy.array(vertices)

def parse_ldcad_flags(arguments):
    ldcad_contents = arguments.split(None, 1)
    if len(ldcad_contents) == 1:
        return ldcad_contents[0], {}
    
    ldcad_command, flag_string = ldcad_contents
    
    flag_tokens = re.findall('\[[^\]]+\]', flag_string)
    flags = {}
    for flag_token in flag_tokens:
        flag_token = flag_token[1:-1]
        flag, value = flag_token.split('=')
        flags[flag.strip().lower()] = value.strip()

    return ldcad_command, flags

class LDrawCommand:
    @staticmethod
    def parse_commands(lines):
        commands = []
        for line in lines:
            try:
                commands.append(LDrawCommand.parse_command(line))
            except InvalidLDrawCommand:
                pass
        return commands
    
    @staticmethod
    def parse_command(line):
        line = re.sub('[^!-~]+', ' ', line).strip()
        line_contents = line.split(None, 1)
        if len(line_contents) != 2:
            raise InvalidLDrawCommand('Requires at least two tokens: %s'%line)
        
        command, arguments = line_contents
        if command == '0':
            return LDrawComment.parse_comment(arguments)
        elif command == '1':
            return LDrawImportCommand(arguments)
        elif command == '2':
            return LDrawLineCommand(arguments)
        elif command == '3':
            return LDrawTriangleCommand(arguments)
        elif command == '4':
            return LDrawQuadCommand(arguments)
        elif command == '5':
            return LDrawOptionalLineCommand(arguments)
        else:
            raise InvalidLDrawCommand(
                    'Unknown LDRAW Command %s: %s'%(command,line))

class LDrawComment(LDrawCommand):
    command = '0'
    
    @staticmethod
    def parse_comment(arguments):
        argument_contents = arguments.strip().split(None, 1)
        if len(argument_contents) == 2:
            comment_type, comment_arguments = argument_contents
            if comment_type == 'FILE':
                return LDrawFileComment(comment_arguments)
            elif 'author' in comment_type.lower():
                return LDrawAuthorComment(comment_arguments)
            elif comment_type == '!LDCAD':
                return LDCadCommand.parse_ldcad(comment_arguments)
        return LDrawComment(arguments)
        
    def __init__(self, comment):
        self.comment = comment
    
    def __str__(self):
        return '%s %s'%(self.command, self.comment)

class LDrawFileComment(LDrawComment):
    def __init__(self, file_name):
        self.comment = 'FILE ' + file_name
        self.file_name = file_name
        self.reference_name = get_reference_name(file_name)

class LDrawAuthorComment(LDrawComment):
    def __init__(self, author):
        self.comment = 'Author: ' + author
        self.author = author

class LDCadCommand(LDrawComment):
    @staticmethod
    def parse_ldcad(arguments):
        ldcad_command, flags = parse_ldcad_flags(arguments)
        if ldcad_command == 'SNAP_INCL':
            return LDCadSnapInclCommand(ldcad_command, flags)
        elif ldcad_command == 'SNAP_CLEAR':
            return LDCadSnapClearCommand(ldcad_command, flags)
        elif ldcad_command == 'SNAP_CYL':
            return LDCadSnapCylCommand(ldcad_command, flags)
        elif ldcad_command == 'SNAP_CLP':
            return LDCadSnapClpCommand(ldcad_command, flags)
        elif ldcad_command == 'SNAP_FGR':
            return LDCadSnapFgrCommand(ldcad_command, flags)
        elif ldcad_command == 'SNAP_GEN':
            return LDCadSnapGenCommand(ldcad_command, flags)
        elif ldcad_command == 'SNAP_SPH':
            return LDCadSnapSphCommand(ldcad_command, flags)
        return LDCadCommand(ldcad_command, flags)
    
    def __init__(self, ldcad_command, flags):
        self.comment = '!LDCAD ' + ldcad_command + ' ' + ' '.join(
                '[%s=%s]'%flag for flag in flags.items())
        self.ldcad_command = ldcad_command
        self.flags = flags

class LDCadSnapInclCommand(LDCadCommand):
    def __init__(self, ldcad_command, flags):
        super(LDCadSnapInclCommand, self).__init__(ldcad_command, flags)
        self.reference_name = get_reference_name(flags['ref'])
        self.transform = matrix_ldcad_to_numpy(flags)

class LDCadSnapClearCommand(LDCadCommand):
    def __init__(self, ldcad_command, flags):
        super(LDCadSnapClearCommand, self).__init__(ldcad_command, flags)
        self.id = flags.get('id', '')

class LDCadSnapStyleCommand(LDCadCommand):
    def __init__(self, ldcad_command, flags):
        super(LDCadSnapStyleCommand, self).__init__(ldcad_command, flags)
        self.id = flags.get('id', '')
        self.transform = matrix_ldcad_to_numpy(flags)

class LDCadSnapCylCommand(LDCadSnapStyleCommand):
    pass

class LDCadSnapClpCommand(LDCadSnapStyleCommand):
    pass

class LDCadSnapFgrCommand(LDCadSnapStyleCommand):
    pass

class LDCadSnapGenCommand(LDCadSnapStyleCommand):
    pass

class LDCadSnapSphCommand(LDCadSnapStyleCommand):
    pass

class LDrawImportCommand(LDrawCommand):
    command = '1'
    def __init__(self, arguments):
        (self.color,
         *matrix_elements,
         reference_name) = arguments.split(None, 13)
        self.reference_name = get_reference_name(reference_name)
        self.transform = matrix_ldraw_to_numpy(matrix_elements)
    
    def __str__(self):
        return '%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s'%(
                self.command,
                self.color,
                self.transform[0,3], self.transform[1,3], self.transform[2,3],
                self.transform[0,0], self.transform[0,1], self.transform[0,2],
                self.transform[1,0], self.transform[1,1], self.transform[1,2],
                self.transform[2,0], self.transform[2,1], self.transform[2,2],
                self.reference_name)

class LDrawContentCommand(LDrawCommand):
    def __init__(self, arguments):
        self.arguments = arguments
        self.color, *vertex_elements = arguments.split()
        self.vertices = vertices_ldraw_to_numpy(vertex_elements)
    
    def __str__(self):
        return '%i %s'%(self.command, self.argments)

class LDrawLineCommand(LDrawContentCommand):
    command = '2'

class LDrawTriangleCommand(LDrawContentCommand):
    command = '3'

class LDrawQuadCommand(LDrawContentCommand):
    command = '4'

class LDrawOptionalLineCommand(LDrawContentCommand):
    command = '5'
