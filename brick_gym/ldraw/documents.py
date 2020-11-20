import os

import brick_gym.ldraw.paths as ldraw_paths
from brick_gym.ldraw.commands import *
from brick_gym.ldraw.exceptions import *

class LDrawMissingFileComment(LDrawException):
    pass

class LDrawDocument:
    @staticmethod
    def parse_document(file_path, reference_table = None, shadow = False):
        file_name, ext = os.path.splitext(file_path)
        if ext == '.mpd':
            try:
                return LDrawMPDMainFile(file_path, reference_table, shadow)
            except LDrawMissingFileComment:
                return LDrawLDR(file_path, reference_table, shadow)
        elif ext == '.ldr':
            return LDrawLDR(file_path, reference_table, shadow)
        elif ext == '.dat':
            return LDrawDAT(file_path, reference_table, shadow)
    
    def set_reference_table(self, reference_table):
        if reference_table is None:
            reference_table = {'ldraw':{}, 'shadow':{}}
        self.reference_table = reference_table
        if self.shadow:
            self.reference_table['shadow'][self.clean_name] = self
        else:   
            self.reference_table['ldraw'][self.clean_name] = self
    
    def resolve_file_path(self, file_path):
        if self.shadow:
            self.resolved_file_path = ldraw_paths.resolve_shadow_path(file_path)
        else:
            self.resolved_file_path = ldraw_paths.resolve_ldraw_path(file_path)
    
    def import_references(self):
        #imported_documents = []
        for command in self.commands:
            # ldraw import commands
            if isinstance(command, LDrawImportCommand):
                reference_name = command.clean_reference_name
                if reference_name not in self.reference_table['ldraw']:
                    LDrawDocument.parse_document(
                            reference_name, self.reference_table)
                #imported_documents.append(
                #        self.reference_table['ldraw'][reference_name])
            
            # ldcad SNAP_INCL commands
            if isinstance(command, LDCadSnapInclCommand):
                #reference_name = ldraw_paths.clean_name(command.flags['ref'])
                reference_name = command.clean_reference_name
                if reference_name not in self.reference_table['shadow']:
                    LDrawDocument.parse_document(
                            reference_name, self.reference_table, shadow=True)
                #imported_documents.append(
                #        self.reference_table['shadow'][reference_name])
        
        # shadow
        if not self.shadow:
            try:
                if self.clean_name not in self.reference_table['shadow']:
                    LDrawDocument.parse_document(
                            self.clean_name, self.reference_table, shadow=True)
                #imported_documents.append(
                #        self.reference_table['shadow'][self.clean_name])
            except ldraw_paths.LDrawMissingPath:
                pass

class LDrawMPDMainFile(LDrawDocument):
    def __init__(self, file_path, reference_table = None, shadow = False):
        
        # initialize reference_table
        self.shadow = shadow
        self.resolve_file_path(file_path)
        self.clean_name = ldraw_paths.clean_name(file_path)
        self.set_reference_table(reference_table)
        
        # resolve the file path and parse all commands in this file
        commands = LDrawCommand.parse_commands(
                open(self.resolved_file_path).readlines())
        
        # make sure that the first line is a file comment
        if not len(commands):
            raise LDrawMissingFileComment(
                    'MPD file appears to be empty, must start with "0 FILE"')
        if not isinstance(commands[0], LDrawFileComment):
            raise LDrawMissingFileComment('MPD file must start with "0 FILE"')
        
        # split the commands into groups for each sub-file
        file_indices = [
                i for i in range(len(commands))
                if isinstance(commands[i], LDrawFileComment)]
        file_indices.append(len(commands))
        subfile_command_lists = [commands[start:end]
                for start, end in zip(file_indices[:-1], file_indices[1:])]
        
        # store the main file's commands
        self.commands = subfile_command_lists[0]
        
        # build internal files
        self.internal_files = [
                LDrawMPDInternalFile(subfile_commands, self.reference_table)
                for subfile_commands in subfile_command_lists[1:]]
        
        # import references
        # We must do this after creating the subfiles because the main file will
        # reference them, and so they need to be in the reference table.
        # Also, this MPDMainFile must take responsiblity to import the subfile
        # references because we need the full reference table filled with each
        # other before any of them bring in their references
        self.import_references()
        for internal_file in self.internal_files:
            internal_file.import_references()

class LDrawMPDInternalFile(LDrawDocument):
    def __init__(self, commands, reference_table = None):
        
        # make sure the commands list starts with a FILE comment
        if not isinstance(commands[0], LDrawFileComment):
            raise LDrawMissingFileComment(
                    'MPD Internal File must start with "0 FILE"')
        
        # initialize reference_table
        self.clean_name = commands[0].clean_name
        self.shadow = False
        self.set_reference_table(reference_table)
        
        # store commands
        self.commands = commands
        
class LDrawLDR(LDrawDocument):
    def __init__(self, file_path, reference_table = None, shadow = False):
        
        # initialize reference table
        self.shadow = shadow
        self.resolve_file_path(file_path)
        self.clean_name = ldraw_paths.clean_name(file_path)
        self.set_reference_table(reference_table)
        
        # resolve the file path and parse all commands in this file
        self.commands = LDrawCommand.parse_commands(
                open(self.resolved_file_path).readlines())
        
        self.import_references()

class LDrawDAT(LDrawLDR):
    pass
'''
    def __init__(self, file_path, reference_table = None, shadow = False):
        super(LDrawDAT, self).__init__(file_path, reference_table)
        
        # import the shadow file if available
        if not shadow:
            try:
                shadow_path = ldraw_paths.resolve_shadow_path(file_path)
                LDrawDocument.parse_document(
                        shadow_path, self.reference_table, shadow = True)
'''
