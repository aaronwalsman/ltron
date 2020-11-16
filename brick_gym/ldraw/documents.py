import os

import brick_gym.ldraw.paths as ldraw_paths
from brick_gym.ldraw.commands import *
from brick_gym.ldraw.exceptions import *

'''
There are two differentiating factors here.
One is the actual file type.  The other is where it comes from.

The file types are (.mpd, .ldr, .dat).  These are all the same, except
that .mpd files can have multiple sub-sections contained in one document
that each represent a separate file.

Where the file comes from is a differnt story.  A file could be something
that a user creates that sits anywhere.  It could also be a part file in
the "parts" directory, or a sub-assembly in the "parts/s" directory, or
a component file in the "p" directory, etc. etc.  I want to be able to stop
recursion at any of these stages to limit the ammount of garbage we need to
load.  Although maaaaaaybe if we only load things once and use references, we
can get away with loading everything?  Let's try that real quick.

Ok, ALSO, when do we import all the references, and what is the structure?
I currently have these documents that each contains a list of "commands".
I think the structure probably needs to be: first the document gets all the
commands, then the document goes through all the import commands and brings
in anything that is not already in the references?  But then what?  Who stores the connections between documents?  Is it just an implicit connections formed by the import commands and the names looked up in the reference data?

Ok, did I go too far here?  It seems like what I had yesterday was smaller than all of this new stuff, and was close to actually working.  Why did I want to do this?  One thing was to only store one copy of each file.  Another thing was not liking the old format where I had each line stored as a tuple of raw text and then imported lines related to that text.

Let's think about format here:
The naive solution would be to just use the imports to merge all the text into one big list each time.  This would be ok as long as I'm ok with resolving all the transformations and colors right away.  This would mean that each line in the new list would not be the original data in the file, but a modified version with the transformations applied and colors inherited.

A contrast to this would be to have each line (command) be represented faithfully, and then store everything in a structure so that we can compute the correct transforms and colors later.  Doing this would involve a second form of parsing or recursive evaluation of the structure later.  This feels nice because now we will have a faithful representation of the original data.

After thinking about it for a minute, I like building it up in classes how I'm doing it here.  I think the main thing that was getting me a minute ago is that I don't have a good way to store the references.  But now that I think about it, the reference table should be stored as a member of the class, and instantiated the first time, but passed to the other sub-documents.  This means that each full model will share references with other components imported into that model, but not other models... unless you manually take the extra step to supply a reference table to that first call.  Ok, I like this.
'''

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
        for command in self.commands:
            if isinstance(command, LDrawImportCommand):
                reference_name = command.clean_reference_name
                if reference_name not in self.reference_table['ldraw']:
                    LDrawDocument.parse_document(
                            reference_name, self.reference_table)
            if isinstance(command, LDCadSnapInclCommand):
                reference_name = ldraw_paths.clean_name(command.flags['ref'])
                if reference_name not in self.reference_table['shadow']:
                    LDrawDocument.parse_document(
                            reference_name, self.reference_table, shadow=True)
        if not self.shadow:
            try:
                LDrawDocument.parse_document(
                        self.clean_name, self.reference_table, shadow=True)
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
