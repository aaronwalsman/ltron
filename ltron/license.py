import os

import ltron.settings as settings

def generate_license():
    license = '='*80
    license += '''LICENSING INFORMATION:
--------------------------------------------------------------------------------
This document contains information about the various licenses used by different
components of LTRON.  Please see the sections below for more details.


'''
    license += '='*80
    license += ltron_license()
    license += '='*80
    license += ldraw_license()
    license += '='*80
    license += omr_license()
    license += '='*80
    license += ldcad_license()
    license += '='*80
    license += splendor_mesh_license()
    license += '='*80
    return license

def ltron_license():
    license = '''LTRON
Source: %s
--------------------------------------------------------------------------------
The ltron software package is provided under the MIT license.
https://opensource.org/licenses/MIT
--------------------------------------------------------------------------------
Copyright 2021 Aaron Walsman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


'''%(settings.urls['ltron'])
    return license

def ldraw_license():
    readme_path = os.path.join(settings.paths['ldraw'], 'CAreadme.txt')
    license_path = os.path.join(settings.paths['ldraw'], 'CAlicense.txt')
    license = '''LDraw
Source: %s
--------------------------------------------------------------------------------
The LDraw parts library is used by LTRON to describe the 3D structure of
individual bricks.  It is provided by the LDraw contributors under a
Creative Commons license.  For more information on this license see:

https://ldraw.org/legal-info
https://creativecommons.org/licenses/by/2.0/


'''%(settings.urls['ldraw'])
    
    return license

def omr_license():
    license = '''Open Model Repository
Source: %s
Modified: %s
--------------------------------------------------------------------------------
The OMR files have been provided by various authors in the LDraw community
under Creative Commons Licensing.  The LTRON authors have bundled these files
into a single package for convenience.  For information on individual file
authorship, please consult the header of the file.  For information on the
Creative Commons licensing see:

https://creativecommons.org/licenses/by/2.0/
--------------------------------------------------------------------------------
MODIFICATIONS:
--------------------------------------------------------------------------------
The following files have been modified by the LTRON authors with small fixes:

42043-1 - Mercedes Benz Arocs 3245 - Flatbed Trailer.mpd
6526-1 - Red Line Racer.mpd

The file headers have been updated to reflect the fact that these changes have
been made.


'''%(
        settings.urls['omr_ldraw'],
        settings.urls['omr'],
    )
    
    return license

ldcad_license_text = '''LDCad license agreement (V3)

LDCad and its configuration files, the software from here on, are free for      personal (non commercial) and educational use. The software might be used in    this manner, free of charge, by anyone as far local law permits.

The author (Roland Melkert) does not guarantee perfect operation of the         software, nor can he be held responsible for any damage and / or loss resulting from the use of the software in anyway.

Using the software to (help) create digital material (including but not limited to instruction booklets) to be sold later on is permitted as long a single copy of the material is donated to the author free of charge.

It is permitted to customize and repackage the software as long this is done    without modifying the main executable or adding third party software (including but not limited to 'adware' and 'spyware').

(re)Distribution of the software in any form is only allowed when done so free  of charge and a reference to the original software's website (www.melkert.net/  LDCad) is included. If it concerns a customized version this must be clearly    stated and in such cases the package must (also) be made available through a    public accessible website.

Permission is granted to post, display and/or distribute screenshots (including videoclips) of the software for use on social media and or promotional material.


By using the software you agree with the contents of this document and          therefore agree with the license.

For questions, contact- or additional information visit: www.melkert.net/LDCad
'''

def ldcad_license():
    license = '''LDCad
Source: %s
--------------------------------------------------------------------------------
LDCad is a Lego CAD modelling software by Roland Melkert which can be found at:
%s.  It comes bundled with data files that LTRON uses for part snapping.

--------------------------------------------------------------------------------
LDCad License:
--------------------------------------------------------------------------------
%s


'''%(
        settings.urls['ldcad'],
        settings.urls['ldcad_home'],
        ldcad_license_text,
    )
    
    return license

def splendor_mesh_license():
    license = '''Splendor Meshes
Sources:
%s
%s
--------------------------------------------------------------------------------
The LDraw components have been converted into mesh obj files for compatibility
with LTRON.  This conversion was done using the import-ldraw plugin for blender:

https://github.com/TobyLobster/ImportLDraw
https://blender.org

These meshes are derivative works of the original LDraw files which are
distributed under a Creative Commons license.  LTRON provides these
derivative meshes under the same Creative Commons license.  For more
information on LDraw licensing and the Creative Commons license see:

https://ldraw.org/legal-info
https://creativecommons.org/licenses/by/2.0/


'''%(settings.urls['ltron_assets_low'], settings.urls['ltron_assets_high'])
    return license
