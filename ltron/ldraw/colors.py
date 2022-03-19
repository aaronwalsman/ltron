import os

import numpy

import ltron.settings as settings

# faster that PIL.ImageColor.getrgb
def hex_to_rgb(rgb):
    if rgb[0] == '#':
        rgb = rgb[1:]
    elif rgb[:2] == '0x':
        rgb = rgb[2:]
    return (int(rgb[0:2], 16), int(rgb[2:4], 16), int(rgb[4:6], 16))

def rgb_to_hex(rgb):
    return '#' + ''.join(['0'*(c <= 16) + hex(c)[2:] for c in rgb]).upper()

ldsettings_path = os.path.join(settings.paths['ldraw'], 'LDConfig.ldr')
color_name_to_index = {}
color_index_to_name = {}
color_index_to_rgb = {}
color_index_to_edge_rgb = {}
color_index_to_hex = {}
color_index_to_edge_hex = {}
with open(ldsettings_path, 'r') as f:
    for line in f.readlines():
        line_parts = line.split()
        if len(line_parts) < 2:
            continue
        if line_parts[1] == '!COLOUR':
            name = line_parts[2]
            index = int(line_parts[4])
            color_name_to_index[name] = index
            color_index_to_name[index] = name
            color_hex = line_parts[6]
            color_rgb = hex_to_rgb(color_hex)
            edge_hex = line_parts[8]
            edge_rgb = hex_to_rgb(edge_hex)
            
            color_index_to_rgb[index] = color_rgb
            color_index_to_hex[index] = color_hex
            color_index_to_edge_rgb[index] = edge_rgb
            color_index_to_edge_hex[index] = edge_hex

# These colors come from the ImportLDraw blender plugin.
# The author says they got them from a piece of software called LGEO
# and provides a link to a page that doesn't seem to exist.
color_index_to_alt_rgb = {}
color_index_to_alt_rgb[0]  = ( 33,  33,  33)
color_index_to_alt_rgb[1]  = ( 13, 105, 171)
color_index_to_alt_rgb[2]  = ( 40, 127,  70)
color_index_to_alt_rgb[3]  = (  0, 143, 155)
color_index_to_alt_rgb[4]  = (196,  40,  27)
color_index_to_alt_rgb[5]  = (205,  98, 152)
color_index_to_alt_rgb[6]  = ( 98,  71,  50)
color_index_to_alt_rgb[7]  = (161, 165, 162)
color_index_to_alt_rgb[8]  = (109, 110, 108)
color_index_to_alt_rgb[9]  = (180, 210, 227)
color_index_to_alt_rgb[10] = ( 75, 151,  74)
color_index_to_alt_rgb[11] = ( 85, 165, 175)
color_index_to_alt_rgb[12] = (242, 112,  94)
color_index_to_alt_rgb[13] = (252, 151, 172)
color_index_to_alt_rgb[14] = (245, 205,  47)
color_index_to_alt_rgb[15] = (242, 243, 242)
color_index_to_alt_rgb[17] = (194, 218, 184)
color_index_to_alt_rgb[18] = (249, 233, 153)
color_index_to_alt_rgb[19] = (215, 197, 153)
color_index_to_alt_rgb[20] = (193, 202, 222)
color_index_to_alt_rgb[21] = (224, 255, 176)
color_index_to_alt_rgb[22] = (107,  50, 123)
color_index_to_alt_rgb[23] = ( 35,  71, 139)
color_index_to_alt_rgb[25] = (218, 133,  64)
color_index_to_alt_rgb[26] = (146,  57, 120)
color_index_to_alt_rgb[27] = (164, 189,  70)
color_index_to_alt_rgb[28] = (149, 138, 115)
color_index_to_alt_rgb[29] = (228, 173, 200)
color_index_to_alt_rgb[30] = (172, 120, 186)
color_index_to_alt_rgb[31] = (225, 213, 237)
color_index_to_alt_rgb[32] = (  0,  20,  20)
color_index_to_alt_rgb[33] = (123, 182, 232)
color_index_to_alt_rgb[34] = (132, 182, 141)
color_index_to_alt_rgb[35] = (217, 228, 167)
color_index_to_alt_rgb[36] = (205,  84,  75)
color_index_to_alt_rgb[37] = (228, 173, 200)
color_index_to_alt_rgb[38] = (255,  43,   0)
color_index_to_alt_rgb[40] = (166, 145, 130)
color_index_to_alt_rgb[41] = (170, 229, 255)
color_index_to_alt_rgb[42] = (198, 255,   0)
color_index_to_alt_rgb[43] = (193, 223, 240)
color_index_to_alt_rgb[44] = (150, 112, 159)
color_index_to_alt_rgb[46] = (247, 241, 141)
color_index_to_alt_rgb[47] = (252, 252, 252)
color_index_to_alt_rgb[52] = (156, 149, 199)
color_index_to_alt_rgb[54] = (255, 246, 123)
color_index_to_alt_rgb[57] = (226, 176,  96)
color_index_to_alt_rgb[65] = (236, 201,  53)
color_index_to_alt_rgb[66] = (202, 176,   0)
color_index_to_alt_rgb[67] = (255, 255, 255)
color_index_to_alt_rgb[68] = (243, 207, 155)
color_index_to_alt_rgb[69] = (142,  66, 133)
color_index_to_alt_rgb[70] = (105,  64,  39)
color_index_to_alt_rgb[71] = (163, 162, 164)
color_index_to_alt_rgb[72] = ( 99,  95,  97)
color_index_to_alt_rgb[73] = (110, 153, 201)
color_index_to_alt_rgb[74] = (161, 196, 139)
color_index_to_alt_rgb[77] = (220, 144, 149)
color_index_to_alt_rgb[78] = (246, 215, 179)
color_index_to_alt_rgb[79] = (255, 255, 255)
color_index_to_alt_rgb[80] = (140, 140, 140)
color_index_to_alt_rgb[82] = (219, 172,  52)
color_index_to_alt_rgb[84] = (170, 125,  85)
color_index_to_alt_rgb[85] = ( 52,  43, 117)
color_index_to_alt_rgb[86] = (124,  92,  69)
color_index_to_alt_rgb[89] = (155, 178, 239)
color_index_to_alt_rgb[92] = (204, 142, 104)
color_index_to_alt_rgb[100]= (238, 196, 182)
color_index_to_alt_rgb[115]= (199, 210,  60)
color_index_to_alt_rgb[134]= (174, 122,  89)
color_index_to_alt_rgb[135]= (171, 173, 172)
color_index_to_alt_rgb[137]= (106, 122, 150)
color_index_to_alt_rgb[142]= (220, 188, 129)
color_index_to_alt_rgb[148]= ( 62,  60,  57)
color_index_to_alt_rgb[151]= ( 14,  94,  77)
color_index_to_alt_rgb[179]= (160, 160, 160)
color_index_to_alt_rgb[183]= (242, 243, 242)
color_index_to_alt_rgb[191]= (248, 187,  61)
color_index_to_alt_rgb[212]= (159, 195, 233)
color_index_to_alt_rgb[216]= (143,  76,  42)
color_index_to_alt_rgb[226]= (253, 234, 140)
color_index_to_alt_rgb[232]= (125, 187, 221)
color_index_to_alt_rgb[256]= ( 33,  33,  33)
color_index_to_alt_rgb[272]= ( 32,  58,  86)
color_index_to_alt_rgb[273]= ( 13, 105, 171)
color_index_to_alt_rgb[288]= ( 39,  70,  44)
color_index_to_alt_rgb[294]= (189, 198, 173)
color_index_to_alt_rgb[297]= (170, 127,  46)
color_index_to_alt_rgb[308]= ( 53,  33,   0)
color_index_to_alt_rgb[313]= (171, 217, 255)
color_index_to_alt_rgb[320]= (123,  46,  47)
color_index_to_alt_rgb[321]= ( 70, 155, 195)
color_index_to_alt_rgb[322]= (104, 195, 226)
color_index_to_alt_rgb[323]= (211, 242, 234)
color_index_to_alt_rgb[324]= (196,   0,  38)
color_index_to_alt_rgb[326]= (226, 249, 154)
color_index_to_alt_rgb[330]= (119, 119,  78)
color_index_to_alt_rgb[334]= (187, 165,  61)
color_index_to_alt_rgb[335]= (149, 121, 118)
color_index_to_alt_rgb[366]= (209, 131,   4)
color_index_to_alt_rgb[373]= (135, 124, 144)
color_index_to_alt_rgb[375]= (193, 194, 193)
color_index_to_alt_rgb[378]= (120, 144, 129)
color_index_to_alt_rgb[379]= ( 94, 116, 140)
color_index_to_alt_rgb[383]= (224, 224, 224)
color_index_to_alt_rgb[406]= (  0,  29, 104)
color_index_to_alt_rgb[449]= (129,   0, 123)
color_index_to_alt_rgb[450]= (203, 132,  66)
color_index_to_alt_rgb[462]= (226, 155,  63)
color_index_to_alt_rgb[484]= (160,  95,  52)
color_index_to_alt_rgb[490]= (215, 240,   0)
color_index_to_alt_rgb[493]= (101, 103,  97)
color_index_to_alt_rgb[494]= (208, 208, 208)
color_index_to_alt_rgb[496]= (163, 162, 164)
color_index_to_alt_rgb[503]= (199, 193, 183)
color_index_to_alt_rgb[504]= (137, 135, 136)
color_index_to_alt_rgb[511]= (250, 250, 250)

for key in color_index_to_rgb:
    if key not in color_index_to_alt_rgb:
        color_index_to_alt_rgb[key] = color_index_to_rgb[key]

def get_color_rgb(index, default=None):
    if settings.render['color_scheme'] == 'ldraw':
        if default is None:
            return color_index_to_rgb[index]
        else:
            return color_index_to_rgb.get(index, default)
    elif settings.render['color_scheme'] == 'alt':
        if default is None:
            return color_index_to_alt_rgb[index]
        else:
            return color_index_to_alt_rgb.get(index, default)
    else:
        raise ValueError(
            'color_scheme (%s) setting must be "ldraw" or "alt"'%
            settings.render['color_scheme'])
