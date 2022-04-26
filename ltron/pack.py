from collections import OrderedDict

# we could use bisect here, but this is designed for a small number of blocks

# this is also exploratory garbage, not used anywhere

class BlockPack:
    def __init__(self, blocks):
        self.total = 0
        self.blocks = OrderedDict()
        for name, items in blocks.items():
            blocks[name] = {
                'items':items,
                'offset':self.total,
            }
            self.total += len(items)
    
    def local_to_global_index(self, name, i):
        return self.blocks[name]['offset'] + i
    
    def __getitem__(self, name, i):
        return self.blocks[name]['items'][i]
    
    def unravel_index(self, i):
        for name, block in self.blocks.items():
            if i >= block['offset']:
                i -= block['offset']
            else:
                break
        return name, i
    
    def unravel_item(self, i):
        for name, block in self.blocks.items():
            if i >= block['offset']:
                i -= block['offset']
            else:
                return block['items'][i]

def block_pack(blocks):
    result = []
    for name, elements in blocks:
        for element in elements:
            result.append((name,element))
    
    return result

# name + element -> global index
result.index((name, element))

# global index -> name + element
result[index]
