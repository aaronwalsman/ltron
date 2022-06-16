from ltron import matching

def edit_distance(assembly_a, assembly_b, part_names):
    first_matches, first_offset = matching.match_assemblies(
        assembly_a, assembly_b, part_names)
    
    import pdb
    pdb.set_trace()
