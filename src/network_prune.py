# Similar Arhcitecture Pruned
def get_architecture_set(backbone_list):
    arc_set = set()
    arc_list = []
    for arc in backbone_list:
        arc_list.append(arc[1])
    arc_list = tuple(arc_list)
    arc_set.add(arc_list)
    return arc_set

def similar_architecture_pruned(before_backbone_set, current_backbone_set):
    if before_backbone_set == current_backbone_set:
        return True
    return False