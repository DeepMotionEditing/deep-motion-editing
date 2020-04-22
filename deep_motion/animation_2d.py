
class Motion():

    def __init__(self, bvh_path, human=False):
        bvh_data = read_bvh(bvh_path)
        self.rotations = #Tx4
        self.global_positions = #Tx3
        self.skeleton_structure = #S
        self.skeleton_bone_length = #S

    if human:
        flag_end_effectors

    @staticmethod
    def read_bvh(bvh_path):
        #parse_bvh
        #euler_to_quaternon
        returnbvh_databvh_data
