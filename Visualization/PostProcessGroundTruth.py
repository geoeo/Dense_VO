from Numerics import SE3

class PostProcessTUW:

    # Optitrack/Rviz coversion capture X and Z are flipped
    def post_process_in_mem(self, se3):
        rot = SE3.extract_rotation(se3)
        euler = SE3.rotationMatrixToEulerAngles(rot)
        rot_new = SE3.makeS03(euler[1], euler[2], euler[0])
        se3[0:3, 0:3] = rot_new
        x = se3[0, 3]
        y = se3[1, 3]
        z = se3[2, 3]
        se3[0, 3] = -y
        se3[1, 3] = z
        se3[2, 3] = -x

class PostProcessTUM:

    def post_process_in_mem(self, se3):
        rot = SE3.extract_rotation(se3)
        euler = SE3.rotationMatrixToEulerAngles(rot)
        rot_new = SE3.makeS03(euler[0], -euler[1], euler[2])
        se3[0:3, 0:3] = rot_new
        se3[0, 3] = -se3[0, 3]
        se3[1, 3] = -se3[1, 3]
        #se3[2, 3] = -se3[2, 3]

class PostProcessTUM_XYZ:

    def post_process_in_mem(self, se3):
        rot = SE3.extract_rotation(se3)
        euler = SE3.rotationMatrixToEulerAngles(rot)
        rot_new = SE3.makeS03(euler[0], -euler[1], euler[2])
        se3[0:3, 0:3] = rot_new
        #se3[0, 3] = -se3[0, 3]
        se3[1, 3] = -se3[1, 3]
        #se3[2, 3] = -se3[2, 3]