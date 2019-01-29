from Numerics import SE3

class PostProcessTUW_R200:

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

class PostProcessTUW_R300_DS2:

    # Optitrack/Rviz coversion capture X and Z are flipped
    def post_process_in_mem(self, se3):
        rot = SE3.extract_rotation(se3)
        euler = SE3.rotationMatrixToEulerAngles(rot)
        #rot_new = SE3.makeS03(euler[1], -euler[2], euler[0])
        rot_new = SE3.makeS03(euler[0], euler[2], euler[1])
        #se3[0:3, 0:3] = rot_new
        x = se3[0, 3]
        y = se3[1, 3]
        z = se3[2, 3]
        #se3[0, 3] = -y
        #se3[1, 3] = z
        #se3[2, 3] = -x

        se3[0, 3] = x
        se3[1, 3] = z
        se3[2, 3] = y


class PostProcessTUW_R300_DS4:

    # Optitrack/Rviz coversion capture X and Z are flipped
    def post_process_in_mem(self, se3):
        rot = SE3.extract_rotation(se3)
        euler = SE3.rotationMatrixToEulerAngles(rot)
        rot_new = SE3.makeS03(euler[1], -euler[2], euler[0])
        se3[0:3, 0:3] = rot_new
        x = se3[0, 3]
        y = se3[1, 3]
        z = se3[2, 3]
        se3[0, 3] = -y
        se3[1, 3] = z
        se3[2, 3] = -x

class PostProcessTUW_R300_DS5:

    # Optitrack/Rviz coversion capture X and Z are flipped
    def post_process_in_mem(self, se3):
        rot = SE3.extract_rotation(se3)
        euler = SE3.rotationMatrixToEulerAngles(rot)
        #rot_new = SE3.makeS03(euler[2], -euler[1], euler[0])
        rot_new = SE3.makeS03(euler[1], euler[2], euler[0])
        se3[0:3, 0:3] = rot_new
        x = se3[0, 3]
        y = se3[1, 3]
        z = se3[2, 3]
        #se3[0, 3] = -z
        #se3[1, 3] = x
        #se3[2, 3] = -y

        se3[0, 3] = -x
        se3[1, 3] = z
        se3[2, 3] = -y

class PostProcessTUM_F2:

    def post_process_in_mem(self, se3):
        rot = SE3.extract_rotation(se3)
        euler = SE3.rotationMatrixToEulerAngles(rot)
        rot_new = SE3.makeS03(euler[0], -euler[1], euler[2])
        se3[0:3, 0:3] = rot_new
        #se3[0, 3] = -se3[0, 3]
        #se3[1, 3] = -se3[1, 3]
        se3[2, 3] = -se3[2, 3]

class PostProcessTUM_F1:

    def post_process_in_mem(self, se3):
        rot = SE3.extract_rotation(se3)
        euler = SE3.rotationMatrixToEulerAngles(rot)
        rot_new = SE3.makeS03(euler[0], euler[1], -euler[2])
        se3[0:3, 0:3] = rot_new
        #se3[0, 3] = -se3[0, 3]
        se3[1, 3] = -se3[1, 3]
        #se3[2, 3] = -se3[2, 3]