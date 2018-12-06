from Numerics import SE3, Utils
import os
import errno
import numpy as np

delimiter = ","
new_line = "\n"
type = '.txt'


def write_vo_output_to_file(name, info, output_dir_path, twist_list, encoder_list = None):

    if not os.path.exists(output_dir_path):
        try:
            os.mkdir(output_dir_path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            else:
                pass

    full_file_path = output_dir_path + name + info + type
    f = open(full_file_path,"w")
    f.write("# x,y,z,roll(x),pitch(y),yaw(z)"+new_line)
    f.write("# max its, eps, image offset, use ndc, use robust, use motion prior, use ackermann post, image range, offset" + new_line)
    f.write("# "+info+new_line)


    for i in range(0,len(twist_list)):
        twist = twist_list[i]
        measurement_string \
            = f"{twist[0,0]:.9f}" + delimiter \
              + f"{twist[1,0]:.9f}" + delimiter \
              + f"{twist[2,0]:.9f}" + delimiter \
              + f"{twist[3,0]:.9f}" + delimiter \
              + f"{twist[4,0]:.9f}" + delimiter \
              + f"{twist[5,0]:.9f}"

        if encoder_list is not None:
            encoder_values = encoder_list[i]
            measurement_string += f"{encoder_values[0]:.9f}" + delimiter + f"{encoder_values[1]:.9f}"

        measurement_string += new_line
        f.write(measurement_string)
    f.close()


def load_vo_from_file(file_path):
    SE3_list = []
    encoder_list = []
    f = open(file_path)
    data = f.read()
    lines = data.replace(delimiter, " ").replace("\t", " ").split("\n")
    list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
            len(line) > 0 and line[0] != "#"]
    for data_line in list:
        data_line_len = len(data_line)
        assert data_line_len == 6 or data_line_len == 8

        twist = np.zeros((6,1), dtype=Utils.matrix_data_type)
        twist[0,0] = data_line[0]
        twist[1,0] = data_line[1]
        twist[2,0] = data_line[2]
        twist[3,0] = data_line[3]
        twist[4,0] = data_line[4]
        twist[5,0] = data_line[5]
        SE3_mat = SE3.twist_to_SE3(twist)
        SE3_list.append(SE3_mat)

        if data_line_len > 6:
            enc = np.zeros(2, dtype=Utils.matrix_data_type)
            enc[0] = data_line[6]
            enc[1] = data_line[7]
            encoder_list.append(enc)

    return SE3_list, encoder_list







