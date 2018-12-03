from Numerics import SE3
import os
import errno

delimiter = ","
new_line = "\n"
type = '.txt'


def write_vo_output_to_file(name, info, output_dir_path, twist_list):

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
    f.write("# max its, eps, image offset, use ndc, use robust, use motion prior, use ackermann post" + new_line)
    f.write("# "+info+new_line)

    for twist in twist_list:
        measurement_string \
            = f"{twist[0,0]:.9f}" + delimiter \
              + f"{twist[1,0]:.9f}" + delimiter \
              + f"{twist[2,0]:.9f}" + delimiter \
              + f"{twist[3,0]:.9f}" + delimiter \
              + f"{twist[4,0]:.9f}" + delimiter \
              + f"{twist[5,0]:.9f}" \
              + new_line
        f.write(measurement_string)
    f.close()


def load_vo_from_file(file_path):
    SE3_list = []
    f = open(file_path)
    data = f.read()
    lines = data.replace(delimiter, " ").replace("\t", " ").split("\n")
    list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
            len(line) > 0 and line[0] != "#"]
    for twist in list:
        assert(len(twist)==6)
        SE3_mat = SE3.twist_to_SE3(twist)
        SE3_list.append(SE3_mat)

    return SE3_list







