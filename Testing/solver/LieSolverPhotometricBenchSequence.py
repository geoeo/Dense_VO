import numpy as np
from Numerics import Utils, SE3
from Camera import Intrinsic, Camera
from VisualOdometry import Frame, SolverThreadManager
from Benchmark import Parser, associate
from Visualization import Visualizer
from math import pi



bench_path = '/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Bench/'
xyz_dataset = 'rgbd_dataset_freiburg1_xyz/'
rgb_folder = 'rgb/'
depth_folder = 'depth/'

dataset_root = bench_path+xyz_dataset
rgb_text = dataset_root +'rgb.txt'
depth_text = dataset_root+'depth.txt'
match_text = dataset_root+'matches.txt'
groundtruth_text = dataset_root+'groundtruth.txt'

#######
#rgb_id_ref = 1305031102.175304
#rgb_id_target = 1305031102.211214

#rgb_id_ref_2 = 1305031102.211214
#rgb_id_target_2 = 1305031102.275326

#rgb_id_ref_3 = 1305031102.275326
#rgb_id_target_3 = 1305031102.311267

#rgb_id_ref_4 = 1305031102.311267
#rgb_id_target_4 = 1305031102.343233

#rgb_id_ref_5 = 1305031102.343233
#rgb_id_target_5 = 1305031102.375329

#rgb_id_ref_6 = 1305031102.375329
#rgb_id_target_6 = 1305031102.411258

#rgb_id_ref_7 = 1305031102.411258
#rgb_id_target_7 = 1305031102.443271
#
# rgb_id_ref_8 = 1305031102.443271
# rgb_id_target_8 = 1305031102.475318
#
# rgb_id_ref_9 = 1305031102.475318
# rgb_id_target_9 = 1305031102.511219
#
# rgb_id_ref_10 = 1305031102.511219
# rgb_id_target_10 = 1305031102.575286 # jump
#
# rgb_id_ref_11 = 1305031102.575286
# rgb_id_target_11 = 1305031102.611233
#
# rgb_id_ref_12 = 1305031102.611233
# rgb_id_target_12 =1305031102.675285
#
# rgb_id_ref_13 = 1305031102.675285
# rgb_id_target_13 = 1305031102.711263
#
# rgb_id_ref_14 = 1305031102.711263
# rgb_id_target_14 = 1305031102.743234
#
# rgb_id_ref_15 = 1305031102.743234
# rgb_id_target_15 = 1305031102.775472


#rgb_id_ref_17 = 1305031102.811232
#rgb_id_target_17 = 1305031102.843290
#######
# Y
#rgb_id_ref = 1305031119.079223
#rgb_id_target = 1305031119.111328

#rgb_id_ref_2 = 1305031119.111328
#rgb_id_target_2 = 1305031119.147616

#rgb_id_ref_3 = 1305031119.147616
#rgb_id_target_3 = 1305031119.179226

#rgb_id_ref_4 = 1305031119.179226
#rgb_id_target_4 = 1305031119.211364

#rgb_id_ref_5 = 1305031119.211364
#rgb_id_target_5 = 1305031119.247399
########

#rgb_id_ref = 1305031105.643273
#rgb_id_target = 1305031105.711309

#rgb_id_ref_2 = 1305031105.711309
#rgb_id_target_2 = 1305031105.743312

#rgb_id_ref_3 = 1305031105.743312
#rgb_id_target_3 = 1305031105.775339

#rgb_id_ref_4 = 1305031105.775339
#rgb_id_target_4 = 1305031105.811283

#rgb_id_ref_5 = 1305031105.811283
#rgb_id_target_5 = 1305031105.843271

##

#rgb_id_ref = 1305031105.643273
#rgb_id_target = 1305031105.711309

#rgb_id_ref_2 = 1305031105.711309
#rgb_id_target_2 = 1305031105.811283

#rgb_id_ref_3 = 1305031105.811283
#rgb_id_target_3 = 1305031105.875337

#rgb_id_ref_4 = 1305031105.875337
#rgb_id_target_4 = 1305031105.943272

#rgb_id_ref_5 = 1305031105.943272
#rgb_id_target_5 = 1305031106.011285

#rgb_id_ref_6 = 1305031106.011285
#rgb_id_target_6 = 1305031106.075330

#rgb_id_ref_7 = 1305031106.075330
#rgb_id_target_7 = 1305031106.143355

#############
# X
rgb_id_ref = 1305031108.143334
rgb_id_target = 1305031108.176058

rgb_id_ref_2 = 1305031108.176058
rgb_id_target_2 = 1305031108.211475

rgb_id_ref_3 = 1305031108.211475
rgb_id_target_3 = 1305031108.243347

rgb_id_ref_4 = 1305031108.243347
rgb_id_target_4 = 1305031108.275358

rgb_id_ref_5 = 1305031108.275358
rgb_id_target_5 = 1305031108.311332

##

#rgb_id_ref = 1305031108.143334
#rgb_id_target = 1305031108.211475

#rgb_id_ref_2 = 1305031108.211475
#rgb_id_target_2 = 1305031108.275358

#rgb_id_ref_3 = 1305031108.275358
#rgb_id_target_3 = 1305031108.343278

##

#rgb_id_ref = 1305031108.243347
#rgb_id_target = 1305031108.275358

#rgb_id_ref_2 = 1305031108.275358
#rgb_id_target_2 = 1305031108.311332

#rgb_id_ref_3 = 1305031108.311332
#rgb_id_target_3 = 1305031108.343278

#rgb_id_ref_4 = 1305031108.343278
#rgb_id_target_4 = 1305031108.375410

#########

# rgb_id_ref = 1305031109.243290
# rgb_id_target = 1305031109.275308
#
# rgb_id_ref_2 = 1305031109.275308
# rgb_id_target_2 = 1305031109.311329
#
# rgb_id_ref_3 = 1305031109.311329
# rgb_id_target_3 = 1305031109.343248
#
# rgb_id_ref_4 = 1305031109.343248
# rgb_id_target_4 = 1305031109.375397
#
# rgb_id_ref_5 = 1305031109.375397
# rgb_id_target_5 = 1305031109.411329
#
# rgb_id_ref_6 = 1305031109.411329
# rgb_id_target_6 = 1305031109.443302
#
# rgb_id_ref_7 = 1305031109.443302
# rgb_id_target_7 = 1305031109.475363


#########

#ref_id_list = [rgb_id_ref]
#target_id_list = [rgb_id_target]

#ref_id_list = [rgb_id_ref, rgb_id_ref_2]
#target_id_list = [rgb_id_target, rgb_id_target_2]

#ref_id_list = [rgb_id_ref, rgb_id_ref_2,rgb_id_ref_3]
#target_id_list = [rgb_id_target, rgb_id_target_2,rgb_id_target_3]

#ref_id_list = [rgb_id_ref, rgb_id_ref_2,rgb_id_ref_3, rgb_id_ref_4]
#target_id_list = [rgb_id_target, rgb_id_target_2,rgb_id_target_3, rgb_id_target_4]

#ref_id_list = [rgb_id_target_3,rgb_id_target_2, rgb_id_target]
#target_id_list = [rgb_id_ref_3, rgb_id_ref_2, rgb_id_ref]

ref_id_list = [rgb_id_ref, rgb_id_ref_2, rgb_id_ref_3, rgb_id_ref_4, rgb_id_ref_5]
target_id_list = [rgb_id_target, rgb_id_target_2, rgb_id_target_3, rgb_id_target_4, rgb_id_target_5]

#ref_id_list = [rgb_id_ref, rgb_id_ref_2, rgb_id_ref_3, rgb_id_ref_4, rgb_id_ref_5, rgb_id_ref_6, rgb_id_ref_7]
#target_id_list = [rgb_id_target, rgb_id_target_2, rgb_id_target_3, rgb_id_target_4, rgb_id_target_5, rgb_id_target_6, rgb_id_target_7]

#ref_id_list = [rgb_id_ref, rgb_id_ref_2, rgb_id_ref_3, rgb_id_ref_4, rgb_id_ref_5, rgb_id_ref_6, rgb_id_ref_7, rgb_id_ref_8 ]
#target_id_list = [rgb_id_target, rgb_id_target_2, rgb_id_target_3, rgb_id_target_4, rgb_id_target_5, rgb_id_target_6, rgb_id_target_7, rgb_id_target_8]

#ref_id_list = [rgb_id_ref, rgb_id_ref_2, rgb_id_ref_3, rgb_id_ref_4, rgb_id_ref_5, rgb_id_ref_6, rgb_id_ref_7, rgb_id_ref_8, rgb_id_ref_9, rgb_id_ref_10, rgb_id_ref_11, rgb_id_ref_12, rgb_id_ref_13, rgb_id_ref_14, rgb_id_ref_15]
#target_id_list = [rgb_id_target, rgb_id_target_2, rgb_id_target_3, rgb_id_target_4, rgb_id_target_5, rgb_id_target_6, rgb_id_target_7, rgb_id_target_8, rgb_id_target_9, rgb_id_target_10, rgb_id_target_11, rgb_id_target_12, rgb_id_target_13, rgb_id_target_14, rgb_id_target_15]


ground_truth_acc = np.identity(4,Utils.matrix_data_type)
se3_estimate_acc = np.identity(4,Utils.matrix_data_type)
ground_truth_list = []
pose_estimate_list = []
ref_image_list = []
target_image_list = []

depth_factor = 5000.0
#depth_factor = 1.0
use_ndc = True


image_groundtruth_dict = dict(associate.match(rgb_text,groundtruth_text))
#se3_ground_truth_prior = np.transpose(SE3.quaternion_to_s03(0.6132, 0.5962, -0.3311, -0.3986))
se3_ground_truth_prior = SE3.makeS03(0,0,pi)
se3_ground_truth_prior = np.append(se3_ground_truth_prior,np.zeros((3,1),dtype=Utils.matrix_data_type),axis=1)
se3_ground_truth_prior = SE3.append_homogeneous_along_y(se3_ground_truth_prior)
#se3_ground_truth_prior = SE3.invert(se3_ground_truth_prior)
se3_ground_truth_prior[0:3,3] = 0


for i in range(0, len(ref_id_list)):

    ref_id = ref_id_list[i]
    target_id = target_id_list[i]

    SE3_ref_target = Parser.generate_ground_truth_se3(groundtruth_text,image_groundtruth_dict,ref_id,target_id,None)
    im_greyscale_reference, im_depth_reference = Parser.generate_image_depth_pair(dataset_root,rgb_text,depth_text,match_text,ref_id)
    im_greyscale_target, im_depth_target = Parser.generate_image_depth_pair(dataset_root,rgb_text,depth_text,match_text,target_id)

    ground_truth_acc = np.matmul(SE3_ref_target,ground_truth_acc)

    ground_truth_list.append(ground_truth_acc)
    ref_image_list.append((im_greyscale_reference, im_depth_reference))
    target_image_list.append((im_greyscale_target, im_depth_target))


im_greyscale_reference_1, im_depth_reference_1 = ref_image_list[0]
(image_height, image_width) = im_greyscale_reference_1.shape
se3_identity = np.identity(4, dtype=Utils.matrix_data_type)
# image gradient induces a coordiante system where y is flipped i.e have to flip it here
intrinsic_identity = Intrinsic.Intrinsic(-517.3, -516.5, 318.6, 239.5) # freiburg_1
if use_ndc:
    #intrinsic_identity = Intrinsic.Intrinsic(1, 1, 1/2, 1/2) # for ndc
    intrinsic_identity = Intrinsic.Intrinsic(-1, -516.5/517.3, 318.6/image_width, 239.5/image_height) # for ndc


camera_reference = Camera.Camera(intrinsic_identity, se3_identity)
camera_target = Camera.Camera(intrinsic_identity, se3_identity)

visualizer = Visualizer.Visualizer(ground_truth_list)

motion_cov_inv = np.identity(6,dtype=Utils.matrix_data_type)
twist_prior = np.zeros((6,1),dtype=Utils.matrix_data_type)

for i in range(0, len(ref_image_list)):
    im_greyscale_reference, im_depth_reference = ref_image_list[i]
    im_greyscale_target, im_depth_target = target_image_list[i]

    im_depth_reference /= depth_factor
    im_depth_target /= depth_factor


    # Since our virtual image plane is on the same side as our depth values
    # we push all depth values out to guarantee that they are always infront of the image plane
    depth_t = (im_depth_reference != 0).astype(Utils.depth_data_type_float)
    im_depth_reference = np.add(im_depth_reference,depth_t)
    depth_t = (im_depth_target != 0).astype(Utils.depth_data_type_float)
    im_depth_target = np.add(im_depth_target,depth_t)

    # We only need the gradients of the target frame
    frame_reference = Frame.Frame(im_greyscale_reference, im_depth_reference, camera_reference, False)
    frame_target = Frame.Frame(im_greyscale_target, im_depth_target, camera_target, True)

    solver_manager = SolverThreadManager.Manager(1,
                                                 "Solver Manager",
                                                 frame_reference,
                                                 frame_target,
                                                 max_its=100,
                                                 eps=0.001,  #0.00001, 0.00005, 0.00000001
                                                 alpha_step=0.6,  # 0.1, 0.04, 0.005, 0.55 - motion prior
                                                 gradient_monitoring_window_start=0,
                                                 image_range_offset_start=0,
                                                 twist_prior=twist_prior,
                                                 motion_cov_inv = motion_cov_inv,
                                                 use_ndc=use_ndc,
                                                 use_robust=True,
                                                 track_pose_estimates=True,
                                                 use_motion_prior=True,
                                                 debug=False)

    solver_manager.start()
    solver_manager.join()  # wait to complete

    motion_cov_inv = solver_manager.motion_cov_inv
    twist_prior = solver_manager.twist_final
    se3_estimate_acc = np.matmul(solver_manager.SE3_est_final,se3_estimate_acc)
    pose_estimate_list.append(se3_estimate_acc)
visualizer.visualize_poses(pose_estimate_list, draw= False)
visualizer.show()







#SE3_final = solver_manager.pose_estimate_list[len(solver_manager.pose_estimate_list)-1]

#euler_angles_XYZ = SE3.rotationMatrixToEulerAngles(SE3.extract_rotation(SE3_final))
#euler_angles_gt_XYZ = SE3.rotationMatrixToEulerAngles(SE3.extract_rotation(SE3_ref_target))

#print('*'*80)
#print('GROUND TRUTH\n')
#print(SE3_ref_target)
#print(Utils.radians_to_degrees(euler_angles_gt_XYZ[0]),
#      Utils.radians_to_degrees(euler_angles_gt_XYZ[1]),
#      Utils.radians_to_degrees(euler_angles_gt_XYZ[2]))
#print('*'*80)

#print(SE3_final)
#print(Utils.radians_to_degrees(euler_angles_XYZ[0]),
#      Utils.radians_to_degrees(euler_angles_XYZ[1]),
#      Utils.radians_to_degrees(euler_angles_XYZ[2]))



