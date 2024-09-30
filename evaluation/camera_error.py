from colmap_file_io import read_cameras_binary, read_images_binary
import pickle
import numpy as np
import os
import argparse
#from opensora.datasets.camera_utils import read_camera_file

def read_camera_file(camera_file):
    with open(camera_file, 'r') as f:
        lines = f.readlines()
    
    intrinsics = []
    poses = []
    for line in lines[1:]:
        line = np.array(line.replace('\n','').split(' '), dtype=np.float32)
        intrinsics.append(line[1:7])
        poses.append(line[7:])
    return {'intrinsics':np.array(intrinsics), 
            'poses':np.array(poses),
            'filename': camera_file}


def get_image_number(image):
    return int(image.name.split('.')[0])

def write_colmap_to_file(images_bin, fpath, normalize=True):

    prefix = "0 0.500000000 0.900000000 0.500000000 0.500000000 0.000000000 0.000000000 "

    images = read_images_binary(images_bin)
    images = sorted([*images.values()], key=get_image_number)

    first_image = images[0]
    rt_3x4 = np.concatenate((first_image.qvec2rotmat(), first_image.tvec[None].T), axis=1)
    inv_first_camera_pred = np.linalg.inv(np.concatenate((rt_3x4, np.array([[0,0,0,1]])), axis=0))

    all_rt = []
    for image in images[:]:

        pred_rt_3x4 = np.concatenate((image.qvec2rotmat(), image.tvec[None].T), axis=1)
        pred_rt = np.concatenate((pred_rt_3x4, np.array([[0,0,0,1]])), axis=0) 
        pred_rt = inv_first_camera_pred @ pred_rt
        all_rt.append(pred_rt[:-1])
    all_rt = np.array(all_rt)

    if normalize:
        scale = np.abs(all_rt[:,:,-1]).max()
        if scale > 0.1:
            all_rt[:,:,-1]/=scale

    with open(fpath, "w") as file:
        file.write(f'{images_bin}\n')
        for rt in all_rt:
            ext = rt.flatten()
            line = prefix  + ' '.join([f"{x:.8f}" for x in ext])
            file.write(line + "\n")



def filter_outliers(arr, n_std=2):
    mean = np.mean(arr)
    std_dev = np.std(arr)

    # Identify elements within 2 standard deviations from the mean
    filtered_arr = arr[(arr >= mean - n_std * std_dev) & (arr <= mean + n_std * std_dev)]
    # if len(filtered_arr) != len(arr):
    #     print("before filter\n", arr)
    #     print("after filter\n", filtered_arr)
    return filtered_arr

def calc_camera_error(images_bin, cameras_gt, image_start_idx=0):
    eps = 1e-4

    #test_root = '/sensei-fs/users/scheong/github/Open-Sora/temp/test_1'
    #image_bin = os.path.join(test_root, 'dense/0/sparse/images.bin')
    #camera_path = os.path.join(test_root, 'camera.pkl')

    images = read_images_binary(images_bin)

    images = sorted([*images.values()], key=get_image_number)
    ## for both gt and pred, grab the first camera pose and store the inverse    
    first_image = images[0]
    first_image_id = get_image_number(first_image)
    
    total_image = len(images)
    #print(f'Total {len(images)} images, first = {first_image.name}.')
    assert  first_image_id >= image_start_idx, f"Image id start from {image_start_idx}, not {first_image_id}."

    rt_3x4 = np.concatenate((first_image.qvec2rotmat(), first_image.tvec[None].T), axis=1)
    inv_first_camera_pred = np.linalg.inv(np.concatenate((rt_3x4, np.array([[0,0,0,1]])), axis=0))

    gt_frame_id = first_image_id - image_start_idx
    inv_first_camera_gt = np.linalg.inv(np.concatenate((cameras_gt[gt_frame_id],np.array([[0,0,0,1]])), axis=0))


    ## loop over all the other cameras, express their poses in the first camera frame, and compute the error
    rotation_errors, translation_error = [], []
    gt_translation,  pred_translation = [], []
    for image_id, image in enumerate(images[0:]): # Skip the first (reference)
        
        gt_frame_id = get_image_number(image) - image_start_idx
        gt_rt = inv_first_camera_gt @ np.concatenate((cameras_gt[gt_frame_id],np.array([[0,0,0,1]])), axis=0)
        pred_rt_3x4 = np.concatenate((image.qvec2rotmat(), image.tvec[None].T), axis=1)
        #pred_rt = inv_first_camera_pred @ np.concatenate((pred_rt_3x4, np.array([[0,0,0,1]])), axis=0)
        pred_rt = np.concatenate((pred_rt_3x4, np.array([[0,0,0,1]])), axis=0) 
        #print('Before invert\n', pred_rt)
        pred_rt = inv_first_camera_pred @ pred_rt
        #print('After invert\n', pred_rt)
        rot_trace = np.trace(gt_rt[:3,:3].T @ pred_rt[:3,:3])
        #print('GT\n', gt_rt)

        try:
            if ((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
                raise ValueError(f"A matrix has trace outside valid range [-1-eps,3+eps]: {rot_trace.item()}")
        except:
            continue
        
        rot_err = np.arccos(0.5 * (rot_trace - 1)) * 180 / np.pi

        rotation_errors.append(rot_err)

        gt_translation.append(gt_rt[:3,3])
        pred_translation.append(pred_rt[:3,3])
        
        #TODO: compute the translation error

        # gt_t = gt_rt[:3,3]/np.linalg.norm(gt_rt[:3,3])
        # pred_t = pred_rt[:3,3]/np.linalg.norm(pred_rt[:3,3]) 
        # transl_error = np.arccos(np.clip(gt_t @ pred_t, -1.0, 1.0)) * 180 / np.pi
        # translation_error.append(transl_error)

    def calc_scale(translation):
        #if dim is None:
        #    return 1
        return np.abs(translation).max()

    gt_translation = np.array(gt_translation)
    pred_translation = np.array(pred_translation)

    #scale_dim = np.abs(gt_translation.mean()).argmax()

    if gt_translation.mean()!=0.:
        gt_scale = calc_scale(gt_translation)
        pred_scale = calc_scale(pred_translation)
    else: # rotation only
        gt_scale = 1
        pred_scale = 10

    # print(pred_translation)
    # print(pred_scale)    
    gt_translation /= gt_scale
    #print(gt_translation)

    pred_translation /= pred_scale
    #print(pred_translation)

    translation_error = gt_translation - pred_translation
    translation_error = np.linalg.norm(translation_error, axis=1)
    #print(translation_error)

    # Filter out outliers
    #translation_error = filter_outliers(translation_error)
    #rotation_errors = filter_outliers(np.array(rotation_errors))

    rot_error_mean = np.mean(rotation_errors)
    transl_error_mean = np.mean(translation_error)

    # print(f"Mean rotation error: {rot_error_mean:.4f}")
    # print(f"Mean translation error: {transl_error_mean:.4f}")
    # print(f"Translation scales. gt {gt_scale:.2f}  pred {pred_scale:.2f} relative {pred_scale/gt_scale:.1f}")

    return {'rot_error_mean': rot_error_mean,
            'transl_error_mean': transl_error_mean,
            'transl_gt_scale': gt_scale,
            'transl_pred_scale': pred_scale,
            'total_image':total_image
            }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_bin', type=str ,help="local path to store results")
    parser.add_argument('--camera_file', type=str ,help="Path to gif files")
    parser.add_argument('--image_start_idx', type=int, default=1, help="Path to gif files")
    args = parser.parse_args()

    
    if args.camera_file.split('.')[-1] == 'txt':
        cameras_gt = read_camera_file(args.camera_file)['poses']
        cameras_gt = cameras_gt.reshape(-1, 3, 4)
    else:
        cameras_gt = pickle.load(open(args.camera_file, 'rb'))

    calc_camera_error(args.images_bin, cameras_gt,
                      image_start_idx = args.image_start_idx)