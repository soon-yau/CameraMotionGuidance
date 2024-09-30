import random
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim):
        self.fig = plt.figure(figsize=(18, 7))
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.view_init(elev=90, azim=-90, roll=0)	
        self.plotly_data = None  # plotly data traces
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        print('initialize camera pose visualizer')

    def extrinsic2pyramid(self, extrinsic, color_map='red', hw_ratio=9/16, base_xval=1, zval=3):
        vertex_std = np.array([[0, 0, 0, 1],
                               [base_xval, -base_xval * hw_ratio, zval, 1],
                               [base_xval, base_xval * hw_ratio, zval, 1],
                               [-base_xval, base_xval * hw_ratio, zval, 1],
                               [-base_xval, -base_xval * hw_ratio, zval, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]

        color = color_map if isinstance(color_map, str) else plt.cm.rainbow(color_map)

        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))

    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=self.ax, orientation='vertical', label='Frame Number')

    def show(self, path):
        plt.title('Extrinsic Parameters')
        save_path = path.replace('.txt','.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_file_path', required=True, help='path to the trajectory txt file')
    parser.add_argument('--out_file_path', default='')
    parser.add_argument('--hw_ratio', default=9/16, type=float, help='the height over width of the film plane')
    parser.add_argument('--sample_stride', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--all_frames', action='store_true')
    parser.add_argument('--base_xval', type=float, default=0.2)
    parser.add_argument('--zval', type=float, default=0.2)
    parser.add_argument('--use_exact_fx', action='store_true')
    parser.add_argument('--relative_c2w', action='store_true')
    parser.add_argument('--norm_transl', action='store_true')
    parser.add_argument('--x_min', type=float, default=-1.1)
    parser.add_argument('--x_max', type=float, default=1.1)
    parser.add_argument('--y_min', type=float, default=-1.1)
    parser.add_argument('--y_max', type=float, default=1.1)
    parser.add_argument('--z_min', type=float, default=-1.1)
    parser.add_argument('--z_max', type=float, default=1.1)
    return parser.parse_args()


def get_c2w(w2cs, transform_matrix, relative_c2w, normalize_translation):
    if relative_c2w:
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ np.linalg.inv(w2c) for w2c in w2cs[1:]]
    else:
        ret_poses = [np.linalg.inv(w2c) for w2c in w2cs]
    ret_poses = [transform_matrix @ x for x in ret_poses]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    if normalize_translation:
        scale = np.abs(ret_poses[:,:-1,-1]).max()
        #scale*=10
        if scale > 0.1:
            ret_poses[:,:-1,-1]/=scale
    return ret_poses


if __name__ == '__main__':
    args = get_args()
    with open(args.pose_file_path, 'r') as f:
        poses = f.readlines()
    w2cs = [np.asarray([float(p) for p in pose.strip().split(' ')[7:]]).reshape(3, 4) for pose in poses[1:]]
    fxs = [float(pose.strip().split(' ')[1]) for pose in poses[1:]]

    if args.all_frames:
        args.num_frames = len(fxs)
        args.sample_stride = 1
    cropped_length = args.num_frames * args.sample_stride
    total_frames = len(w2cs)
    start_frame_ind = random.randint(0, max(0, total_frames - cropped_length - 1))
    end_frame_ind = min(start_frame_ind + cropped_length, total_frames)
    frame_ind = np.linspace(start_frame_ind, end_frame_ind - 1, args.num_frames, dtype=int)

    #drop_sets = [[1,3,5,7,9,11,13],
    #            [2,6,10,14],
    #            [4, 12],
    #            [8]]
    #drops = 4
    #for i in range(drops):
    #    frame_ind = [x for x in frame_ind if x not in drop_sets[i]]
    #print(frame_ind)
    w2cs = [w2cs[x] for x in frame_ind]
    transform_matrix = np.asarray([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).reshape(4, 4)
    last_row = np.zeros((1, 4))
    last_row[0, -1] = 1.0
    w2cs = [np.concatenate((w2c, last_row), axis=0) for w2c in w2cs]
    c2ws = get_c2w(w2cs, transform_matrix, args.relative_c2w, args.norm_transl)

    visualizer = CameraPoseVisualizer([args.x_min, args.x_max], [args.y_min, args.y_max], [args.z_min, args.z_max])
    for frame_idx, c2w in enumerate(c2ws):
        visualizer.extrinsic2pyramid(c2w, frame_idx / args.num_frames, hw_ratio=args.hw_ratio, base_xval=args.base_xval,
                                     zval=(fxs[frame_idx] if args.use_exact_fx else args.zval))

    visualizer.colorbar(args.num_frames)
    fields = args.pose_file_path.split('/')
    subdir = '/'.join(fields[:-2]+['viz']+[fields[-1].split('.')[0]])
    
    pose_file_name = os.path.join(subdir, fields[-2]+'.png')
    if args.out_file_path:
        pose_file_name = args.out_file_path
    os.makedirs(os.path.dirname(pose_file_name), exist_ok=True)
    visualizer.show(pose_file_name)