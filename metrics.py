import argparse
import shutil

import cv2 as cv
import numpy as np
import pykitti

from SuperGluePretrainedNetwork.models import utils
from rpe import transform44


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n - 1):
        if len(lst[i:i + n]) > 1:
            yield lst[i:i + n]


def get_euler_angles(R):
    theta_x = np.arctan2(R[2, 1], R[2, 2])
    theta_y = np.arctan2(-R[2, 0], (R[2, 1] ** 2 + R[2, 2] ** 2) ** 0.5)
    theta_z = np.arctan2(R[1, 0], R[0, 0])

    return theta_x, theta_y, theta_z


def n_of_matches(path_to_results):
    all_resulting_paths = [os.path.join(path_to_results, name) for name in os.listdir(path_to_results) if
                           name.endswith('.npz')]

    number_of_matches = []
    percentage_of_matches = []
    for path in all_resulting_paths:
        res = np.load(path, allow_pickle=True)
        if 'arr_0' in res.keys():
            res = res['arr_0'][()]
        number_of_matches.append(np.sum(res['matches'] > -1))
        percentage_of_matches.append(
            np.sum(res['matches'] > -1) / ((res['keypoints0'].shape[0] + res['keypoints1'].shape[0]) / 2.))

    return {
        "average_number_of_matches": np.mean(number_of_matches),
        "std_number_of_matches": np.std(number_of_matches),
        "average_percentage_of_matches": np.mean(percentage_of_matches),
        "std_percentage_of_matches": np.std(percentage_of_matches),
    }


def avg_number_of_matches(path_to_pairs: str, path_to_output: str,
                          path_to_image_folder: str = 'example_pics',
                          method_name: str = 'superglue', visualize: bool = False) -> dict:
    """
    The function returns dictionary with obtained results
    :param visualize: Flag. If True, images will be visualized
    :param path_to_image_folder: path to a folder with images
    :param method_name: method name
    :return: dict with results
    """
    if method_name.lower() == 'superglue':
        result = avg_number_of_matches_superglue(path_to_pairs=path_to_pairs,
                                                 path_to_image_folder=path_to_image_folder,
                                                 path_to_output=path_to_output, visualize=visualize)
    elif method_name.lower() == 'lowe':
        raise NotImplementedError

    return result


def avg_number_of_matches_superglue(path_to_pairs: str, path_to_image_folder: str,
                                    path_to_output: str,
                                    visualize: bool) -> dict:
    """
    :param path_to_image_folder: path_to_image_folder: path to a folder with images
    :param visualize: if True, images will be visualized
    :return: dict with results
    """
    path_to_results = path_to_output
    path_to_examples = path_to_image_folder

    LAUNCHING_COMMAND_SUPERGLUE = "python SuperGluePretrainedNetwork/match_pairs.py --input_pairs " + \
                                  f"{path_to_pairs} --input_dir " + \
                                  f"{path_to_image_folder} --output_dir " + \
                                  f"{path_to_output}"

    if os.path.exists(path_to_results):
        shutil.rmtree(path_to_results)
    all_images = sorted([name for name in os.listdir(path_to_examples) if name.endswith(".png")])
    with open(path_to_pairs, 'w') as file:
        for i1, i2 in chunks(all_images, 2):
            file.write(f"{i1} {i2}\n")

    if visualize:
        LAUNCHING_COMMAND_SUPERGLUE += ' --viz'
    os.system(LAUNCHING_COMMAND_SUPERGLUE)

    result = n_of_matches(path_to_results)
    return result


def get_rbt_estimation(points1: np.ndarray, points2: np.ndarray, dataset_name: str = 'kitti_1') -> dict:
    """
    The function computes estimateion of RBT
    :param dataset_name: dataset name
    :param points1: An array of keypoints from the 1st image
    :param points2: An array of keypoints from the 2nd image in the corresponding order
    :return:
    The function returns a dictionary with estimations of camera rotation, translation and their composition (RBT)
    """
    if dataset_name == 'kitti_1':
        K = np.array(
            "9.812178e+02 0.000000e+00 6.900000e+02 0.000000e+00 9.758994e+02 2.471364e+02 0.000000e+00 0.000000e+00 1.000000e+00".split()).astype(
            float).reshape(3, 3)
        distCoeffs = np.array(
            "-3.791375e-01 2.148119e-01 1.227094e-03 2.343833e-03 -7.910379e-02".split()).astype(float).reshape(1, 5)
    elif dataset_name == 'tum_1':
        K = np.array("517.3 0 318.6 0 516.5 255.3 0 0 1".split()).astype(float).reshape(3, 3)
        distCoeffs = np.array("0.2624  -0.9531  -0.0054  0.0026  1.1633".split()).astype(float).reshape(1, 5)

    pts_l_norm = cv.undistortPoints(np.expand_dims(points1, axis=1), cameraMatrix=K, distCoeffs=distCoeffs)
    pts_r_norm = cv.undistortPoints(np.expand_dims(points2, axis=1), cameraMatrix=K, distCoeffs=distCoeffs)

    E, mask = cv.findEssentialMat(pts_l_norm, pts_r_norm, focal=1.0, pp=(0., 0.), method=cv.RANSAC, prob=0.999,
                                  threshold=3.0)
    points, R_hat, t_hat, mask = cv.recoverPose(E, pts_l_norm, pts_r_norm)

    T_hat = np.vstack((np.hstack((R_hat, t_hat)), np.array([0., 0., 0., 1.])))

    return {
        "R_hat": R_hat,
        "t_hat": t_hat,
        "T_hat": T_hat,
    }


import os
import numpy as np


def get_observations_from_gt(filepath, obs_path='groundtruth.txt', index=0):
    best_res, best_obs = float('inf'), None
    gt_ts = float(os.path.basename(filepath).split("_")[index])
    with open(obs_path, 'r') as file:
        for _ in range(3): file.readline()
        for line in file:
            ts, *obs = line.split(' ')

            res = abs(gt_ts - float(ts.strip()))
            if res < best_res:
                best_res, best_obs = res, obs

    return np.fromiter(map(lambda x: x.strip(), best_obs), dtype=float)


def rgbd_rotation_error(path_to_results='results/kitti_test_output/'):
    K = np.array("517.3 0 318.6 0 516.5 255.3 0 0 1".split()).astype(float).reshape(3, 3)
    names = [n for n in sorted(os.listdir(path_to_results)) if n.endswith('npz')]

    angle_error = []
    translation_error = []

    for i in range(len(names)):
        n = os.path.join(path_to_results, names[i])
        loaded_data = np.load(n, allow_pickle=True)

        if 'arr_0' in loaded_data.keys():
            loaded_data = loaded_data['arr_0'][()]
        match_indices = loaded_data['matches'][loaded_data['matches'] > -1]
        points1 = loaded_data['keypoints0'][loaded_data['matches'] > -1]
        points2 = loaded_data['keypoints1'][match_indices]

        ret = utils.estimate_pose(points1, points2, K, K, 3.)
        if ret is None:
            continue
            err_t, err_R = np.inf, np.inf
        else:
            R, t, inliers = ret

            vec1 = get_observations_from_gt(n, index=0)
            vec1 = np.insert(vec1, 0, 6., axis=0)
            T1 = transform44(vec1)

            vec2 = get_observations_from_gt(n, index=1)
            vec2 = np.insert(vec2, 0, 6., axis=0)
            T2 = transform44(vec2)

            T_rel = np.dot(np.linalg.inv(T1), T2)
            err_t, err_R = utils.compute_pose_error(T_rel, R, t)

        angle_error.append(err_R)
        translation_error.append(err_t)

    return {
        "R_hat_mean": np.mean(angle_error),
        "R_hat_std": np.std(angle_error),
        "t_hat_mean": np.mean(translation_error),
        "t_hat_std": np.std(translation_error),
    }


def kitti_rotation_error(path_to_results='results/kitti_test_output/'):
    K = np.array(
        "9.812178e+02 0.000000e+00 6.900000e+02 0.000000e+00 9.758994e+02 2.471364e+02 0.000000e+00 0.000000e+00 1.000000e+00".split()).astype(
        float).reshape(3, 3)
    names = [n for n in sorted(os.listdir(path_to_results)) if n.endswith('npz')]
    OXTS_BASE_PATH = os.path.abspath('./data/kitti/campus/2011_09_28_image/oxts/data/')
    files = [os.path.join(OXTS_BASE_PATH, file_name) for file_name in os.listdir(OXTS_BASE_PATH) if
             file_name.endswith('.txt')]
    output = pykitti.utils.load_oxts_packets_and_poses(files)

    angle_error = []
    translation_error = []

    for i in range(len(names)):
        n = os.path.join(path_to_results, names[i])

        loaded_data = np.load(n, allow_pickle=True)
        if 'arr_0' in loaded_data.keys():
            loaded_data = loaded_data['arr_0'][()]
        match_indices = loaded_data['matches'][loaded_data['matches'] > -1]
        points1 = loaded_data['keypoints0'][loaded_data['matches'] > -1]
        points2 = loaded_data['keypoints1'][match_indices]

        ret = utils.estimate_pose(points1, points2, K, K, 3.)
        R, t, inliers = ret
        T_rel = np.dot(np.linalg.inv(output[i].T_w_imu), output[i + 1].T_w_imu)
        err_t, err_R = utils.compute_pose_error(T_rel, R, t)

        angle_error.append(err_R)
        translation_error.append(err_t)

    return {
        "R_hat_mean": np.mean(angle_error),
        "R_hat_std": np.std(angle_error),
        "t_hat_mean": np.mean(translation_error),
        "t_hat_std": np.std(translation_error),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute average number of matches metric')
    parser.add_argument(
        '--path_to_pairs', type=str, default='SuperGluePretrainedNetwork/test_pairs.txt')
    parser.add_argument(
        '--path_to_images', type=str, default='SuperGluePretrainedNetwork/example_pics')
    parser.add_argument(
        '--path_to_output', type=str, default='SuperGluePretrainedNetwork/test_output')
    parser.add_argument(
        '--viz', action='store_true',
        help='Visualize the matches and dump the plots')

    args = parser.parse_args()
    print(args)

    result = avg_number_of_matches(path_to_pairs=args.path_to_pairs,
                                   path_to_image_folder=args.path_to_images, path_to_output=args.path_to_output,
                                   visualize=args.viz)
    print(
        f"For SuperGlue average number of matches is {result['average_number_of_matches']}" + \
        f" +/- {result['std_number_of_matches']} and" + \
        f" percentage {100 * result['average_percentage_of_matches']}% +/- {100 * result['std_percentage_of_matches']}")
