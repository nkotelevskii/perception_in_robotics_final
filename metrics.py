import argparse
import os
import shutil

import cv2 as cv
import numpy as np


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

    all_resulting_paths = [os.path.join(path_to_results, name) for name in os.listdir(path_to_results) if
                           name.endswith('.npz')]

    number_of_matches = []
    percentage_of_matches = []
    for path in all_resulting_paths:
        res = np.load(path)
        number_of_matches.append(np.sum(res['matches'] > -1))
        percentage_of_matches.append(
            np.sum(res['matches'] > -1) / ((res['keypoints0'].shape[0] + res['keypoints1'].shape[0]) / 2.))

    return {
        "average_number_of_matches": np.mean(number_of_matches),
        "std_number_of_matches": np.std(number_of_matches),
        "average_percentage_of_matches": np.mean(percentage_of_matches),
        "std_percentage_of_matches": np.std(percentage_of_matches),
    }


def get_rbt_estimation(points1: np.ndarray, points2: np.ndarray, K: np.ndarray, distCoeffs: np.ndarray) -> dict:
    """
    The function computes estimateion of RBT
    :param points1: An array of keypoints from the 1st image
    :param points2: An array of keypoints from the 2nd image in the corresponding order
    :param K: Camera matrix (matrix of intrinsics)
    :param distCoeffs: Distortion coefficients
    :return:
    The function returns a dictionary with estimations of camera rotation, translation and their composition (RBT)
    """
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
