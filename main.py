import numpy as np
import cv2
from scipy.linalg import svd
from glob import glob
import pandas as pd


def extract_features(images):
    sift = cv2.SIFT_create()
    keypoints = []
    descriptors = []

    for gray in images:
        kp, desc = sift.detectAndCompute(gray, None)
        keypoints.append(kp)
        descriptors.append(desc)

    return keypoints, descriptors


def match_features(keypoints, descriptors):
    matcher = cv2.BFMatcher()
    matches = []

    for i in range(len(descriptors) - 1):
        matches.append(matcher.knnMatch(descriptors[i], descriptors[i + 1], k=2))

    return matches


def estimate_camera_pose(keypoints, matches, camera_matrix):
    poses = [np.zeros((3, 1))]
    rotation_matrices = []
    translation_vectors = []

    for i in range(len(matches)):
        src_pts = np.float32([keypoints[i][m[0].queryIdx].pt for m in matches[i]]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[i+1][m[0].trainIdx].pt for m in matches[i]]).reshape(-1, 1, 2)

        essential_matrix, _ = cv2.findEssentialMat(src_pts, dst_pts, camera_matrix)
        _, rotation_matrix, translation_vector, _ = cv2.recoverPose(essential_matrix, src_pts, dst_pts, camera_matrix)

        rotation_matrices.append(rotation_matrix)
        translation_vectors.append(translation_vector)

        poses.append(poses[i] + rotation_matrix.dot(translation_vector))

    return poses, rotation_matrices, translation_vectors

def triangulate_points(keypoints, matches, poses, rotation_matrices, translation_vectors, camera_matrix):
    points_3d = []

    for i in range(len(poses) - 1):
        src_pts = np.float32([keypoints[i][m[0].queryIdx].pt for m in matches[i]]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[i+1][m[0].trainIdx].pt for m in matches[i]]).reshape(-1, 1, 2)

        projection_matrix_1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        projection_matrix_2 = np.hstack((rotation_matrices[i], translation_vectors[i]))

        points = cv2.triangulatePoints(projection_matrix_1, projection_matrix_2, src_pts, dst_pts)
        points /= points[3]
        points_3d.append(points)

    return points_3d


def save_to_submission(submission, dataset, scene, image_path, rotation_matrices, translation_vectors):
    for i in range(len(rotation_matrices)):
        rotation_matrix = rotation_matrices[i]
        translation_vector = translation_vectors[i]

        string1 = ";".join(rotation_matrix.flatten().astype(str))
        string2 = ";".join(translation_vector.flatten().astype(str))

        new_row = pd.DataFrame({'dataset': [dataset], 'scene': [scene], 'image_path': [image_path],
                                'rotation_matrix': [string1], 'translation_vector': [string2]})
        submission = pd.concat([submission, new_row], ignore_index=True)

    return submission



def sfm(images, submission, j, dataset, scene, image_path):
    camera_matrix = np.eye(3)

    keypoints, descriptors = extract_features(images)
    matches = match_features(keypoints, descriptors)

    poses, rotation_matrices, translation_vectors = estimate_camera_pose(keypoints, matches, camera_matrix)

    submission = save_to_submission(submission, dataset, scene, image_path[j-1], rotation_matrices, translation_vectors)

    points_3d = triangulate_points(keypoints, matches, poses, rotation_matrices, translation_vectors, camera_matrix)

    return poses, points_3d, submission



# Main part of your code
submission = pd.read_csv('./image-matching-challenge-2023/sample_submission.csv')

dataset = '2cfa01ab573141e4'
scene = '2fa124afd1f74f38'
src = f'./image-matching-challenge-2023/train/heritage/dioscuri'
image_path = glob(f'{src}/images/*')
images = [cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2GRAY) for im in glob(f'{src}/images/*')[:3]]

str1 = '0;0;0;0;0;0;0;0;0'
str2 = '0;0;0'
nrow = pd.DataFrame({'dataset': [dataset], 'image_path': [image_path[0]], 'scene': [scene],
                     'rotation_matrix': [str1], 'translation_vector': [str2]})
submission = pd.concat([submission, nrow], ignore_index=True)

j = 1
poses, points_3d, submission = sfm(images, submission, j, dataset, scene, image_path)

# Save submission to CSV
submission.to_csv("submission.csv", index=False)
