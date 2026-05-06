"""
This is the main script you should call to train and test UrsoNet

Copyright (c) Pedro F. Proenza
Licensed under the MIT License (see LICENSE for details)

------------------------------------------------------------

Usage: Check README

"""

import os
import numpy as np
import os.path
import skimage
import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import se3lib

import utils
import net
print("net file =", getattr(net, "__file__", None))
print("has UrsoNet =", hasattr(net, "UrsoNet"))

from config import Config

import urso
import speed

# [Modified] Set matplotlib backend to avoid issues in some environments
# If you are running locally on Windows with a screen, 'TkAgg' or 'Qt5Agg' is fine.
# If running on a headless server, use 'Agg'.
try:
    matplotlib.use('TkAgg')
except:
    pass

# Models directory (where weights are stored)
MODEL_DIR = os.path.abspath("./models")
DEFAULT_LOGS_DIR = os.path.join(MODEL_DIR, "logs")

# Dataset directory
DATA_DIR = os.path.abspath("./datasets")

# Path to trained weights file of Mask-RCNN on COCO
COCO_WEIGHTS_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")

OrientationParamOptions = ['quaternion', 'euler_angles', 'angle_axis']

#新增函数：位置解析
def extract_location_from_result(model, result, dataset=None):
    """Parse location prediction from model.detect() result dict.

        Returns:
            loc_est: np.ndarray, shape (3,)
            loc_info: dict with optional extra fields
        """
    loc_info = {}

    if model.config.REGRESS_LOC:
        loc_est = np.asarray(result['loc']).reshape(-1)

        # epistemic uncertainty 来自 MC Dropout，本身不依赖 BAYESIAN_LOC
        if 'loc_epistemic_var' in result:
            loc_info['loc_epistemic_var'] = np.asarray(result['loc_epistemic_var']).reshape(-1)

        # 以下这些才依赖 Bayesian location head
        if getattr(model.config, "BAYESIAN_LOC", False):
            if 'loc_logvar' in result:
                loc_info['loc_logvar'] = np.asarray(result['loc_logvar']).reshape(-1)
            if 'loc_var' in result:
                loc_info['loc_var'] = np.asarray(result['loc_var']).reshape(-1)
            if 'loc_aleatoric_var' in result:
                loc_info['loc_aleatoric_var'] = np.asarray(result['loc_aleatoric_var']).reshape(-1)
            if 'loc_total_var' in result:
                loc_info['loc_total_var'] = np.asarray(result['loc_total_var']).reshape(-1)
    else:
        loc_pmf = utils.stable_softmax(result['loc'])
        loc_est = np.asarray(
            np.asmatrix(loc_pmf) * np.asmatrix(dataset.histogram_3D_map)
        ).reshape(-1)
        loc_info['loc_pmf'] = loc_pmf

    return loc_est, loc_info


#新增函数：姿态解析，把原始的姿态头输出的bin^3数量的原始向量（概率分布）进行解析，输出加权拟合后的姿态（四元数形式）
def extract_orientation_from_result(model, result, dataset=None):
    """Parse orientation prediction from model.detect()/detect_mc() result dict.

    Returns:
        q_est: np.ndarray, shape (4,)
        ori_info: dict with optional extra fields
    """
    ori_info = {}

    if model.config.REGRESS_ORI:
        ori_raw = np.asarray(result['ori']).reshape(-1)

        if model.config.ORIENTATION_PARAM == 'quaternion':
            q_est = ori_raw
            q_norm = np.linalg.norm(q_est)
            if q_norm > 1e-12:
                q_est = q_est / q_norm

        elif model.config.ORIENTATION_PARAM == 'euler_angles':
            q_est = se3lib.SO32quat(
                se3lib.euler2SO3_left(ori_raw[0], ori_raw[1], ori_raw[2])
            )
        elif model.config.ORIENTATION_PARAM == 'angle_axis':
            theta = np.linalg.norm(ori_raw)
            if theta < 1e-6:
                v = np.zeros(3, dtype=np.float32)
            else:
                v = ori_raw / theta
            q_est = se3lib.angleaxis2quat(v, theta)

        else:
            raise ValueError("Unknown orientation parameterization: {}".format(
                model.config.ORIENTATION_PARAM
            ))

        if 'ori_epistemic_var' in result:
            ori_info['ori_epistemic_var'] = np.asarray(result['ori_epistemic_var']).reshape(-1)

    else:
        if 'ori_mean_prob' in result:
            ori_pmf = np.asarray(result['ori_mean_prob']).reshape(-1)
        else:
            ori_pmf = utils.stable_softmax(result['ori'])

        q_est, q_est_cov = se3lib.quat_weighted_avg(dataset.ori_histogram_map, ori_pmf)

        q_est = np.asarray(q_est).reshape(-1)
        ori_info['ori_pmf'] = ori_pmf
        ori_info['q_est_cov'] = q_est_cov

        if 'ori_entropy' in result:
            ori_info['ori_entropy'] = result['ori_entropy']
        if 'ori_expected_entropy' in result:
            ori_info['ori_expected_entropy'] = result['ori_expected_entropy']
        if 'ori_mutual_info' in result:
            ori_info['ori_mutual_info'] = result['ori_mutual_info']
        if 'ori_var' in result:
            ori_info['ori_var'] = np.asarray(result['ori_var']).reshape(-1)
        if 'ori_var_mean' in result:
            ori_info['ori_var_mean'] = result['ori_var_mean']

    return q_est, ori_info

#新增：专门决定调用 detect() 还是 detect_mc()
def run_model_inference(model, image, verbose=1):
    """Unified inference wrapper for detect() / detect_mc()."""
    if getattr(model.config, "MC_DROPOUT", False):
        return model.detect_mc([image], verbose=verbose)
    else:
        return model.detect([image], verbose=verbose)


def fit_GMM_to_orientation(q_map, pmf, nr_iterations, var, nr_max_modes=4):
    ''' Fits multiple quaternions to a PMF using Expectation Maximization'''

    nr_total_bins = len(pmf)
    scores = []

    # Sorting bins per probability
    pmf_sorted_indices = pmf.argsort()[::-1]

    for N in range(1, nr_max_modes):

        # 1. Initialize Gaussians
        Q_mean = np.zeros((N,4), np.float32)
        Q_var = np.ones(N, np.float32)*var
        priors = np.ones(N, np.float32)/N

        # Initialize Gaussian means by picking up the strongest bins
        check_q_mask = np.zeros_like(pmf)>0

        ptr = 0
        for k in range(N):

            # Select bin
            for i in range(ptr, nr_total_bins):
                if not check_q_mask[i]:
                    check_q_mask[i] = True
                    q_max = q_map[pmf_sorted_indices[i], :]
                    Q_mean[k, :] = q_max
                    ptr = i + 1
                    break

            # Mask out neighbours
            for i in range(nr_total_bins):
                q_i = q_map[pmf_sorted_indices[i], :]
                if not check_q_mask[i]:
                    #d_i = (1 - np.sum(q_i * q_max)) ** 2
                    d_i = (se3lib.angle_between_quats(q_i, q_max) / 180) ** 2
                    if d_i < 9 * var:
                        check_q_mask[i] = 1


        # 2. Expectation Maximization loop
        for it in range(nr_iterations):

            # Expectation step

            # Normalized angular distance
            Distances = np.asarray(se3lib.angle_between_quats(q_map, Q_mean))/180

            # Compute p(X|Theta)
            eps = 1e-18
            p_X_given_models = eps + np.divide(np.exp(np.divide(-Distances ** 2, 2.0 * Q_var)),
                                                 np.sqrt(2.0 * np.pi * Q_var))

            # Compute p(Theta|X) by applying Bayes rule
            # Get marginal likelihood
            p_X_given_models_times_priors = p_X_given_models*priors
            p_X = np.sum(p_X_given_models_times_priors, axis=1)
            p_models_given_X = p_X_given_models_times_priors/p_X[:,np.newaxis]

            # Maximization step

            # Compute weights
            W = p_models_given_X * pmf[:, np.newaxis]
            Z = np.sum(W, axis=0)
            W_n = W / Z

            # Compute average quaternions
            for k in range(N):

                q_mean_k, _ = se3lib.quat_weighted_avg(q_map, W_n[:, k])
                Q_mean[k, :] = q_mean_k
                Q_var[k] = 0
                Distances = np.asarray(se3lib.angle_between_quats(q_map,q_mean_k)/180)**2
                for i in range(nr_total_bins):
                    Q_var[k] += W_n[i, k] * Distances[i]

            # print('New mixture means:\n', Q_mean)
            # print('New mixture priors:\n', priors)
            # print('New mixture var:\n', Q_var)
            # print('\n')

            # Compute priors
            priors = Z

            if N == 1 and it == 1:
                break

        # Check model likelihood by reusing last iteration state
        score = np.sum(pmf * np.log(p_X))

        if len(scores)==0 or score > scores[-1]+0.005:
            # Update best model
            Q_mean_best = Q_mean
            Q_var_best = Q_var
            Q_priors_best = priors
            scores.append(score)
        else:
            # Stop model searching to return last state
            break

    # TODO: Sort by likelihood
    sorting_indices = Q_priors_best.argsort()[::-1]

    Q_mean_best = Q_mean_best[sorting_indices]
    Q_priors_best = Q_priors_best[sorting_indices]
    Q_var_best = Q_var_best[sorting_indices]

    print('Q priors:',Q_priors_best)
    print('Q :', Q_mean_best)
    print('Scores:', scores)

    return Q_mean_best, Q_var_best, Q_priors_best, scores

def evaluate_image(model, dataset, image_id):

    # Load pose in all formats
#修改：统一错误计算中的 shape
    loc_gt = np.asarray(dataset.load_location(image_id)).reshape(-1)
    q_gt = np.asarray(dataset.load_quaternion(image_id)).reshape(-1)
    image = dataset.load_image(image_id)
    I, I_meta, loc_encoded_gt, ori_encoded_gt = \
        net.load_image_gt(dataset, model.config, image_id)

#修改：loc 和 ori 的解析逻辑，利用前面写好的函数
    results = run_model_inference(model, image, verbose=1)  #第二次修改,为了加入MCDropout
    result = results[0]

    loc_est, loc_info = extract_location_from_result(model, result, dataset)
    q_est, ori_info = extract_orientation_from_result(model, result, dataset)

    if not model.config.REGRESS_LOC:
        loc_decoded_gt = np.asarray(
            np.asmatrix(loc_encoded_gt) * np.asmatrix(dataset.histogram_3D_map)
        ).reshape(-1)
        loc_encoded_err = np.linalg.norm(loc_decoded_gt - loc_gt)

    if not model.config.REGRESS_ORI:
        q_encoded_gt, _ = se3lib.quat_weighted_avg(dataset.ori_histogram_map, ori_encoded_gt)
        ori_encoded_err = 2 * np.arccos(
            np.abs(np.asmatrix(q_encoded_gt) * np.asmatrix(q_gt).transpose())
        ) * 180 / np.pi

    # Compute errors
    angular_err = 2 * np.arccos(np.abs(np.asmatrix(q_est) * np.asmatrix(q_gt).transpose()))
    # angular_err_in_deg = angular_err* 180 / np.pi

    loc_err = np.linalg.norm(loc_est - loc_gt)
    loc_rel_err = loc_err / np.linalg.norm(loc_gt)

    # Compute ESA score
    esa_score = loc_rel_err + angular_err

    return loc_err, angular_err, loc_rel_err, esa_score

def test_and_submit(model, dataset_virtual, dataset_real):
    """ Evaluates model on ESA challenge test-set (no labels)
    and outputs submission file in a format compatible with the ESA server (probably down by now)
    """

    # ESA API
    from submission import SubmissionWriter
    submission = SubmissionWriter()

    # TODO: Make the next 2 loops a nested loop

    # Synthetic test set
    for image_id in dataset_virtual.image_ids:

        print('Image ID:', image_id)

        image = dataset_virtual.load_image(image_id)
        info = dataset_virtual.image_info[image_id]
#修改：统一 loc/ori 解析
        results = run_model_inference(model, image, verbose=1)  #第二次修改
        result = results[0]

        loc_est, loc_info = extract_location_from_result(model, result, dataset_virtual)
        q_est, ori_info = extract_orientation_from_result(model, result, dataset_virtual)

        # Change quaternion order
        q_rect = [q_est[3], q_est[0], q_est[1], q_est[2]]

        submission.append_test(info['path'].split('/')[-1], q_rect, loc_est.tolist())

    # Real test set

    for image_id in dataset_real.image_ids:

        print('Image ID:', image_id)

        image = dataset_real.load_image(image_id)
        info = dataset_real.image_info[image_id]
#修改：统一 loc/ori 解析
        results = model.detect([image], verbose=1)
        result = results[0]

        loc_est, loc_info = extract_location_from_result(model, result, dataset_real)
        q_est, ori_info = extract_orientation_from_result(model, result, dataset_real)

        # Change quaternion order
        q_rect = [q_est[3], q_est[0], q_est[1], q_est[2]]

        submission.append_real_test(info['path'].split('/')[-1], q_rect, loc_est)

    submission.export(suffix='debug')
    print('Submission exported.')


import os
import pandas as pd
import numpy as np


def evaluate(model, dataset):
    """Evaluates model on all dataset images. Assumes all images have corresponding pose labels.
        遍历整个测试/验证集，统计平均误差并保存 CSV
        适合做定量评估
        这里已经基本实现了贝叶斯位置不确定性的统计输出，并将所有结果整合到一个CSV中
    """

    # 1. 增加一个用于存储图片名称的列表
    image_names_acc = []

    loc_err_acc = []
    loc_encoded_err_acc = []
    ori_err_acc = []
    ori_encoded_err_acc = []
    distances_acc = []
    esa_scores_acc = []

    # ++++++++++++++++++++++ 新增点 1：初始化 X, Y, Z 独立的 GT 和 误差列表 ++++++++++++++++++++++
    gt_x_acc = []
    gt_y_acc = []
    gt_z_acc = []

    loc_err_x_acc = []
    loc_err_y_acc = []
    loc_err_z_acc = []
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # ================= 修改点 1：将位置不确定性拆分为 X, Y, Z 三个列表 =================
    loc_aleatoric_unc_x_acc = []
    loc_aleatoric_unc_y_acc = []
    loc_aleatoric_unc_z_acc = []

    loc_epistemic_unc_x_acc = []
    loc_epistemic_unc_y_acc = []
    loc_epistemic_unc_z_acc = []

    loc_total_unc_x_acc = []
    loc_total_unc_y_acc = []
    loc_total_unc_z_acc = []
    # ==================================================================================

    ori_entropy_acc = []
    ori_expected_entropy_acc = []
    ori_mutual_info_acc = []
    ori_var_mean_acc = []

    # Variance used only for prob. orientation estimation
    delta = model.config.BETA / model.config.ORI_BINS_PER_DIM
    var = delta ** 2 / 12

    for image_id in dataset.image_ids:

        print('Image ID:', image_id)

        # 获取当前图片的真实文件名 (从绝对路径中提取)
        image_info = dataset.image_info[image_id]

        # 优先尝试从 'path' 中提取真实文件名 (如 img000001.jpg)
        if 'path' in image_info and image_info['path'] is not None:
            image_name = os.path.basename(image_info['path'])
        else:
            # 如果没有 path，作为备选方案尝试获取 id
            image_name = str(image_info.get('id', image_id))

        image_names_acc.append(image_name)

        # Load pose in all formats
        loc_gt = dataset.load_location(image_id)
        q_gt = dataset.load_quaternion(image_id)
        loc_gt = np.asarray(loc_gt).reshape(-1)
        q_gt = np.asarray(q_gt).reshape(-1)
        image = dataset.load_image(image_id)

        results = run_model_inference(model, image, verbose=1)

        if model.config.REGRESS_KEYPOINTS:
            # Experimental branch
            I, I_meta, loc_gt, k1_gt, k2_gt = \
                net.load_image_gt(dataset, model.config, image_id)

            loc_est = np.asarray(results[0]['loc']).reshape(-1)
            k1_est = np.asarray(results[0]['k1']).reshape(-1)
            k2_est = np.asarray(results[0]['k2']).reshape(-1)

            # Prepare keypoint matches
            P1 = np.zeros((3, 3))
            P1[2, 0] = 3.0
            P1[1, 1] = 3.0

            P2 = np.zeros((3, 3))
            P2[:, 0] = k1_est
            P2[:, 1] = k2_est
            P2[:, 2] = loc_est

            t, R = se3lib.pose_3Dto3D(np.asmatrix(P1), np.asmatrix(P2))
            q_est = se3lib.SO32quat(R.T)

            loc_info = {}
            ori_info = {}

        else:
            I, I_meta, loc_encoded_gt, ori_encoded_gt = \
                net.load_image_gt(dataset, model.config, image_id)

            result = results[0]

            loc_est, loc_info = extract_location_from_result(model, result, dataset)
            q_est, ori_info = extract_orientation_from_result(model, result, dataset)

            if not model.config.REGRESS_LOC:
                loc_decoded_gt = np.asarray(
                    np.asmatrix(loc_encoded_gt) * np.asmatrix(dataset.histogram_3D_map)
                ).reshape(-1)
                loc_encoded_err = np.linalg.norm(loc_decoded_gt - loc_gt)
                loc_encoded_err_acc.append(loc_encoded_err)

            if not model.config.REGRESS_ORI:
                ori_pmf = ori_info['ori_pmf']

                q_encoded_gt, _ = se3lib.quat_weighted_avg(
                    dataset.ori_histogram_map, ori_encoded_gt
                )
                # 修复浮点数越界导致的 NaN
                dot_prod_encoded = np.abs(np.asmatrix(q_encoded_gt) * np.asmatrix(q_gt).transpose())
                dot_prod_encoded = np.clip(dot_prod_encoded, -1.0, 1.0)
                ori_encoded_err = 2 * np.arccos(dot_prod_encoded) * 180 / np.pi
                ori_encoded_err_acc.append(ori_encoded_err)

        # 3. Angular error
        dot_prod_est = np.abs(np.asmatrix(q_est) * np.asmatrix(q_gt).transpose())
        dot_prod_est = np.clip(dot_prod_est, -1.0, 1.0)
        angular_err = 2 * np.arccos(dot_prod_est) * 180 / np.pi

        angular_err_scalar = float(np.asarray(angular_err).reshape(-1)[0])
        ori_err_acc.append(angular_err_scalar)

        # 4. Loc error
        loc_err = np.linalg.norm(loc_est - loc_gt)
        loc_err_acc.append(float(loc_err))

        print('Loc Error: ', loc_err)
        print('Ori Error: ', angular_err_scalar)

        # Compute ESA score (复用上面已经 clip 过的 dot_prod_est)
        esa_score = loc_err / np.linalg.norm(loc_gt) + 2 * np.arccos(dot_prod_est)

        esa_score_scalar = float(np.asarray(esa_score).reshape(-1)[0])
        esa_scores_acc.append(esa_score_scalar)

        # Store depth
        distances_acc.append(float(loc_gt[2]))

        # ++++++++++++++++++++++ 新增点 2：提取并保存 X, Y, Z 的 GT 坐标和绝对误差 ++++++++++++++++++++++
        gt_x_acc.append(float(loc_gt[0]))
        gt_y_acc.append(float(loc_gt[1]))
        gt_z_acc.append(float(loc_gt[2]))

        loc_err_x_acc.append(float(abs(loc_est[0] - loc_gt[0])))
        loc_err_y_acc.append(float(abs(loc_est[1] - loc_gt[1])))
        loc_err_z_acc.append(float(abs(loc_est[2] - loc_gt[2])))
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # ================= 修改点 2：分别提取并保存 X, Y, Z 的方差 =================
        if 'loc_aleatoric_var' in loc_info:
            ale_var = np.asarray(loc_info['loc_aleatoric_var']).reshape(-1)
            loc_aleatoric_unc_x_acc.append(float(ale_var[0]))
            loc_aleatoric_unc_y_acc.append(float(ale_var[1]))
            loc_aleatoric_unc_z_acc.append(float(ale_var[2]))

        if 'loc_epistemic_var' in loc_info:
            epi_var = np.asarray(loc_info['loc_epistemic_var']).reshape(-1)
            loc_epistemic_unc_x_acc.append(float(epi_var[0]))
            loc_epistemic_unc_y_acc.append(float(epi_var[1]))
            loc_epistemic_unc_z_acc.append(float(epi_var[2]))

        if 'loc_total_var' in loc_info:
            tot_var = np.asarray(loc_info['loc_total_var']).reshape(-1)
            loc_total_unc_x_acc.append(float(tot_var[0]))
            loc_total_unc_y_acc.append(float(tot_var[1]))
            loc_total_unc_z_acc.append(float(tot_var[2]))
        # ==================================================================================

        # Store orientation uncertainty
        if 'ori_entropy' in ori_info:  # 预测熵 / 姿态总不确定性
            ori_entropy_acc.append(float(ori_info['ori_entropy']))
        if 'ori_expected_entropy' in ori_info:  # 期望熵 / 姿态数据不确定性 / Aleatoric
            ori_expected_entropy_acc.append(float(ori_info['ori_expected_entropy']))
        if 'ori_mutual_info' in ori_info:  # 互信息 / 姿态模型不确定性 / Epistemic
            ori_mutual_info_acc.append(float(ori_info['ori_mutual_info']))
        if 'ori_var_mean' in ori_info:  # 概率方差均值 / 姿态波动程度
            ori_var_mean_acc.append(float(ori_info['ori_var_mean']))

    print('\n--- Evaluation Summary ---')
    print('Mean est. location error: ', np.mean(loc_err_acc))
    print('Mean est. orientation error: ', np.mean(ori_err_acc))
    print('ESA score: ', np.mean(esa_scores_acc))

    if len(loc_encoded_err_acc) > 0:
        print('Mean encoded location error: ', np.mean(loc_encoded_err_acc))
    if len(ori_encoded_err_acc) > 0:
        print('Mean encoded orientation error: ', np.mean(ori_encoded_err_acc))

    # ================= 修改点 3：分别打印 X, Y, Z 的平均不确定性 =================
    if len(loc_aleatoric_unc_x_acc) > 0:
        print(
            f'Mean location aleatoric uncertainty (X, Y, Z): {np.mean(loc_aleatoric_unc_x_acc):.5f}, {np.mean(loc_aleatoric_unc_y_acc):.5f}, {np.mean(loc_aleatoric_unc_z_acc):.5f}')
    if len(loc_epistemic_unc_x_acc) > 0:
        print(
            f'Mean location epistemic uncertainty (X, Y, Z): {np.mean(loc_epistemic_unc_x_acc):.5f}, {np.mean(loc_epistemic_unc_y_acc):.5f}, {np.mean(loc_epistemic_unc_z_acc):.5f}')
    if len(loc_total_unc_x_acc) > 0:
        print(
            f'Mean location total uncertainty (X, Y, Z): {np.mean(loc_total_unc_x_acc):.5f}, {np.mean(loc_total_unc_y_acc):.5f}, {np.mean(loc_total_unc_z_acc):.5f}')
    # ==================================================================================

    if len(ori_entropy_acc) > 0:
        print('Mean orientation entropy: ', np.mean(ori_entropy_acc))
    if len(ori_expected_entropy_acc) > 0:
        print('Mean orientation expected entropy: ', np.mean(ori_expected_entropy_acc))
    if len(ori_mutual_info_acc) > 0:
        print('Mean orientation mutual information: ', np.mean(ori_mutual_info_acc))
    if len(ori_var_mean_acc) > 0:
        print('Mean orientation variance: ', np.mean(ori_var_mean_acc))

    # ==========================================
    # 优化后的 CSV 保存逻辑：整合为一个大表
    # ==========================================

    # 基础数据字典
    results_dict = {
        'Image_Name': image_names_acc,
        'Loc_Error(m)': loc_err_acc,
        'Ori_Error(deg)': ori_err_acc,
        'ESA_Score': esa_scores_acc,
        'GT_Distance(Z)': distances_acc,
        # ++++++++++++++++++++++ 新增点 3：将 X, Y, Z 的 GT 和误差写入字典 ++++++++++++++++++++++
        'GT_X': gt_x_acc,
        'GT_Y': gt_y_acc,
        'GT_Z': gt_z_acc,
        'Loc_Error_X': loc_err_x_acc,
        'Loc_Error_Y': loc_err_y_acc,
        'Loc_Error_Z': loc_err_z_acc
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    }

    # 动态添加可选数据
    if len(loc_encoded_err_acc) > 0:
        results_dict['Loc_Encoded_Error'] = loc_encoded_err_acc
    if len(ori_encoded_err_acc) > 0:
        results_dict['Ori_Encoded_Error'] = np.asarray(ori_encoded_err_acc).reshape(-1).tolist()

    # ================= 修改点 4：将 X, Y, Z 独立作为列写入 CSV =================
    if len(loc_aleatoric_unc_x_acc) > 0:
        results_dict['Loc_Aleatoric_Unc_X'] = loc_aleatoric_unc_x_acc
        results_dict['Loc_Aleatoric_Unc_Y'] = loc_aleatoric_unc_y_acc
        results_dict['Loc_Aleatoric_Unc_Z'] = loc_aleatoric_unc_z_acc

    if len(loc_epistemic_unc_x_acc) > 0:
        results_dict['Loc_Epistemic_Unc_X'] = loc_epistemic_unc_x_acc
        results_dict['Loc_Epistemic_Unc_Y'] = loc_epistemic_unc_y_acc
        results_dict['Loc_Epistemic_Unc_Z'] = loc_epistemic_unc_z_acc

    if len(loc_total_unc_x_acc) > 0:
        results_dict['Loc_Total_Unc_X'] = loc_total_unc_x_acc
        results_dict['Loc_Total_Unc_Y'] = loc_total_unc_y_acc
        results_dict['Loc_Total_Unc_Z'] = loc_total_unc_z_acc
    # ==================================================================================

    if len(ori_entropy_acc) > 0:
        results_dict['Ori_Entropy'] = ori_entropy_acc
    if len(ori_expected_entropy_acc) > 0:
        results_dict['Ori_Expected_Entropy'] = ori_expected_entropy_acc
    if len(ori_mutual_info_acc) > 0:
        results_dict['Ori_Mutual_Info'] = ori_mutual_info_acc
    if len(ori_var_mean_acc) > 0:
        results_dict['Ori_Var_Mean'] = ori_var_mean_acc

    # 转换为 DataFrame
    df_results = pd.DataFrame(results_dict)

    # 获取保存路径 (优先尝试保存在模型的 log 文件夹下，否则保存在当前运行目录)
    try:
        save_dir = model.model_dir
    except AttributeError:
        save_dir = "."

    # 获取数据集名称，用于命名文件
    dataset_name = getattr(dataset, 'name', 'dataset')
    csv_filename = f"{dataset_name}_evaluation_results.csv"

    save_path = os.path.join(save_dir, csv_filename)

    # 保存为 CSV
    df_results.to_csv(save_path, index=False)
    print(f"\n[INFO] All evaluation results successfully saved to: {save_path}")


def extract_ood_uncertainty(model, dataset, output_filename="real_ood_uncertainty.csv"):
    """
    用于 real 文件夹（有标签）的不确定性提取与误差评估。
    与 evaluate 逻辑一致，额外输出 Epistemic 不确定性。
    """
    print(f"\n[INFO] 开始提取 real 数据集的不确定性与误差评估...")

    image_names_acc = []
    loc_err_acc = []
    ori_err_acc = []
    esa_scores_acc = []
    distances_acc = []

    gt_x_acc, gt_y_acc, gt_z_acc = [], [], []
    loc_err_x_acc, loc_err_y_acc, loc_err_z_acc = [], [], []

    loc_aleatoric_unc_x_acc, loc_aleatoric_unc_y_acc, loc_aleatoric_unc_z_acc = [], [], []
    loc_epistemic_unc_x_acc, loc_epistemic_unc_y_acc, loc_epistemic_unc_z_acc = [], [], []
    loc_total_unc_x_acc,     loc_total_unc_y_acc,     loc_total_unc_z_acc     = [], [], []

    ori_entropy_acc        = []
    ori_expected_entropy_acc = []
    ori_mutual_info_acc    = []
    ori_var_mean_acc       = []

    for image_id in dataset.image_ids:

        # ── 获取图片名称 ──────────────────────────────────────────────
        image_info = dataset.image_info[image_id]
        if 'path' in image_info and image_info['path'] is not None:
            image_name = os.path.basename(image_info['path'])
        else:
            image_name = str(image_info.get('id', image_id))

        print(f'Processing Real Image: {image_name}')
        image_names_acc.append(image_name)

        # ── 加载 GT（与 evaluate 完全一致）────────────────────────────
        loc_gt = np.asarray(dataset.load_location(image_id)).reshape(-1)
        q_gt   = np.asarray(dataset.load_quaternion(image_id)).reshape(-1)

        # ── 加载图像并推理 ────────────────────────────────────────────
        image   = dataset.load_image(image_id)
        print(f"  [DEBUG] image.shape = {image.shape}, dtype = {image.dtype}")
        if image.ndim == 4:
            image = image[..., 0]  # 取第一个 stack，shape: (1200, 1920, 3)
        results = run_model_inference(model, image, verbose=0)
        result  = results[0]

        # ── 提取位置与姿态估计 ────────────────────────────────────────
        loc_est, loc_info = extract_location_from_result(model, result, dataset)
        q_est,   ori_info = extract_orientation_from_result(model, result, dataset)

        # ── 位置误差 ──────────────────────────────────────────────────
        loc_err = float(np.linalg.norm(loc_est - loc_gt))
        loc_err_acc.append(loc_err)
        print(f'  Loc Error: {loc_err:.4f} m')

        # ── 姿态误差 ──────────────────────────────────────────────────
        dot_prod = np.clip(
            float(np.abs(np.asmatrix(q_est) * np.asmatrix(q_gt).T)),
            -1.0, 1.0
        )
        angular_err = float(2 * np.arccos(dot_prod) * 180 / np.pi)
        ori_err_acc.append(angular_err)
        print(f'  Ori Error: {angular_err:.4f} deg')

        # ── ESA Score ─────────────────────────────────────────────────
        esa_score = loc_err / np.linalg.norm(loc_gt) + 2 * np.arccos(dot_prod)
        esa_scores_acc.append(float(esa_score))

        # ── GT 坐标与分轴误差 ─────────────────────────────────────────
        distances_acc.append(float(loc_gt[2]))
        gt_x_acc.append(float(loc_gt[0]))
        gt_y_acc.append(float(loc_gt[1]))
        gt_z_acc.append(float(loc_gt[2]))
        loc_err_x_acc.append(float(abs(loc_est[0] - loc_gt[0])))
        loc_err_y_acc.append(float(abs(loc_est[1] - loc_gt[1])))
        loc_err_z_acc.append(float(abs(loc_est[2] - loc_gt[2])))

        # ── 位置不确定性（X/Y/Z 分轴）────────────────────────────────
        if 'loc_aleatoric_var' in loc_info:
            v = np.asarray(loc_info['loc_aleatoric_var']).reshape(-1)
            loc_aleatoric_unc_x_acc.append(float(v[0]))
            loc_aleatoric_unc_y_acc.append(float(v[1]))
            loc_aleatoric_unc_z_acc.append(float(v[2]))

        if 'loc_epistemic_var' in loc_info:
            v = np.asarray(loc_info['loc_epistemic_var']).reshape(-1)
            loc_epistemic_unc_x_acc.append(float(v[0]))
            loc_epistemic_unc_y_acc.append(float(v[1]))
            loc_epistemic_unc_z_acc.append(float(v[2]))

        if 'loc_total_var' in loc_info:
            v = np.asarray(loc_info['loc_total_var']).reshape(-1)
            loc_total_unc_x_acc.append(float(v[0]))
            loc_total_unc_y_acc.append(float(v[1]))
            loc_total_unc_z_acc.append(float(v[2]))

        # ── 姿态不确定性 ──────────────────────────────────────────────
        if 'ori_entropy'          in ori_info: ori_entropy_acc.append(float(ori_info['ori_entropy']))
        if 'ori_expected_entropy' in ori_info: ori_expected_entropy_acc.append(float(ori_info['ori_expected_entropy']))
        if 'ori_mutual_info'      in ori_info: ori_mutual_info_acc.append(float(ori_info['ori_mutual_info']))
        if 'ori_var_mean'         in ori_info: ori_var_mean_acc.append(float(ori_info['ori_var_mean']))

    # ── 汇总打印 ──────────────────────────────────────────────────────
    print('\n--- Real Dataset Evaluation Summary ---')
    print(f'Mean Loc Error : {np.mean(loc_err_acc):.4f} m')
    print(f'Median Loc Error : {np.median(loc_err_acc):.4f} m')
    print(f'Mean Ori Error : {np.mean(ori_err_acc):.4f} deg')
    print(f'Median Ori Error : {np.median(ori_err_acc):.4f} deg')
    print(f'Mean ESA Score : {np.mean(esa_scores_acc):.4f}')

    # ── 构建 CSV ──────────────────────────────────────────────────────
    results_dict = {
        'Image_Name'      : image_names_acc,
        'Loc_Error(m)'    : loc_err_acc,
        'Ori_Error(deg)'  : ori_err_acc,
        'ESA_Score'       : esa_scores_acc,
        'GT_Distance(Z)'  : distances_acc,
        'GT_X'            : gt_x_acc,
        'GT_Y'            : gt_y_acc,
        'GT_Z'            : gt_z_acc,
        'Loc_Error_X'     : loc_err_x_acc,
        'Loc_Error_Y'     : loc_err_y_acc,
        'Loc_Error_Z'     : loc_err_z_acc,
    }

    if loc_aleatoric_unc_x_acc:
        results_dict.update({
            'Loc_Aleatoric_Unc_X': loc_aleatoric_unc_x_acc,
            'Loc_Aleatoric_Unc_Y': loc_aleatoric_unc_y_acc,
            'Loc_Aleatoric_Unc_Z': loc_aleatoric_unc_z_acc,
        })
    if loc_epistemic_unc_x_acc:
        results_dict.update({
            'Loc_Epistemic_Unc_X': loc_epistemic_unc_x_acc,
            'Loc_Epistemic_Unc_Y': loc_epistemic_unc_y_acc,
            'Loc_Epistemic_Unc_Z': loc_epistemic_unc_z_acc,
        })
    if loc_total_unc_x_acc:
        results_dict.update({
            'Loc_Total_Unc_X': loc_total_unc_x_acc,
            'Loc_Total_Unc_Y': loc_total_unc_y_acc,
            'Loc_Total_Unc_Z': loc_total_unc_z_acc,
        })
    if ori_entropy_acc:         results_dict['Ori_Entropy']          = ori_entropy_acc
    if ori_expected_entropy_acc:results_dict['Ori_Expected_Entropy'] = ori_expected_entropy_acc
    if ori_mutual_info_acc:     results_dict['Ori_Mutual_Info']      = ori_mutual_info_acc
    if ori_var_mean_acc:        results_dict['Ori_Var_Mean']         = ori_var_mean_acc

    # ── 保存 CSV ──────────────────────────────────────────────────────
    try:
        save_dir = model.model_dir
    except AttributeError:
        save_dir = "."

    save_path = os.path.join(save_dir, output_filename)
    pd.DataFrame(results_dict).to_csv(save_path, index=False)
    print(f"[INFO] Real 数据集评估结果已保存至: {save_path}")
    return save_path


def detect_dataset(model, dataset, nr_images):
    """ Tests model on N random images of the dataset
     and shows the results.
     参数设置为test时，随机抽取 10 张图，逐张打印结果并可视化
    """

    # Variance used only for prob. orientation estimation
    delta = model.config.BETA / model.config.ORI_BINS_PER_DIM
    var = delta ** 2 / 12

    for i in range(nr_images):
        image_id = random.choice(dataset.image_ids)

        # Load pose in all formats   已经修改统一 GT shape
        loc_gt = np.asarray(dataset.load_location(image_id)).reshape(-1)
        q_gt = np.asarray(dataset.load_quaternion(image_id)).reshape(-1)

        I, I_meta, loc_encoded_gt, ori_encoded_gt = \
            net.load_image_gt(dataset, model.config, image_id)
        image_ori = dataset.load_image(image_id)

        info = dataset.image_info[image_id]

        # Run detection
        #已经修改：loc/ori 解析全部替换
        result = run_model_inference(model, image_ori, verbose=1)[0]  #第二次修改

        loc_est, loc_info = extract_location_from_result(model, result, dataset)
        q_est, ori_info = extract_orientation_from_result(model, result, dataset)

        if not model.config.REGRESS_LOC:
            loc_decoded_gt = np.asarray(
                np.asmatrix(loc_encoded_gt) * np.asmatrix(dataset.histogram_3D_map)
            ).reshape(-1)
            loc_encoded_err = np.linalg.norm(loc_decoded_gt - loc_gt)

        if not model.config.REGRESS_ORI:
            ori_pmf = ori_info['ori_pmf']


            # Multimodal estimation
            # Uncomment this block to try the EM framework
            # nr_EM_iterations = 5
            # Q_mean, Q_var, Q_priors, model_scores = fit_GMM_to_orientation(dataset.ori_histogram_map, ori_pmf, nr_EM_iterations, var)
            # print('Multimodal errors',2 * np.arccos(np.abs(np.asmatrix(Q_mean) * np.asmatrix(q_gt).transpose())) * 180 / np.pi)
            #
            # q_est_1 = Q_mean[0, :]
            # q_est_2 = Q_mean[1, :]
            # utils.polar_plot(q_est_1, q_est_2)

        # Compute Errors
        angular_err = 2 * np.arccos(np.abs(np.asmatrix(q_est) * np.asmatrix(q_gt).transpose())) * 180 / np.pi
        loc_err = np.linalg.norm(loc_est - loc_gt)

        print('GT location: ', loc_gt)
        print('Est location: ', loc_est)

        if getattr(model.config, "BAYESIAN_LOC", False):
            if 'loc_logvar' in loc_info:
                print('Est location logvar: ', loc_info['loc_logvar'])
            if 'loc_aleatoric_var' in loc_info:
                print('Est location aleatoric var: ', loc_info['loc_aleatoric_var'])

        print('Processed Image:', info['path'])
        print('Est orientation: ', q_est)
        print('GT_orientation: ', q_gt)

        print('Location error: ', loc_err)
        print('Angular error: ', angular_err)


        # Visualize PMFs
        if not model.config.REGRESS_ORI:

            nr_bins_per_dim = model.config.ORI_BINS_PER_DIM
            utils.visualize_weights(ori_encoded_gt,ori_pmf,nr_bins_per_dim)

        # Show image
        fig, (ax_1, ax_2) = plt.subplots(1,2,figsize=(12, 8))
        ax_1.imshow(image_ori)
        ax_1.set_xticks([])
        ax_1.set_yticks([])
        ax_2.imshow(image_ori)
        ax_2.set_xticks([])
        ax_2.set_yticks([])

        height_ori = np.shape(image_ori)[0]
        width_ori = np.shape(image_ori)[1]

        # Recover focal lengths
        fx = dataset.camera.fx
        fy = dataset.camera.fy

        K = np.matrix([[fx,0,width_ori/2],[0,fy,height_ori/2],[0,0,1]])

        # Speed  labels expresses q_obj_cam whereas
        # Urso labels expresses q_cam_obj
        if dataset.name == 'Speed':
            q_est = se3lib.quat_inv(q_est)
            q_gt = se3lib.quat_inv(q_gt)

        utils.visualize_axes(ax_1, q_gt, loc_gt, K, 100)
        utils.visualize_axes(ax_2, q_est, loc_est, K, 100)

        utils.polar_plot(q_gt,q_est)

        # Location overlap visualization
        fig, ax = plt.subplots()
        ax.imshow(image_ori)

        # Project 3D coords for visualization
        x_est = loc_est[0] / loc_est[2]
        y_est = loc_est[1] / loc_est[2]

        x_gt = loc_gt[0] / loc_gt[2]
        y_gt = loc_gt[1] / loc_gt[2]

        if not model.config.REGRESS_LOC:
            x_decoded_gt = loc_decoded_gt[0] / loc_decoded_gt[2]
            y_decoded_gt = loc_decoded_gt[1] / loc_decoded_gt[2]

            circ = Circle((x_decoded_gt * fx + width_ori / 2, height_ori / 2 + y_decoded_gt * fy), 7, facecolor='b',
                          label='encoded')
            ax.add_patch(circ)

        # Plot locations
        circ_gt = Circle((x_gt*fx + width_ori/2, height_ori/2 + y_gt*fy), 15, facecolor='r', label='gt')
        ax.add_patch(circ_gt)

        circ = Circle((x_est*fx + width_ori/2, height_ori/2 + y_est*fy), 10, facecolor='g',label='pred')
        ax.add_patch(circ)

        ax.legend(loc='upper right', shadow=True, fontsize='x-small')
        plt.show()

def detect_video(model, dataset, video_path):
    ''' Experimental'''

    import cv2

    # Video capture
    vcapture = cv2.VideoCapture(video_path)
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)

    # Camera projection mat
    width = dataset.camera.width/2  # TODO: work on original image size not 1/2
    height = dataset.camera.height/2
    fov_horizontal = np.pi / 2
    fx = width / (2 * np.tan(dataset.camera.fov_x / 2))
    fy = - height / (2 * np.tan(dataset.camera.fov_y / 2))
    K = np.matrix([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]])

    R_cam_unreal = np.matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    # Define codec and create video writer
    vwriter = cv2.VideoWriter("video_real.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (int(width), int(height)))

    count = 0
    pose_est_acc = []
    success = True
    while success:
        print("frame: ", count)
        count += 1
        # Read next image
        success, image = vcapture.read()
        if success and count>16900:
            # OpenCV returns images as BGR, convert to RGB
            image = image[..., ::-1]
            image = image[:,1:-150,:] # crop
            image = np.pad(image, [(400, 400), (400, 400), (0, 0)], mode='constant', constant_values=0)
            image[:,:,0] = 0.21*image[:,:,0]+0.72*image[:,:,1]+0.07*image[:,:,2]
            image[:, :, 1] = image[:,:,0]
            image[:, :, 2] = image[:, :, 0]

            # Resize to network input shape
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=model.config.IMAGE_MIN_DIM,
                min_scale=model.config.IMAGE_MIN_SCALE,
                max_dim=model.config.IMAGE_MAX_DIM,
                mode=model.config.IMAGE_RESIZE_MODE)

            # Detect objects
#已修改：统一解析
            result = run_model_inference(model, image, verbose=0)[0]      #第二次修改

            loc_est, loc_info = extract_location_from_result(model, result, dataset)
            q_est, ori_info = extract_orientation_from_result(model, result, dataset)

            z = loc_est[2]
            x = loc_est[0]
            y = loc_est[1]
            print(str(z) + " " + str(x) + " " + str(y))

            # Recover Unreal orientation: R_wo
            R_co = se3lib.quat2SO3(q_est)
            R_co = R_cam_unreal.T * R_co
            R_wc = se3lib.euler2SO3_unreal(0, 0, 0)
            R_wo = R_wc*R_co
            roll, pitch, yaw = se3lib.SO32euler(R_wo)
            #
            print(str(-pitch) + " " + str(yaw) + " " + str(-roll))

            # Stack frame gt
            pose_est = np.array([loc_est[2], loc_est[0], loc_est[1], -pitch, yaw, -roll])
            pose_est_acc.append(pose_est)

            # Crop and resize image to match original input size
            margin = (model.config.IMAGE_MAX_DIM - 480) // 2
            image = molded_image[margin:model.config.IMAGE_MAX_DIM-margin, :, :]

            # Show image
            #fig, ax_1 = plt.subplots(1, 1, figsize=(12, 8))

            utils.plot_axes(image, q_est, loc_est, K, 5.0)
            # ax_1.imshow(image)
            # ax_1.set_xticks([])
            # ax_1.set_yticks([])
#已修改：保留 orientation PMF 可视化，否则如果以后切到 REGRESS_ORI=True 会直接炸。
            if not model.config.REGRESS_ORI:
                ori_pmf = ori_info['ori_pmf']
                nr_bins_per_dim = model.config.ORI_BINS_PER_DIM
                utils.visualize_weights(ori_pmf, ori_pmf, nr_bins_per_dim)

            # plt.show(block=True)
            # Add image to video writer
            vwriter.write(image)

        if count > 17200:
            success = False

    vwriter.release()

    # Connect to simulator and load estimated poses

    # from unrealcv.automation import UE4Binary
    # from unrealcv.util import read_png, read_npy
    # from unrealcv import client
    #
    # client.connect()
    #
    # # Define codec and create video writer
    # vwriter2 = cv2.VideoWriter("video_virtual.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (1280, 960))
    #
    # # Rotation between reference frames
    # # Set up camera
    # command = 'vset /camera/0/location ' + str(0) + " " + str(0) + " " + str(0)
    # client.request(command)
    # command = 'vset /camera/0/rotation ' + str(0) + " " + str(0) + " " + str(0)
    # client.request(command)
    #
    # object_name = 'Soyuz_HP_10'
    # object_set_loc_command_prefix = 'vset /object/' + object_name + '/location '
    # object_set_ori_command_prefix = 'vset /object/' + object_name + '/rotation '
    #
    # for pose_est in pose_est_acc:
    #
    #     # Translate object
    #     command = object_set_loc_command_prefix + str(pose_est[0]*100.0) + " " + str(pose_est[1]*100.0) + " " + str(pose_est[2]*100.0)
    #     client.request(command)
    #
    #     # Rotate object
    #     command = object_set_ori_command_prefix + str(pose_est[3]) + " " + str(pose_est[4]) + " " + str(pose_est[5])
    #     client.request(command)
    #
    #     # Load and save rgb
    #     res = client.request('vget /camera/0/lit png')
    #     im = read_png(res)
    #
    #     # Convert to opencv and record video frame
    #     img_cv = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #     vwriter2.write(img_cv)
    #
    # vwriter2.release()

def train(model, dataset_train, dataset_val):
    """Train the model."""

    model.config.STEPS_PER_EPOCH = min(1000,int(len(dataset_train.image_ids)/model.config.BATCH_SIZE))

    # Write config to disk
    config_filename = 'config_' + str(model.epoch) + '.json'
    config_filepath = os.path.join(model.log_dir, config_filename)
    model.config.write_to_file(config_filepath)

    print("Training")
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=model.config.LEARNING_RATE,
        epochs=model.config.EPOCHS,
        layers='all'           ###重要！！！在这里修改要训练的部分，源函数在model.py中
    )

def apply_args_to_config(config, args):
    """Use Args in pose_estimator.py to override defaults in config.py."""

    # 基本参数
    config.NAME = args.dataset
    config.BACKBONE = args.backbone
    config.EPOCHS = args.epochs
    config.LEARNING_RATE = args.learn_rate
    config.BOTTLENECK_WIDTH = args.bottleneck
    config.BRANCH_SIZE = args.branch_size
    config.NR_DENSE_LAYERS = 1
    config.OPTIMIZER = "SGD"

    # 开关 / 增强
    config.ROT_AUG = args.rot_aug
    config.ROT_IMAGE_AUG = args.rot_image_aug
    config.SIM2REAL_AUG = args.sim2real
    config.CLR = args.clr
    config.F16 = args.f16

    # 任务配置
    config.REGRESS_ORI = args.regress_ori
    config.REGRESS_LOC = args.regress_loc
    config.REGRESS_KEYPOINTS = args.regress_keypoints
    config.BAYESIAN_LOC = args.bayesian_loc

    # MC Dropout 配置
    config.MC_DROPOUT = args.mc_dropout
    config.DROPOUT_RATE = args.dropout_rate
    config.MC_SAMPLES = args.mc_samples

    # 朝向配置
    config.ORIENTATION_PARAM = args.ori_param
    config.ORI_BINS_PER_DIM = args.ori_resolution

    # 损失权重
    config.LOSS_WEIGHTS["loc_loss"] = args.loc_weight
    config.LOSS_WEIGHTS["ori_loss"] = args.ori_weight

    return config
############################################################
#  Main
############################################################
if __name__ == '__main__':
    # ==========================================================================================
    #  内置参数配置区域 (无需命令行参数)
    # ==========================================================================================
    class Args:
        # 运行模式: 'train', 'evaluate', 'test', 'submit' 'extract_ood'
        command = "extract_ood"

        # 数据集名称: 'soyuz_hard', 'dragon_hard', 'speed'
        dataset = "speed"

        # 权重文件:
        # 1. 具体路径 (e.g., "models/logs/weights.h5")
        # 2. "coco" (使用 COCO 预训练权重)
        # 3. "imagenet" (使用 ImageNet 预训练权重)
        # 4. "last" (自动寻找最后一次训练的权重)
        # 5. 直接写训练出的权重路径
        #weights = "models/logs/speed20251225T1506/weights_best.h5"
        #weights = "coco"
        weights = "models/logs/speed20260413T1059/weights_best.h5"
        # 基础网络架构
        backbone = "resnet50"

        # 训练参数
        epochs = 96
        learn_rate = 1e-5
        batch_size = 2

        # 模型结构参数
        image_scale = 1.0
        bottleneck = 32
        branch_size = 1024

        # 数据增强 / 功能开关
        rot_aug = False
        rot_image_aug = False
        sim2real = False
        clr = False
        f16 = True
        square_image = False

        # 任务配置
        bayesian_loc = False          #LocationHead输出均值与方差
        regress_ori = False           #False = 分类朝向, True = 回归朝向
        regress_loc = True            #回归位置
        regress_keypoints = False

        # MC Dropout
        mc_dropout = True
        dropout_rate = 0.2
        mc_samples = 10
        use_mc_in_video = False

        # 损失函数权重
        ori_weight = 2
        loc_weight = 1

        # 朝向参数
        ori_param = 'quaternion'
        ori_resolution = 16

        # 路径配置
        logs = DEFAULT_LOGS_DIR
        image = None
        video = None

    args = Args()

    dataset_dir = os.path.join(DATA_DIR, args.dataset)

    print("Command: ", args.command)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print("Weights: ", args.weights)

    assert args.ori_param in OrientationParamOptions

    # ------------------------------------------------------------------------------------------
    # Build config from defaults in config.py, then override by Args
    # ------------------------------------------------------------------------------------------
    config = Config()
    config = apply_args_to_config(config, args)

    # Set image resize mode
    if args.square_image:
        config.IMAGE_RESIZE_MODE = 'square'
    else:
        config.IMAGE_RESIZE_MODE = 'pad64'

    # Determine original image dimensions based on dataset
    if args.dataset == "speed":
        width_original = speed.Camera.width
        height_original = speed.Camera.height
    else:
        width_original = urso.Camera.width
        height_original = urso.Camera.height

    config.IMAGE_MAX_DIM = round(width_original * args.image_scale)

    if config.IMAGE_MAX_DIM % 64 > 0:
        raise Exception("Scale problem. Image maximum dimension must be dividable by 2 at least 6 times.")

    # assumes height < width
    height_scaled = round(height_original * args.image_scale)
    if height_scaled % 64 > 0:
        config.IMAGE_MIN_DIM = height_scaled - height_scaled % 64 + 64
    else:
        config.IMAGE_MIN_DIM = height_scaled

    # 如果从头训练灰度图，可在这里改
    # if args.dataset == "speed":
    #     config.NR_IMAGE_CHANNELS = 1

    # Batch size settings
    if args.command == "train":
        config.IMAGES_PER_GPU = args.batch_size
    else:
        config.IMAGES_PER_GPU = 1

    # Recompute derived attributes after all overrides
    config.update()
    config.display()

    # ------------------------------------------------------------------------------------------
    # Create model
    # ------------------------------------------------------------------------------------------
    if args.command == "train":
        model = net.UrsoNet(mode="training", config=config, model_dir=args.logs)
    else:
        model = net.UrsoNet(mode="inference", config=config, model_dir=args.logs)

    # ------------------------------------------------------------------------------------------
    # Select weights file
    # ------------------------------------------------------------------------------------------
    weights_path = ""

    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)

    elif args.weights.lower() == "last":
        _, weights_path = model.find_last()

    elif args.weights.lower() == "imagenet":
        weights_path = model.get_imagenet_weights(config.BACKBONE)

    elif args.weights.lower() in ['soyuz_hard', 'dragon_hard', 'speed']:
        weights_path = model.get_urso_weights(args.weights)

    elif args.weights.lower() != "none":
        weights_path = args.weights
        if not os.path.exists(weights_path):
            _, weights_path = model.get_last_checkpoint(args.weights)

    print(f"Loading weights from: {weights_path}")
    print("CWD:", os.getcwd())

    if weights_path:
        weights_path = os.path.abspath(weights_path)

    print("weights_path (abs):", weights_path,
          "exists:", os.path.exists(weights_path) if weights_path else False)

    # ------------------------------------------------------------------------------------------
    # Load weights
    # ------------------------------------------------------------------------------------------
    if args.weights.lower() == "coco":
        model.load_weights(weights_path, None, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"
        ])

    elif args.weights.lower() == "imagenet":
        model.load_weights(weights_path, None, by_name=True)

    elif args.weights.lower() in ['soyuz_hard', 'dragon_hard', 'speed']:
        model.load_weights(weights_path, None, by_name=True)

    elif args.weights.lower() != "none":
        if weights_path and os.path.exists(weights_path):
            model.load_weights(weights_path, None, by_name=True)
        else:
            print("WARNING: Weights file not found, starting from scratch or check path.")

    # ------------------------------------------------------------------------------------------
    # Run command
    # ------------------------------------------------------------------------------------------
    if args.command == "train":
        if args.dataset != "speed":
            dataset_train = urso.Urso()
            dataset_train.load_dataset(dataset_dir, model.config, "train")

            dataset_val = urso.Urso()
            dataset_val.load_dataset(dataset_dir, model.config, "val")
        else:
            dataset_train = speed.Speed()
            dataset_train.load_dataset(dataset_dir, model.config, "train_no_val")

            dataset_val = speed.Speed()
            dataset_val.load_dataset(dataset_dir, model.config, "val")

        train(model, dataset_train, dataset_val)

    elif args.command == "test":
        if args.video:
            dataset = urso.Urso()
            dataset.load_dataset(dataset_dir, config, "test")
            detect_video(model, dataset, args.video)
        else:
            if args.dataset != "speed":
                dataset = urso.Urso()
                dataset.load_dataset(dataset_dir, config, "test")
            else:
                dataset = speed.Speed()
                dataset.load_dataset(dataset_dir, config, "my_test")      #这里可以在my_test子集中测试了

            detect_dataset(model, dataset, 10)

    elif args.command == "evaluate":
        if args.dataset != "speed":
            dataset_test = urso.Urso()
            dataset_test.load_dataset(dataset_dir, config, "test")
        else:
            dataset_test = speed.Speed()
            dataset_test.load_dataset(dataset_dir, config, "my_test")    #同理，这里可以在my_test子集中测试了

        evaluate(model, dataset_test)

    elif args.command == "extract_ood":
        assert args.dataset == "speed"
        # 加载 real 文件夹的数据集
        dataset_real = speed.Speed()
        # 注意：这里传入 "real" 或 "real_test"，取决于你在 speed.py 里是怎么定义加载无标签数据的
        dataset_real.load_dataset(dataset_dir, config, "real")
        # 调用我们刚写的新函数
        extract_ood_uncertainty(model, dataset_real, output_filename="real_ood_uncertainty.csv")

    elif args.command == "submit":
        assert args.dataset == "speed"

        dataset_real = speed.Speed()
        dataset_real.load_dataset(dataset_dir, config, "real_test")

        dataset_virtual = speed.Speed()
        dataset_virtual.load_dataset(dataset_dir, config, "test")

        test_and_submit(model, dataset_virtual, dataset_real)

    else:
        print("wrong command. Please check Args.command")
