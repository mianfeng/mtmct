//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <sstream>
// for setprecision
#include <chrono>
#include <iomanip>

#include "include/jde_detector.h"
#include "include/lapjv.h"
#include "include/mtmct_tracker.h"
#include "include/reid_detector.h"
#include "spdlog/cfg/env.h"
#include "spdlog/spdlog.h"
using namespace paddle_infer;

namespace PaddleDetection {

MtmctTracker::MtmctTracker(void)
    : maxLostimes(30), activeCameraresultPools(), lostCameraresultPools() {}

// 匈牙利算法实现最大权匹配
/**
 * 对给定的成本矩阵进行线性分配最小成本匹配。
 *
 * @param cost 用于匹配的成本矩阵。
 * @param cost_limit 允许的最大成本匹配。
 * @param matches 指向Match对象的指针，用于存储最小成本匹配。
 * @param mismatch_row 指向向量的指针，用于存储未匹配行的索引。
 * @param mismatch_col 指向向量的指针，用于存储未匹配列的索引。
 *
 * @throws 无。
 */
void MtmctTracker::linear_assignment(const cv::Mat& cost, float cost_limit,
                                     Match* matches,
                                     std::vector<int>* mismatch_row,
                                     std::vector<int>* mismatch_col) {
    matches->clear();
    mismatch_row->clear();
    mismatch_col->clear();
    if (cost.empty()) {
        for (int i = 0; i < cost.rows; ++i) mismatch_row->push_back(i);
        for (int i = 0; i < cost.cols; ++i) mismatch_col->push_back(i);
        return;
    }

    float opt = 0;
    cv::Mat x(cost.rows, 1, CV_32S);
    cv::Mat y(cost.cols, 1, CV_32S);

    lapjv_internal(cost, true, cost_limit, reinterpret_cast<int*>(x.data),
                   reinterpret_cast<int*>(y.data));

    for (int i = 0; i < x.rows; ++i) {
        int j = *x.ptr<int>(i);
        if (j >= 0)
            matches->insert({i, j});
        else
            mismatch_row->push_back(i);
    }

    for (int i = 0; i < y.rows; ++i) {
        int j = *y.ptr<int>(i);
        if (j < 0) mismatch_col->push_back(i);
    }

    return;
}

// 计算向量的范数，返回行向量
/**
 * 计算给定矩阵的范数。
 *
 * @param matrix 要计算范数的矩阵
 *
 * @return 计算出的矩阵范数
 *
 * @throws ErrorType 可能引发的错误的描述
 */
cv::Mat MtmctTracker::computeNorm(const cv::Mat& matrix) {
    cv::Mat norm;
    cv::reduce(matrix.mul(matrix), norm, 1, cv::REDUCE_SUM);
    cv::sqrt(norm, norm);
    return norm;
}
// 计算余弦相似度矩阵
/**
 * 计算两个输入矩阵之间的余弦矩阵。
 *
 * @param matrixCurrent 当前矩阵。
 * @param matrixPrevious 先前矩阵。
 *
 * @return 余弦矩阵，如果输入矩阵为空或特征长度不相等，则返回空矩阵。
 *
 * @throws 无。
 */
cv::Mat MtmctTracker::cosineMatrix(const cv::Mat& matrixCurrent,
                                   const cv::Mat& matrixPrevious) {
    if (matrixCurrent.empty() || matrixPrevious.empty() ||
        matrixCurrent.cols != matrixPrevious.cols) {
        // 输入矩阵为空或者特征长度不相等，返回空矩阵
        return cv::Mat();
    }

    // 归一化操作
    cv::Mat normCurrent = computeNorm(matrixCurrent);
    cv::Mat normPrevious = computeNorm(matrixPrevious);
    cv::Mat normalizedMatrixCurrent =
        matrixCurrent / cv::repeat(normCurrent, 1, matrixCurrent.cols);
    cv::Mat normalizedMatrixPrevious =
        matrixPrevious / cv::repeat(normPrevious, 1, matrixPrevious.cols);

    // 计算矩阵的点积，得到一个矩阵
    cv::Mat dotProductMatrix = matrixCurrent * matrixPrevious.t();

    // 计算余弦相似度矩阵
    cv::Mat similarityMatrix =
        dotProductMatrix / (normCurrent * normPrevious.t());
    std::cout << 1 - similarityMatrix << std::endl;

    return 1 - similarityMatrix;
}
void MtmctTracker::mtmct(CameraResult& cameraResult1,
                         CameraResult& cameraResult2,
                         std::vector<float>& initObjects,
                         std::vector<Match>& matchResults) {
    std::vector<std::vector<float>> matrixCam1;
    std::vector<std::vector<float>> matrixCam2;
    Match matchResult1;
    Match matchResult2;
    std::vector<int> mismatchCurrent;
    std::vector<int> mismatchPrevious;
    cv::Mat matObject(1, initObjects.size(), CV_32F);
    for (int i = 0; i < matObject.rows; ++i) {
        for (int j = 0; j < matObject.cols; ++j) {
            matObject.at<float>(i, j) = initObjects[j];
        }
    }
    if (!cameraResult1.results.empty()) {
        for (const auto& itemCam1 : cameraResult1.results) {
            matrixCam1.push_back(itemCam1.feat);
        }
        cv::Mat matCam1(matrixCam1.size(), matrixCam1[0].size(), CV_32F);
        for (int i = 0; i < matCam1.rows; ++i) {
            for (int j = 0; j < matCam1.cols; ++j) {
                matCam1.at<float>(i, j) = matrixCam1[i][j];
            }
        }
        cv::Mat cam1simMatrix = cosineMatrix(matCam1, matObject);

        linear_assignment(cam1simMatrix, 0.7f, &matchResult1, &mismatchCurrent,
                          &mismatchPrevious);
    }
    if (!cameraResult2.results.empty()) {
        for (const auto& itemCam2 : cameraResult2.results) {
            matrixCam2.push_back(itemCam2.feat);
        }
        cv::Mat matCam2(matrixCam2.size(), matrixCam2[0].size(), CV_32F);
        for (int i = 0; i < matCam2.rows; ++i) {
            for (int j = 0; j < matCam2.cols; ++j) {
                matCam2.at<float>(i, j) = matrixCam2[i][j];
            }
        }
        cv::Mat cam2simMatrix = cosineMatrix(matCam2, matObject);

        linear_assignment(cam2simMatrix, 0.7f, &matchResult2, &mismatchCurrent,
                          &mismatchPrevious);
    }
    SPDLOG_INFO("hungarianMatching :");

    matchResults.push_back(matchResult1);
    matchResults.push_back(matchResult2);
    for (const auto& item : matchResult1) {
        SPDLOG_INFO("matchResult1: {} {}", item.first, item.second);
    }
}

void MtmctTracker::mtmct(CameraResult& resultCurrent,
                         CameraResult& resultsPrevious) {
    // std::vector<std::vector<float>> matrixCurrent;
    // std::vector<std::vector<float>> matrixPrevious;
    // std::vector<int> mismatchCurrent;
    // std::vector<int> mismatchPrevious;
    // for (const auto& itemCurrent : resultCurrent) {
    //     matrixCurrent.push_back(itemCurrent.feat);
    // }
    // for (const auto& itemPrevious : resultsPrevious) {
    //     matrixPrevious.push_back(itemPrevious.feat);
    // }
    // cv::Mat matCurrent(matrixCurrent.size(), matrixCurrent[0].size(),
    // CV_32F); for (int i = 0; i < matCurrent.rows; ++i) {
    //     for (int j = 0; j < matCurrent.cols; ++j) {
    //         matCurrent.at<float>(i, j) = matrixCurrent[i][j];
    //     }
    // }
    // cv::Mat matPrevious(matrixPrevious.size(), matrixPrevious[0].size(),
    //                     CV_32F);
    // for (int i = 0; i < matPrevious.rows; ++i) {
    //     for (int j = 0; j < matPrevious.cols; ++j) {
    //         matPrevious.at<float>(i, j) = matrixPrevious[i][j];
    //     }
    // }
    // cv ::Mat similarityMatrix = cosineMatrix(matCurrent, matPrevious);
    // // SPDLOG_INFO("hungarianMatching Matrix:");

    // linear_assignment(similarityMatrix, 0.7f, &matchResult, &mismatchCurrent,
    //                   &mismatchPrevious);
    // for (const auto& item : matchResult) {
    //     if (resultsPrevious[item.second].trackState == false)
    //         resultsPrevious[item.second].trackState = true;
    //     resultsPrevious[item.second].feat = resultCurrent[item.first].feat;
    //     resultsPrevious[item.second].rect = resultCurrent[item.first].rect;
    // }
}
}  // namespace PaddleDetection