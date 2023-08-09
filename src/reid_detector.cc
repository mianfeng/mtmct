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
#include "include/reid_detector.h"
#include "spdlog/cfg/env.h"
#include "spdlog/spdlog.h"

using namespace paddle_infer;

namespace PaddleDetection {

// Load Model and create model predictor
void ReidDetector::LoadModel(const std::string& model_dir, const int batch_size,
                             const std::string& run_mode) {
    paddle_infer::Config config;
    std::string prog_file = model_dir + OS_PATH_SEP + "model.pdmodel";
    std::string params_file = model_dir + OS_PATH_SEP + "model.pdiparams";

    config.SetModel(prog_file, params_file);
    if (this->device_ == "GPU") {
        config.EnableUseGpu(200, this->gpu_id_);
        config.SwitchIrOptim(true);
        // use tensorrt
        if (run_mode != "paddle") {
            auto precision = paddle_infer::Config::Precision::kFloat32;
            if (run_mode == "trt_fp32") {
                precision = paddle_infer::Config::Precision::kFloat32;
            } else if (run_mode == "trt_fp16") {
                precision = paddle_infer::Config::Precision::kHalf;
            } else if (run_mode == "trt_int8") {
                precision = paddle_infer::Config::Precision::kInt8;
            } else {
                printf(
                    "run_mode should be 'paddle', 'trt_fp32', 'trt_fp16' or "
                    "'trt_int8'");
            }
            // set tensorrt
            config.EnableTensorRtEngine(1 << 30, batch_size,
                                        this->min_subgraph_size_, precision,
                                        false, this->trt_calib_mode_);

            // set use dynamic shape
            if (this->use_dynamic_shape_) {
                // set DynamicShsape for image tensor
                const std::vector<int> min_input_shape = {
                    1, 3, this->trt_min_shape_, this->trt_min_shape_};
                const std::vector<int> max_input_shape = {
                    1, 3, this->trt_max_shape_, this->trt_max_shape_};
                const std::vector<int> opt_input_shape = {
                    1, 3, this->trt_opt_shape_, this->trt_opt_shape_};
                const std::map<std::string, std::vector<int>>
                    map_min_input_shape = {{"image", min_input_shape}};
                const std::map<std::string, std::vector<int>>
                    map_max_input_shape = {{"image", max_input_shape}};
                const std::map<std::string, std::vector<int>>
                    map_opt_input_shape = {{"image", opt_input_shape}};

                config.SetTRTDynamicShapeInfo(map_min_input_shape,
                                              map_max_input_shape,
                                              map_opt_input_shape);
                std::cout << "TensorRT dynamic shape enabled" << std::endl;
            }
        }

    } else if (this->device_ == "XPU") {
        config.EnableXpu(10 * 1024 * 1024);
    } else {
        config.DisableGpu();
        if (this->use_mkldnn_) {
            config.EnableMKLDNN();
            // cache 10 different shapes for mkldnn to avoid memory leak
            config.SetMkldnnCacheCapacity(10);
        }
        config.SetCpuMathLibraryNumThreads(this->cpu_math_library_num_threads_);
    }
    config.SwitchUseFeedFetchOps(false);
    config.SwitchIrOptim(true);
    config.DisableGlogInfo();
    // Memory optimization
    config.EnableMemoryOptim();
    // std::string info = config.Summary();
    // fprintf(stdout, "%s\n", info.c_str());
    predictor_ = std::move(CreatePredictor(config));
}
cv::Scalar GetColor(int idx) {
    idx = idx * 3;
    cv::Scalar color =
        cv::Scalar((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255);
    return color;
}

void VisualizeReidResult(const std::vector<Match>& matchs,
                         const CameraResult& results1,
                         const CameraResult& results2) {
    std::vector<cv::Mat> vis_img;
    // SPDLOG_INFO("VisualizeReidResult");
    vis_img.push_back(results1.imgs.clone());
    vis_img.push_back(results2.imgs.clone());

    cv::Point origin;
    for (int i = 0; i < matchs.size(); ++i) {
        // SPDLOG_INFO("VisualizeReidResult {}", matchs.size());
        ReidResult result;
        cv::Mat img = vis_img[i];
        if (matchs[i].size()) {
            if (i == 0)
                result = results1.results[matchs[i].begin()->first];
            else
                result = results2.results[matchs[i].begin()->first];
            // for (int i = 0; i < results.size(); ++i) {
            const int obj_id = 888;
            // const float score = results[i].score;

            cv::Scalar color = GetColor(obj_id);

            // Draw object, text, and background
            cv::Point pt1 = cv::Point(result.rect[0], result.rect[1]);
            cv::Point pt2 = cv::Point(result.rect[2], result.rect[3]);
            cv::rectangle(img, pt1, pt2, color, 2.5);
            std::ostringstream idoss;
            idoss << std::setiosflags(std::ios::fixed) << std::setprecision(4);
            idoss << obj_id;
            std::string id_text = idoss.str();
            origin.x = result.rect[0];
            origin.y = result.rect[1];
            cv::putText(img, id_text, origin, cv::FONT_HERSHEY_PLAIN, 2,
                        cv::Scalar(0, 255, 255), 2);
        }
        std::string windowName = "result" + std::to_string(i);
        cv::imshow(windowName, img);
        cv::waitKey(1);
    }
}
// Visualization MaskDetector results
// cv::Mat VisualizeReidResult(const cv::Mat& img, const CameraResult& results)
// {
//     cv::Mat vis_img = img.clone();
//     cv::Point origin;

//     for (int i = 0; i < results.size(); ++i) {
//         const int obj_id = results[i].id;
//         // const float score = results[i].score;

//         cv::Scalar color = GetColor(obj_id);

//         // Draw object, text, and background
//         cv::Point pt1 = cv::Point(results[i].rect[0], results[i].rect[1]);
//         cv::Point pt2 = cv::Point(results[i].rect[2], results[i].rect[3]);
//         cv::rectangle(vis_img, pt1, pt2, color, 2.5);
//         std::ostringstream idoss;
//         idoss << std::setiosflags(std::ios::fixed) << std::setprecision(4);
//         idoss << obj_id;
//         std::string id_text = idoss.str();
//         origin.x = results[i].rect[0];
//         origin.y = results[i].rect[1];
//         cv::putText(vis_img, id_text, origin, cv::FONT_HERSHEY_PLAIN, 2,
//                     cv::Scalar(0, 255, 255), 2);
//     }

//     return vis_img;
// }

void ReidDetector::Preprocess(const cv::Mat& ori_im) {
    // Clone the image : keep the original mat for postprocess
    cv::Mat im = ori_im.clone();
    cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
    preprocessor_.Run(&im, &inputs_);
}
void ReidDetector::Preprocess(const cv::Mat& oriImg, ImageBlob* data,
                              const std::vector<double>& mean,
                              const std::vector<double>& std) {
    cv::Size input_wh(128, 256);
    cv::Mat normalized_img;
    // Clone the image : keep the original mat for postprocess
    // 进行缩放
    int origin_w = oriImg.cols;
    int origin_h = oriImg.rows;

    int target_h = input_wh.height;
    int target_w = input_wh.width;

    float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h);
    float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w);
    float resize_scale = std::min(ratio_h, ratio_w);

    // input_wh.height = std::round(oriImg.cols * resize_scale);
    // input_wh.width = std::round(oriImg.rows * resize_scale);
    cv::Mat resized_img;
    cv::resize(oriImg, resized_img, input_wh, cv::INTER_AREA);
    cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);
    double e = 1.0 / 255.0;
    resized_img.convertTo(normalized_img, CV_32FC3, e);

    // 一次完成归一化的操作
    cv::subtract(normalized_img, mean,
                 normalized_img);  // 归一化图像 = 原始图像 - 均值
    cv::divide(normalized_img, std,
               normalized_img);  // 归一化图像 = 归一化图像 / 标准差

    (data->im_data_)
        .resize(normalized_img.channels() * normalized_img.rows *
                normalized_img.cols);
    float* base = (data->im_data_).data();
    for (int i = 0; i < normalized_img.channels(); ++i) {
        cv::extractChannel(
            normalized_img,
            cv::Mat(normalized_img.rows, normalized_img.cols, CV_32FC1,
                    base + i * normalized_img.rows * normalized_img.cols),
            i);
    }
}

void ReidDetector::Predict(const cv::Mat& img, const double threshold,
                           PaddleDetection::CameraResult* results) {
    // in_data_batch
    std::vector<float> in_data_all;
    std::vector<float> im_shape_all(1 * 2);
    std::vector<float> scale_factor_all(1 * 2);

    // Preprocess image
    for (int bs_idx = 0; bs_idx < 1; bs_idx++) {
        cv::Mat im = img.clone();
        Preprocess(im, &inputs_);
        in_data_all.insert(in_data_all.end(), inputs_.im_data_.begin(),
                           inputs_.im_data_.end());
    }

    // Prepare input tensor

    auto input_names = predictor_->GetInputNames();
    for (const auto& tensor_name : input_names) {
        // fprintf(stdout, "%s\n", tensor_name.c_str());
        auto in_tensor = predictor_->GetInputHandle(tensor_name);
        if (tensor_name == "x") {
            in_tensor->Reshape({1, 3, 256, 128});
            in_tensor->CopyFromCpu(in_data_all.data());
        }
    }
    std::vector<int> output_shape, idx_shape;
    // Run predictor

    predictor_->Run();
    // Get output tensor
    auto output_names = predictor_->GetOutputNames();
    auto out_tensor = predictor_->GetOutputHandle(output_names[0]);
    output_shape = out_tensor->shape();

    // Calculate output length
    int output_size = 1;
    for (int j = 0; j < output_shape.size(); ++j) {
        output_size *= output_shape[j];
    }
    if (output_size < 6) {
        std::cerr << "[WARNING] No object detected." << std::endl;
    }
    ReidResult result;
    result.feat.resize(output_size);
    output_data_.resize(output_size);
    out_tensor->CopyToCpu(result.feat.data());

    result.rect = {-1, -1, -1, -1};
    result.id = -1;
    results->results.emplace_back(result);
}
void ReidDetector::Predict(const std::vector<cv::Mat> imgs,
                           std::vector<int> rect, const double threshold,
                           const int warmup, const int repeats,
                           PaddleDetection::CameraResult* results, int num,
                           std::vector<double>* times) {
    auto preprocess_start = std::chrono::steady_clock::now();
    int batch_size = imgs.size();
    // in_data_batch
    std::vector<float> in_data_all;
    std::vector<float> im_shape_all(batch_size * 2);
    std::vector<float> scale_factor_all(batch_size * 2);

    // Preprocess image
    for (int bs_idx = 0; bs_idx < batch_size; bs_idx++) {
        cv::Mat im = imgs.at(bs_idx);
        Preprocess(im, &inputs_);
        in_data_all.insert(in_data_all.end(), inputs_.im_data_.begin(),
                           inputs_.im_data_.end());
    }

    // Prepare input tensor

    auto input_names = predictor_->GetInputNames();
    for (const auto& tensor_name : input_names) {
        // fprintf(stdout, "%s\n", tensor_name.c_str());
        auto in_tensor = predictor_->GetInputHandle(tensor_name);
        if (tensor_name == "x") {
            // int rh = inputs_.in_net_shape_[0];
            // int rw = inputs_.in_net_shape_[1];
            in_tensor->Reshape({batch_size, 3, 256, 128});
            in_tensor->CopyFromCpu(in_data_all.data());
        } else if (tensor_name == "im_shape") {
            in_tensor->Reshape({batch_size, 2});
            in_tensor->CopyFromCpu(im_shape_all.data());
        } else if (tensor_name == "scale_factor") {
            in_tensor->Reshape({batch_size, 2});
            in_tensor->CopyFromCpu(scale_factor_all.data());
        }
    }
    // fprintf(stdout, "LINE:%d, %s\n", __LINE__, __func__);
    auto preprocess_end = std::chrono::steady_clock::now();
    std::vector<int> output_shape, idx_shape;
    // Run predictor
    // warmup
    for (int i = 0; i < warmup; i++) {
        predictor_->Run();
        // Get output tensor
        auto output_names = predictor_->GetOutputNames();
        // fprintf(stderr, "output_names: %s\n", output_names[0].c_str());
        auto out_tensor = predictor_->GetOutputHandle(output_names[0]);
        output_shape = out_tensor->shape();
        // SPDLOG_INFO("output name {} shape {}, {}", output_names.size(),
        //             output_shape[0], output_shape[1]);

        // Calculate output length
        int output_size = 1;
        for (int j = 0; j < output_shape.size(); ++j) {
            output_size *= output_shape[j];
        }
        if (output_size < 6) {
            std::cerr << "[WARNING] No object detected." << std::endl;
        }
        output_data_.resize(output_size);
        out_tensor->CopyToCpu(output_data_.data());
    }

    auto inference_start = std::chrono::steady_clock::now();
    // for (int i = 0; i < repeats; i++) {
    predictor_->Run();
    // Get output tensor
    auto output_names = predictor_->GetOutputNames();
    // fprintf(stderr, "output_names: %s\n", output_names[0].c_str());
    auto out_tensor = predictor_->GetOutputHandle(output_names[0]);
    output_shape = out_tensor->shape();
    // SPDLOG_INFO("output name {} shape {}, {}", output_names.size(),
    //             output_shape[0], output_shape[1]);

    // Calculate output length
    int output_size = 1;
    for (int j = 0; j < output_shape.size(); ++j) {
        output_size *= output_shape[j];
    }
    if (output_size < 6) {
        std::cerr << "[WARNING] No object detected." << std::endl;
    }
    ReidResult result;
    auto inference_end = std::chrono::steady_clock::now();

    result.feat.resize(output_size);
    // SPDLOG_INFO("output size:{}", output_size);
    output_data_.resize(output_size);
    out_tensor->CopyToCpu(result.feat.data());
    // auto inference_end = std::chrono::steady_clock::now();

    result.rect = {rect[0], rect[1], rect[2], rect[3]};
    result.id = num;
    results->results.emplace_back(result);
    // }
    std::chrono::duration<float> preprocess_diff =
        preprocess_end - preprocess_start;
    std::chrono::duration<float, std::milli> preprocess_diff_ms =
        std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(
            preprocess_diff);
    double premilliseconds = preprocess_diff_ms.count();

    std::chrono::duration<float> inference_diff =
        inference_end - inference_start;
    std::chrono::duration<float, std::milli> inference_diff_ms =
        std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(
            inference_diff);
    double milliseconds = inference_diff_ms.count();

    // SPDLOG_INFO("preprocess time:{}, inference time:{}", premilliseconds,
    //             milliseconds);
}

}  // namespace PaddleDetection
