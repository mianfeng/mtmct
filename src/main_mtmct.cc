#include <gflags/gflags.h>
#include <glog/logging.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <queue>
#include <string>

#include "include/common.h"
#include "include/mtmct_tracker.h"
// #include "include/ff_get_frame.h"
#include "include/jde_detector.h"
#include "include/myqueue.h"
#include "include/object_detector.h"
#include "include/reid_detector.h"
#include "paddle_inference_api.h"
#include "spdlog/cfg/env.h"
#include "spdlog/spdlog.h"

DEFINE_string(model_dir, "", "Path of inference model");
DEFINE_string(model_dir_reid, "", "Path of reid inference model");
DEFINE_string(image_file, "", "Path of input image");
DEFINE_string(image_dir, "",
              "Dir of input image, `image_file` has a higher priority.");
DEFINE_int32(batch_size, 1, "batch_size");
DEFINE_int32(batch_size_reid, 1, "batch_size of ried detector");
DEFINE_string(
    video_file, "",
    "Path of input video, `video_file` or `camera_id` has a highest priority.");
DEFINE_int32(camera_id, -1, "Device id of camera to predict");
DEFINE_bool(
    use_gpu, false,
    "Deprecated, please use `--device` to set the device you want to run.");
DEFINE_string(device, "GPU",
              "Choose the device you want to run, it can be: CPU/GPU/XPU, "
              "default is CPU.");
DEFINE_double(threshold, 0.5, "Threshold of score.");
DEFINE_string(output_dir, "output", "Directory of output visualization files.");
DEFINE_string(run_mode, "paddle",
              "Mode of running(paddle/trt_fp32/trt_fp16/trt_int8)");
DEFINE_int32(gpu_id, 0, "Device id of GPU to execute");
DEFINE_bool(run_benchmark, false,
            "Whether to predict a image_file repeatedly for benchmark");
DEFINE_bool(use_mkldnn, false, "Whether use mkldnn with CPU");
DEFINE_int32(cpu_threads, 1, "Num of threads with CPU");
DEFINE_int32(trt_min_shape, 1, "Min shape of TRT DynamicShapeI");
DEFINE_int32(trt_max_shape, 1280, "Max shape of TRT DynamicShapeI");
DEFINE_int32(trt_opt_shape, 640, "Opt shape of TRT DynamicShapeI");
DEFINE_bool(trt_calib_mode, false,
            "If the model is produced by TRT offline quantitative calibration, "
            "trt_calib_mode need to set True");
DEFINE_bool(use_dark, true, "Whether use dark decode in keypoint postprocess");

static void initLogger() noexcept {
    spdlog::cfg::load_env_levels();
#ifdef NDEBUG
    spdlog::set_pattern("%^[%L] %v [%Y-%m-%d %H:%M:%S.%e]%$");
#else
    spdlog::set_pattern("%^[%L] %v [%Y-%m-%d %H:%M:%S.%e] [%@]%$");
#endif
}
void captureVideo(const std::string& video_path,
                  std::shared_ptr<AysncQueue<cv::Mat>> queue, int threadID) {
    cv::Mat frame;
    cv::VideoCapture capture;
    SPDLOG_INFO("  thread {} captureVideo thread start", threadID);
    capture.open(video_path.c_str());
    while (1) {
        capture.read(frame);
        if (!frame.empty()) {
            queue->enqueue(frame);
            // SPDLOG_INFO("captureVideo thread start");
        }
    }
}
void PredictVideo(
    const std::string& video_path, PaddleDetection::ObjectDetector* det,
    PaddleDetection::ReidDetector* reidDet,
    std::shared_ptr<AysncQueue<PaddleDetection::CameraResult>> cameraResult,
    int threadID = 0) {
    SPDLOG_INFO("PredictVideo thread {} start", threadID);

    std::vector<PaddleDetection::ObjectResult> result;
    PaddleDetection::MOT_Result motResult;

    std::vector<int> bbox_num;
    std::vector<double> det_times;

    // Store reid results
    std::vector<cv::Mat> imgs_kpts;

    // Capture all frames and do inference
    cv::Mat frame;
    int frame_id = 1;
    bool is_rbox = false;
    std::shared_ptr<AysncQueue<cv::Mat>> qu(new AysncQueue<cv::Mat>);
    std::thread capImg(captureVideo, video_path, qu, threadID);

    while (1) {
        PaddleDetection::CameraResult resultFrame;
        PaddleDetection::Match matchResult;
        std::vector<cv::Mat> imgs;
        frame = qu->dequeue();
        imgs.push_back(frame);
        // printf("thread %d detect frame: %d\n", threadID, frame_id);
        det->Predict(imgs, FLAGS_threshold, 0, 1, &result, &bbox_num,
                     &det_times);
        std::vector<PaddleDetection::ObjectResult> out_result;
        for (const auto& item : result) {
            if (item.confidence < FLAGS_threshold || item.class_id == -1) {
                continue;
            }
            out_result.push_back(item);
            if (item.rect.size() > 6) {
                is_rbox = true;
                printf(
                    "class=%d confidence=%.4f rect=[%d %d %d %d %d %d %d %d]\n",
                    item.class_id, item.confidence, item.rect[0], item.rect[1],
                    item.rect[2], item.rect[3], item.rect[4], item.rect[5],
                    item.rect[6], item.rect[7]);
            }
        }

        if (reidDet) {
            // resultFrame.clear();
            int imsize = out_result.size();

            for (int i = 0; i < imsize; i++) {
                auto item = out_result[i];
                cv::Mat crop_img;
                std::vector<double> reid_times(2);
                std::vector<int> rect = {item.rect[0], item.rect[1],
                                         item.rect[2], item.rect[3]};
                std::vector<float> center;
                std::vector<float> scale;
                if (item.class_id == 0) {
                    // 切割图像，还没处理过
                    PaddleDetection::CropImg(frame, crop_img, rect, center,
                                             scale);
                    // emplace_back允许我们通过传递构造函数的参数来构造新元素，
                    // 而不是先创建一个临时对象再将其复制或移动到容器中。

                    imgs_kpts.emplace_back(crop_img);
                }

                reidDet->Predict(imgs_kpts, rect, FLAGS_threshold, 0, 1,
                                 &resultFrame, i, &reid_times);
                imgs_kpts.clear();
            }
            SPDLOG_INFO("thread {} frame: {} reid result size: {}", threadID,
                        frame_id, resultFrame.results.size());
        }
        frame_id += 1;
        resultFrame.cameraId = threadID;
        resultFrame.timestamp = frame_id;
        resultFrame.imgs = frame.clone();
        if (cameraResult->size() > 1) {
            cameraResult->dequeue();
        }
        cameraResult->enqueue(resultFrame);
    }
    capImg.join();
}
void InitObject(PaddleDetection::ReidDetector* reidDet,
                PaddleDetection::CameraResult& initObjects,
                std::vector<float>& matrixObject) {
    initObjects.cameraId = -1;
    initObjects.timestamp = -1;
    // 图片数量
    int numImages = 4;
    // 图片文件格式
    std::string filePath = "./goal_img/000";
    std::string fileFormat = ".png";
    float avgFeat;

    for (int i = 1; i <= numImages; ++i) {
        // 构建完整的文件名
        std::string filename = filePath + std::to_string(i) + fileFormat;
        // 读取图片并存储到容器中
        cv::Mat image = cv::imread(filename);
        if (!image.empty()) {
            reidDet->Predict(image, 0.6, &initObjects);

        } else {
            SPDLOG_INFO("read image {} failed", filename);
        }
    }
    for (int j = 0; j < initObjects.results[0].feat.size(); ++j) {
        for (int i = 0; i < initObjects.results.size(); ++i)
            avgFeat += initObjects.results[i].feat[j];
        matrixObject.push_back(avgFeat / initObjects.results.size());
        avgFeat = 0;
    }
    SPDLOG_INFO("read image done");
}
int main(int argc, char** argv) {
    initLogger();

    // Parsing command-line
    google::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_model_dir.empty() ||
        (FLAGS_image_file.empty() && FLAGS_image_dir.empty() &&
         FLAGS_video_file.empty())) {
        std::cout << "Usage: ./main --model_dir=/PATH/TO/INFERENCE_MODEL/ "
                  << "--image_file=/PATH/TO/INPUT/IMAGE/" << std::endl;
        return -1;
    }
    if (!(FLAGS_run_mode == "paddle" || FLAGS_run_mode == "trt_fp32" ||
          FLAGS_run_mode == "trt_fp16" || FLAGS_run_mode == "trt_int8")) {
        std::cout << "run_mode should be 'paddle', 'trt_fp32', 'trt_fp16' or "
                     "'trt_int8'.";
        return -1;
    }
    transform(FLAGS_device.begin(), FLAGS_device.end(), FLAGS_device.begin(),
              ::toupper);
    if (!(FLAGS_device == "CPU" || FLAGS_device == "GPU" ||
          FLAGS_device == "XPU")) {
        std::cout << "device should be 'CPU', 'GPU' or 'XPU'.";
        return -1;
    }
    if (FLAGS_use_gpu) {
        std::cout
            << "Deprecated, please use `--device` to set the device you want "
               "to run.";
        return -1;
    }

    // Load model and create a object detector
    PaddleDetection::ObjectDetector detThead1(
        FLAGS_model_dir, FLAGS_device, FLAGS_use_mkldnn, FLAGS_cpu_threads,
        FLAGS_run_mode, FLAGS_batch_size, FLAGS_gpu_id, FLAGS_trt_min_shape,
        FLAGS_trt_max_shape, FLAGS_trt_opt_shape, FLAGS_trt_calib_mode);

    PaddleDetection::ReidDetector* reidDetThead1 = nullptr;
    if (!FLAGS_model_dir_reid.empty()) {
        reidDetThead1 = new PaddleDetection::ReidDetector(
            FLAGS_model_dir_reid, FLAGS_device, FLAGS_use_mkldnn,
            FLAGS_cpu_threads, FLAGS_run_mode, FLAGS_batch_size_reid,
            FLAGS_gpu_id, FLAGS_trt_min_shape, FLAGS_trt_max_shape,
            FLAGS_trt_opt_shape, FLAGS_trt_calib_mode, FLAGS_use_dark);
    }
    PaddleDetection::ObjectDetector detThead2(
        FLAGS_model_dir, FLAGS_device, FLAGS_use_mkldnn, FLAGS_cpu_threads,
        FLAGS_run_mode, FLAGS_batch_size, FLAGS_gpu_id, FLAGS_trt_min_shape,
        FLAGS_trt_max_shape, FLAGS_trt_opt_shape, FLAGS_trt_calib_mode);

    PaddleDetection::ReidDetector* reidDetThead2 = nullptr;
    if (!FLAGS_model_dir_reid.empty()) {
        reidDetThead2 = new PaddleDetection::ReidDetector(
            FLAGS_model_dir_reid, FLAGS_device, FLAGS_use_mkldnn,
            FLAGS_cpu_threads, FLAGS_run_mode, FLAGS_batch_size_reid,
            FLAGS_gpu_id, FLAGS_trt_min_shape, FLAGS_trt_max_shape,
            FLAGS_trt_opt_shape, FLAGS_trt_calib_mode, FLAGS_use_dark);
    }
    std::vector<float> matrixObject;
    PaddleDetection::CameraResult initObjects;
    InitObject(reidDetThead1, initObjects, matrixObject);

    std::thread thread1;
    std::thread thread2;
    std::shared_ptr<AysncQueue<PaddleDetection::CameraResult>> cameraResult1qu(
        new AysncQueue<PaddleDetection::CameraResult>);
    std::shared_ptr<AysncQueue<PaddleDetection::CameraResult>> cameraResult2qu(
        new AysncQueue<PaddleDetection::CameraResult>);

    PaddleDetection::MtmctTracker mtmct;
    std::string videoPath =
        "rtsp://admin:abc123456@192.168.0.81:554/cam/"
        "realmonitor?channel=1&subtype=0";

    thread1 = std::thread(PredictVideo, FLAGS_video_file, &detThead1,
                          reidDetThead1, cameraResult1qu, 1);
    thread2 = std::thread(PredictVideo, videoPath, &detThead2, reidDetThead2,
                          cameraResult2qu, 2);
    PaddleDetection::CameraResult cameraResult1;
    PaddleDetection::CameraResult cameraResult2;
    std::vector<PaddleDetection::Match> matchResults;
    while (1) {
        cameraResult1 = cameraResult1qu->dequeue();
        cameraResult2 = cameraResult2qu->dequeue();
        mtmct.mtmct(cameraResult1, cameraResult2, matrixObject, matchResults);
        VisualizeReidResult(matchResults, cameraResult1, cameraResult2);
        matchResults.clear();
    }
    thread1.join();
    thread2.join();

    return 0;
}
