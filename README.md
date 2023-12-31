# Linux平台编译指南

## 说明
基于飞桨的C++预测库进行的跨镜头跟踪项目,以预先存入待检测目标照片的方式进行跨镜头跟踪,目前支持两路rtsp流。
## 一些思考
目前这种方法实际上是一种偷懒的跨镜头跟踪方法，无法对镜头中的每一个人物进行跟踪。但如果要达到实时跨镜头的效果，目前尝试过的方法是将每个镜头所有出现过的人物特征存储起来，然后进行对比。但效果不是很好

### 前置条件
* G++ 8.2(实测G++版本11.4也可)
* CUDA 9.0 / CUDA 10.1, cudnn 7+ （实测环境cuda11.7）
* CMake 3.0+

请确保系统已经安装好上述基本软件，**下面所有示例以工作目录为 `/root/projects/`演示**。
### 所需库与环境
* SPDLOG (主要习惯于用这个输出信息)
* opencv (注意编译版本的匹配)

### 主要目录和文件

```bash
mtmct/
|
├── src
│   ├── main.cc # 集成代码示例, 程序入口
│   ├── object_detector.cc # 模型加载和预测主要逻辑封装类实现
│   └── preprocess_op.cc # 预处理相关主要逻辑封装实现
|
├── include
│   ├── config_parser.h # 导出模型配置yaml文件解析
│   ├── object_detector.h # 模型加载和预测主要逻辑封装类
│   └── preprocess_op.h # 预处理相关主要逻辑类封装
|
│
├── build.sh # 编译命令脚本
│
├── CMakeList.txt # cmake编译入口文件
|
├── CMakeSettings.json # Visual Studio 2019 CMake项目编译设置
│
└── cmake # 依赖的外部项目cmake（目前仅有yaml-cpp）

```


### Step1: 下载PaddlePaddle C++ 预测库 paddle_inference

PaddlePaddle C++ 预测库针对不同的`CPU`和`CUDA`版本提供了不同的预编译版本，请根据实际情况下载:  [C++预测库下载列表](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html)


下载并解压后`/root/projects/paddle_inference`目录包含内容为：
```
paddle_inference
├── paddle # paddle核心库和头文件
|
├── third_party # 第三方依赖库和头文件
|
└── version.txt # 版本和编译信息
```

**注意:** 预编译版本除`nv-jetson-cuda10-cudnn7.5-trt5` 以外其它包都是基于`GCC 4.8.5`编译，使用高版本`GCC`可能存在 `ABI`兼容性问题，建议降级或[自行编译预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html)。


### Step2: 编译

编译`cmake`的命令在`build.sh`中，请根据实际情况修改主要参数，其主要内容说明如下(因为主要实现跨镜头算,其余功能未开启)：

```
# 是否使用GPU(即是否使用 CUDA)
WITH_GPU=OFF

# 使用MKL or openblas
WITH_MKL=ON

# 是否集成 TensorRT(仅WITH_GPU=ON 有效)
WITH_TENSORRT=OFF

# TensorRT 的include路径
TENSORRT_LIB_DIR=/path/to/TensorRT/include

# TensorRT 的lib路径
TENSORRT_LIB_DIR=/path/to/TensorRT/lib

# Paddle 预测库路径
PADDLE_DIR=/path/to/paddle_inference

# Paddle 预测库名称
PADDLE_LIB_NAME=paddle_inference

# CUDA 的 lib 路径
CUDA_LIB=/path/to/cuda/lib

# CUDNN 的 lib 路径
CUDNN_LIB=/path/to/cudnn/lib

# 是否开启关键点模型预测功能
WITH_KEYPOINT=ON

# 请检查以上各个路径是否正确

# 以下无需改动
cmake .. \
    -DWITH_GPU=${WITH_GPU} \
    -DWITH_MKL=${WITH_MKL} \
    -DWITH_TENSORRT=${WITH_TENSORRT} \
    -DTENSORRT_LIB_DIR=${TENSORRT_LIB_DIR} \
    -DTENSORRT_INC_DIR=${TENSORRT_INC_DIR} \
    -DPADDLE_DIR=${PADDLE_DIR} \
    -DCUDA_LIB=${CUDA_LIB} \
    -DCUDNN_LIB=${CUDNN_LIB} \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DPADDLE_LIB_NAME=${PADDLE_LIB_NAME} \
    -DWITH_KEYPOINT=${WITH_KEYPOINT}
make

```

修改脚本设置好主要参数后，执行`build`脚本：
 ```shell
 sh ./scripts/build.sh
 ```

**注意**: OPENCV依赖OPENBLAS，Ubuntu用户需确认系统是否已存在`libopenblas.so`。如未安装，可执行apt-get install libopenblas-dev进行安装。可能有些库需要复制至运行目录

### Step4: 预测及可视化
编译成功后，预测入口程序为`build/main`其主要命令参数说明如下：
| 参数                  | 说明                                                               |
| --------------------- | ------------------------------------------------------------------ |
| --model_dir           | 导出的检测预测模型所在路径                                         |
| --model_dir_keypoint  | Option                                                             | 导出的关键点预测模型所在路径                         |
| --image_file          | 要预测的图片文件路径                                               |
| --image_dir           | 要预测的图片文件夹路径                                             |
| --video_file          | 要预测的视频文件路径                                               |
| --camera_id           | Option                                                             | 用来预测的摄像头ID，默认为-1（表示不使用摄像头预测） |
| --device              | 运行时的设备，可选择`CPU/GPU/XPU`，默认为`CPU`                     |
| --gpu_id              | 指定进行推理的GPU device id(默认值为0)                             |
| --run_mode            | 使用GPU时，默认为paddle, 可选（paddle/trt_fp32/trt_fp16/trt_int8） |
| --batch_size          | 检测模型预测时的batch size，在指定`image_dir`时有效                |
| --batch_size_keypoint | 关键点模型预测时的batch size，默认为8                              |
| --run_benchmark       | 是否重复预测来进行benchmark测速 ｜                                 |
| --output_dir          | 输出图片所在的文件夹, 默认为output ｜                              |
| --use_mkldnn          | CPU预测中是否开启MKLDNN加速                                        |
| --cpu_threads         | 设置cpu线程数，默认为1                                             |
| --use_dark            | 关键点模型输出预测是否使用DarkPose后处理，默认为true               |

**注意**:
- 优先级顺序：`camera_id` > `video_file` > `image_dir` > `image_file`。
- --run_benchmark如果设置为True，则需要安装依赖`pip install pynvml psutil GPUtil`。

`样例一`：

```shell
#使用 `GPU`预测
./build/main --model_dir=./mot_ppyoloe_s_36e_pipeline --model_dir_reid=./reid_model --video_file='rtsp://admin:abc123456@192.168.0.82:554/cam/realmonitor?channel=1&subtype=0' --device=GPU

```

## api以及使用说明可以参见[飞桨文档](https://www.paddlepaddle.org.cn/inference/v2.5/guides/introduction/index_intro.html)
