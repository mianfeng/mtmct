#include "include/reid_detector.h"
namespace PaddleDetection {
struct CameraresultPools {
    std::vector<ReidResult> cam1result;
    std::vector<ReidResult> cam2result;
};

class MtmctTracker {
   public:
    MtmctTracker(void);
    void mtmct(CameraResult& resultCurrent, CameraResult& resultsPrevious);
    void mtmct(CameraResult& cameraResult1, CameraResult& cameraResult2,
               std::vector<float>& initObjects,
               std::vector<Match>& matchResult);

   private:
    int maxLostimes;
    CameraresultPools activeCameraresultPools;
    CameraresultPools lostCameraresultPools;
    void linear_assignment(const cv::Mat& cost, float cost_limit,
                           Match* matches, std::vector<int>* mismatch_row,
                           std::vector<int>* mismatch_col);
    cv::Mat cosineMatrix(const cv::Mat& matrixCurrent,
                         const cv::Mat& matrixPrevious);
    cv::Mat computeNorm(const cv::Mat& matrix);
};
}  // namespace PaddleDetection