#include <opencv2/opencv.hpp>
#include <stdexcept>

#include "common/det.h"
#include "common/visualize.h"

void DrawDetections(cv::Mat& img_bgr, const std::vector<Det>& dets) {
    for (const auto& d : dets) {
        cv::rectangle(img_bgr,
            cv::Point((int)d.x1, (int)d.y1),
            cv::Point((int)d.x2, (int)d.y2),
            cv::Scalar(0, 255, 0), 2);

        char buf[64];
        std::snprintf(buf, sizeof(buf), "id=%d %.2f", d.class_id, d.score);
        int y = std::max(0, (int)d.y1 - 5);
        cv::putText(img_bgr, buf, cv::Point((int)d.x1, y),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }
}

bool DrawAndSaveDetections(
    const cv::Mat& src_bgr,
    const std::vector<Det>& dets,
    const std::string& out_path
) {
    if (src_bgr.empty()) return false;

    cv::Mat vis = src_bgr.clone();
    DrawDetections(vis, dets);
    return cv::imwrite(out_path, vis);
}
