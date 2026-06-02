#pragma once
#include <opencv2/core.hpp>

namespace improc::core {

/**
 * @brief Result of CalcHist: a 1-D histogram with its configuration metadata.
 *
 * For Gray images: data is (bins × 1) CV_32F, channels = 1.
 * For BGR images: data is (bins × 3) CV_32F (columns = B, G, R), channels = 3.
 */
struct HistogramData {
    cv::Mat data;
    int     bins      = 256;
    float   range_min = 0.0f;
    float   range_max = 256.0f;
    int     channels  = 1;

    bool empty() const { return data.empty(); }
};

} // namespace improc::core
