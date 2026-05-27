// src/core/ops/photo.cpp
#include "improc/core/ops/photo.hpp"

namespace improc::core {

// ── EdgePreservingFilter ──────────────────────────────────────────────────────

Image<BGR> EdgePreservingFilter::operator()(const Image<BGR>& img) const {
    int flags = (filter_ == Filter::Recursive)
        ? cv::RECURS_FILTER
        : cv::NORMCONV_FILTER;
    cv::Mat result;
    cv::edgePreservingFilter(img.mat(), result, flags, sigma_s_, sigma_r_);
    return Image<BGR>(std::move(result));
}

// ── DetailEnhance ─────────────────────────────────────────────────────────────

Image<BGR> DetailEnhance::operator()(const Image<BGR>& img) const {
    cv::Mat result;
    cv::detailEnhance(img.mat(), result, sigma_s_, sigma_r_);
    return Image<BGR>(std::move(result));
}

// ── Stylize ───────────────────────────────────────────────────────────────────

Image<BGR> Stylize::operator()(const Image<BGR>& img) const {
    cv::Mat result;
    cv::stylization(img.mat(), result, sigma_s_, sigma_r_);
    return Image<BGR>(std::move(result));
}

// ── PencilSketch ──────────────────────────────────────────────────────────────

PencilSketchResult PencilSketch::operator()(const Image<BGR>& img) const {
    cv::Mat gray_out, color_out;
    cv::pencilSketch(img.mat(), gray_out, color_out, sigma_s_, sigma_r_, shade_factor_);
    return {Image<Gray>(std::move(gray_out)), Image<BGR>(std::move(color_out))};
}

// ── SeamlessClone ─────────────────────────────────────────────────────────────

Image<BGR> SeamlessClone::operator()(const Image<BGR>& src,
                                     const Image<BGR>& dst,
                                     const Image<Gray>& mask,
                                     cv::Point center) const {
    if (center.x < 0 || center.y < 0 ||
        center.x >= dst.cols() || center.y >= dst.rows())
        throw improc::ParameterError{"center",
            "must be within dst bounds", "SeamlessClone"};

    int flags = (mode_ == Mode::Normal)   ? cv::NORMAL_CLONE :
                (mode_ == Mode::Mixed)    ? cv::MIXED_CLONE  :
                                            cv::MONOCHROME_TRANSFER;
    cv::Mat result;
    cv::seamlessClone(src.mat(), dst.mat(), mask.mat(), center, result, flags);
    return Image<BGR>(std::move(result));
}

// ── NLMeansDenoisingMulti ─────────────────────────────────────────────────────

Image<BGR> NLMeansDenoisingMulti::operator()(const std::vector<Image<BGR>>& frames) const {
    if (frames.size() < 3)
        throw improc::ParameterError{"frames",
            "at least 3 frames required", "NLMeansDenoisingMulti"};
    if (tws_ < 3 || tws_ % 2 == 0)
        throw improc::ParameterError{"temporal_window_size",
            "must be odd and >= 3", "NLMeansDenoisingMulti"};

    std::vector<cv::Mat> mats;
    mats.reserve(frames.size());
    for (const auto& f : frames) mats.push_back(f.mat());

    int imgToDenoiseIndex = static_cast<int>(mats.size() / 2);
    cv::Mat result;
    cv::fastNlMeansDenoisingColoredMulti(mats, result, imgToDenoiseIndex, tws_, h_, h_);
    return Image<BGR>(std::move(result));
}

// ── MergeHDR ─────────────────────────────────────────────────────────────────

Image<Float32C3> MergeHDR::operator()(const std::vector<Image<BGR>>& images,
                                      const std::vector<float>& times) const {
    if (images.empty())
        throw improc::ParameterError{"images", "must not be empty", "MergeHDR"};
    if (method_ == Method::Debevec && times.size() != images.size())
        throw improc::ParameterError{"times",
            "must match images.size() for Debevec", "MergeHDR"};

    std::vector<cv::Mat> mats;
    mats.reserve(images.size());
    for (const auto& img : images) mats.push_back(img.mat());

    cv::Mat hdr;
    if (method_ == Method::Mertens) {
        cv::createMergeMertens()->process(mats, hdr);
    } else {
        cv::Mat response;
        cv::createCalibrateDebevec()->process(mats, response, times);
        cv::createMergeDebevec()->process(mats, hdr, times, response);
    }
    return Image<Float32C3>(std::move(hdr));
}

// ── ToneMap ───────────────────────────────────────────────────────────────────

Image<BGR> ToneMap::operator()(const Image<Float32C3>& hdr) const {
    cv::Ptr<cv::Tonemap> tonemap;
    switch (algo_) {
        case Algorithm::Linear:   tonemap = cv::createTonemap(gamma_);         break;
        case Algorithm::Drago:    tonemap = cv::createTonemapDrago(gamma_);    break;
        case Algorithm::Reinhard: tonemap = cv::createTonemapReinhard(gamma_); break;
        case Algorithm::Mantiuk:  tonemap = cv::createTonemapMantiuk(gamma_);  break;
    }
    cv::Mat ldr_float, ldr_8u;
    tonemap->process(hdr.mat(), ldr_float);
    ldr_float.convertTo(ldr_8u, CV_8UC3, 255.0);
    return Image<BGR>(std::move(ldr_8u));
}

} // namespace improc::core
