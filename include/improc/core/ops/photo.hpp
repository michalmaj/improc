#pragma once
#include <vector>
#include <opencv2/photo.hpp>
#include "improc/core/image.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

struct PencilSketchResult {
    Image<Gray> gray;
    Image<BGR>  color;
};

struct SeamlessClone {
    enum class Mode { Normal, Mixed, Monochrome };
    SeamlessClone& mode(Mode m) { mode_ = m; return *this; }
    Image<BGR> operator()(const Image<BGR>& src,
                          const Image<BGR>& dst,
                          const Image<Gray>& mask,
                          cv::Point center) const;
private:
    Mode mode_ = Mode::Normal;
};

struct EdgePreservingFilter {
    enum class Filter { Recursive, NormConv };
    EdgePreservingFilter& sigma_s(float s) { sigma_s_ = s; return *this; }
    EdgePreservingFilter& sigma_r(float r) { sigma_r_ = r; return *this; }
    EdgePreservingFilter& filter(Filter f) { filter_  = f; return *this; }
    Image<BGR> operator()(const Image<BGR>&) const;
private:
    float  sigma_s_ = 60.f;
    float  sigma_r_ = 0.4f;
    Filter filter_  = Filter::Recursive;
};

struct DetailEnhance {
    DetailEnhance& sigma_s(float s) { sigma_s_ = s; return *this; }
    DetailEnhance& sigma_r(float r) { sigma_r_ = r; return *this; }
    Image<BGR> operator()(const Image<BGR>&) const;
private:
    float sigma_s_ = 10.f;
    float sigma_r_ = 0.15f;
};

struct Stylize {
    Stylize& sigma_s(float s) { sigma_s_ = s; return *this; }
    Stylize& sigma_r(float r) { sigma_r_ = r; return *this; }
    Image<BGR> operator()(const Image<BGR>&) const;
private:
    float sigma_s_ = 60.f;
    float sigma_r_ = 0.45f;
};

struct PencilSketch {
    PencilSketch& sigma_s(float s)      { sigma_s_      = s; return *this; }
    PencilSketch& sigma_r(float r)      { sigma_r_      = r; return *this; }
    PencilSketch& shade_factor(float f) { shade_factor_ = f; return *this; }
    PencilSketchResult operator()(const Image<BGR>&) const;
private:
    float sigma_s_      = 60.f;
    float sigma_r_      = 0.07f;
    float shade_factor_ = 0.05f;
};

struct NLMeansDenoisingMulti {
    NLMeansDenoisingMulti& h(float h)                  { h_   = h; return *this; }
    NLMeansDenoisingMulti& temporal_window_size(int w) { tws_ = w; return *this; }
    Image<BGR> operator()(const std::vector<Image<BGR>>&) const;
private:
    float h_   = 3.f;
    int   tws_ = 3;
};

struct MergeHDR {
    enum class Method { Mertens, Debevec };
    MergeHDR& method(Method m) { method_ = m; return *this; }
    Image<Float32C3> operator()(const std::vector<Image<BGR>>& images,
                                const std::vector<float>& times = {}) const;
private:
    Method method_ = Method::Mertens;
};

struct ToneMap {
    enum class Algorithm { Linear, Drago, Reinhard, Mantiuk };
    ToneMap& gamma(float g)        { gamma_ = g; return *this; }
    ToneMap& algorithm(Algorithm a) { algo_ = a; return *this; }
    Image<BGR> operator()(const Image<Float32C3>&) const;
private:
    float     gamma_ = 1.f;
    Algorithm algo_  = Algorithm::Reinhard;
};

} // namespace improc::core
