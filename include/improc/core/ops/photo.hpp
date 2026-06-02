#pragma once
#include <vector>
#include <opencv2/photo.hpp>
#include "improc/core/image.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/**
 * @brief Result of PencilSketch containing both the grayscale and colour sketch outputs.
 */
struct PencilSketchResult {
    Image<Gray> gray;  ///< Grayscale pencil-sketch rendition.
    Image<BGR>  color; ///< Colour pencil-sketch rendition.
};

/**
 * @brief Clones a region from a source image into a destination image with seamless blending.
 */
struct SeamlessClone {
    /// @brief Blending mode for SeamlessClone.
    enum class Mode {
        Normal,     ///< Standard seamless cloning (cv::NORMAL_CLONE).
        Mixed,      ///< Mixed seamless cloning preserving destination gradients (cv::MIXED_CLONE).
        Monochrome, ///< Monochrome transfer (cv::MONOCHROME_TRANSFER).
    };
    /// @brief Sets the blending mode (default: Mode::Normal).
    SeamlessClone& mode(Mode m) { mode_ = m; return *this; }
    /// @brief Clones the masked region of src into dst centred at center.
    [[nodiscard]] Image<BGR> operator()(const Image<BGR>& src,
                          const Image<BGR>& dst,
                          const Image<Gray>& mask,
                          cv::Point center) const;
private:
    Mode mode_ = Mode::Normal;
};

/**
 * @brief Smooths an image while preserving edges using cv::edgePreservingFilter.
 */
struct EdgePreservingFilter {
    /// @brief Filter type for edge-preserving smoothing.
    enum class Filter {
        Recursive, ///< Recursive filter (default, faster).
        NormConv,  ///< Normalised convolution filter (higher quality).
    };
    /// @brief Sets the spatial standard deviation (range: 0–200; default: 60).
    EdgePreservingFilter& sigma_s(float s) { sigma_s_ = s; return *this; }
    /// @brief Sets the colour standard deviation (range: 0–1; default: 0.4).
    EdgePreservingFilter& sigma_r(float r) { sigma_r_ = r; return *this; }
    /// @brief Sets the filter type (default: Filter::Recursive).
    EdgePreservingFilter& filter(Filter f) { filter_  = f; return *this; }
    /// @return Smoothed BGR image with preserved edges.
    [[nodiscard]] Image<BGR> operator()(const Image<BGR>&) const;
private:
    float  sigma_s_ = 60.f;
    float  sigma_r_ = 0.4f;
    Filter filter_  = Filter::Recursive;
};

/**
 * @brief Enhances fine detail in an image using cv::detailEnhance.
 */
struct DetailEnhance {
    /// @brief Sets the spatial standard deviation (default: 10).
    DetailEnhance& sigma_s(float s) { sigma_s_ = s; return *this; }
    /// @brief Sets the colour standard deviation (default: 0.15).
    DetailEnhance& sigma_r(float r) { sigma_r_ = r; return *this; }
    /// @return Detail-enhanced BGR image.
    [[nodiscard]] Image<BGR> operator()(const Image<BGR>&) const;
private:
    float sigma_s_ = 10.f;
    float sigma_r_ = 0.15f;
};

/**
 * @brief Renders a painterly stylisation of a BGR image using cv::stylization.
 */
struct Stylize {
    /// @brief Sets the spatial standard deviation (default: 60).
    Stylize& sigma_s(float s) { sigma_s_ = s; return *this; }
    /// @brief Sets the colour standard deviation (default: 0.45).
    Stylize& sigma_r(float r) { sigma_r_ = r; return *this; }
    /// @return Stylised BGR image.
    [[nodiscard]] Image<BGR> operator()(const Image<BGR>&) const;
private:
    float sigma_s_ = 60.f;
    float sigma_r_ = 0.45f;
};

/**
 * @brief Renders a pencil-sketch stylisation of a BGR image using cv::pencilSketch.
 */
struct PencilSketch {
    /// @brief Sets the spatial standard deviation (default: 60).
    PencilSketch& sigma_s(float s)      { sigma_s_      = s; return *this; }
    /// @brief Sets the colour standard deviation (default: 0.07).
    PencilSketch& sigma_r(float r)      { sigma_r_      = r; return *this; }
    /// @brief Sets the shading intensity for the colour sketch (default: 0.05).
    PencilSketch& shade_factor(float f) { shade_factor_ = f; return *this; }
    /// @return PencilSketchResult containing grayscale and colour sketch images.
    [[nodiscard]] PencilSketchResult operator()(const Image<BGR>&) const;
private:
    float sigma_s_      = 60.f;
    float sigma_r_      = 0.07f;
    float shade_factor_ = 0.05f;
};

/**
 * @brief Denoises a BGR image sequence using Non-Local Means over a temporal window.
 */
struct NLMeansDenoisingMulti {
    /// @brief Sets the filter strength — higher h removes more noise but blurs more (default: 3).
    NLMeansDenoisingMulti& h(float h)                  { h_   = h; return *this; }
    /// @brief Sets the number of surrounding frames in the temporal window (default: 3).
    NLMeansDenoisingMulti& temporal_window_size(int w) { tws_ = w; return *this; }
    /// @return Denoised BGR image of the middle frame.
    [[nodiscard]] Image<BGR> operator()(const std::vector<Image<BGR>>&) const;
private:
    float h_   = 3.f;
    int   tws_ = 3;
};

/**
 * @brief Merges multiple exposures into a single HDR/LDR image.
 *
 * Method::Mertens (default): exposure fusion — no exposure times required.
 * Method::Debevec: camera response recovery — requires exposure times.
 */
struct MergeHDR {
    /// @brief Merge algorithm selection.
    enum class Method {
        Mertens, ///< Exposure fusion (Mertens et al.) — produces a tone-mapped LDR result.
        Debevec, ///< Debevec HDR reconstruction — requires exposure times vector.
    };
    /// @brief Sets the merge algorithm (default: Method::Mertens).
    MergeHDR& method(Method m) { method_ = m; return *this; }
    /// @return Image<Float32C3> HDR image.
    [[nodiscard]] Image<Float32C3> operator()(const std::vector<Image<BGR>>& images,
                                const std::vector<float>& times = {}) const;
private:
    Method method_ = Method::Mertens;
};

/**
 * @brief Maps an HDR float image to displayable 8-bit range using a tone-mapping operator.
 */
struct ToneMap {
    /// @brief Tone-mapping algorithm selection.
    enum class Algorithm {
        Linear,   ///< Simple linear scaling.
        Drago,    ///< Drago logarithmic tone mapping.
        Reinhard, ///< Reinhard global tone mapping (default).
        Mantiuk,  ///< Mantiuk contrast-preserving tone mapping.
    };
    /// @brief Sets the output gamma correction (default: 1.0 = no correction).
    ToneMap& gamma(float g)        { gamma_ = g; return *this; }
    /// @brief Sets the tone-mapping algorithm (default: Algorithm::Reinhard).
    ToneMap& algorithm(Algorithm a) { algo_ = a; return *this; }
    /// @return Tone-mapped BGR image (CV_8UC3).
    [[nodiscard]] Image<BGR> operator()(const Image<Float32C3>&) const;
private:
    float     gamma_ = 1.f;
    Algorithm algo_  = Algorithm::Reinhard;
};

} // namespace improc::core
