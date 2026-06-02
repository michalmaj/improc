// include/improc/calib/ops/undistort.hpp
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>  // template body calls cv::undistort inline
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/calib/ops/calib_types.hpp"
#include "improc/exceptions.hpp"

namespace improc::calib {

using improc::core::Image;
using improc::core::AnyFormat;

/**
 * @brief Removes lens distortion from an image using a camera matrix and distortion coefficients.
 *
 * Wraps `cv::undistort`. Both K and dist must be set before invoking.
 *
 * @throws improc::ParameterError if K or dist have not been set.
 *
 * @code
 * auto undist = img | Undistort{}.K(camera_matrix).dist(dist_coeffs);
 * @endcode
 */
struct Undistort {
    /// @brief Sets the 3x3 camera intrinsic matrix (CV_64F).
    Undistort& K(cv::Mat k)    { K_ = std::move(k); return *this; }
    /// @brief Sets the distortion coefficient vector.
    Undistort& dist(cv::Mat d) { dist_ = std::move(d); return *this; }

    template<AnyFormat F>
    [[nodiscard]] Image<F> operator()(Image<F> img) const {
        if (K_.empty())
            throw improc::ParameterError{"K", "must be set before calling operator()", "Undistort"};
        if (dist_.empty())
            throw improc::ParameterError{"dist", "must be set before calling operator()", "Undistort"};
        cv::Mat dst;
        cv::undistort(img.mat(), dst, K_, dist_);
        return Image<F>(std::move(dst));
    }

private:
    cv::Mat K_;
    cv::Mat dist_;
};

/**
 * @brief Computes undistortion and rectification maps for use with Remap.
 *
 * Wraps `cv::initUndistortRectifyMap`. Both K and dist must be set before invoking.
 * Optionally accepts a new camera matrix and rotation matrix for stereo rectification.
 *
 * @throws improc::ParameterError if K or dist have not been set.
 *
 * @code
 * auto maps = UndistortMap{}.K(camera_matrix).dist(dist_coeffs)(image_size);
 * auto undist = img | Remap{maps.map1, maps.map2};
 * @endcode
 */
struct UndistortMap {
    /// @brief Sets the 3x3 camera intrinsic matrix (CV_64F).
    UndistortMap& K(cv::Mat k)       { K_     = std::move(k); return *this; }
    /// @brief Sets the distortion coefficient vector.
    UndistortMap& dist(cv::Mat d)    { dist_  = std::move(d); return *this; }
    /// @brief Sets the new camera matrix (optional; defaults to K).
    UndistortMap& new_K(cv::Mat nk)  { new_K_ = std::move(nk); return *this; }
    /// @brief Sets the rectification rotation matrix (optional; defaults to identity).
    UndistortMap& R(cv::Mat r)       { R_     = std::move(r); return *this; }

    /// @brief Computes the undistortion maps for the given image size.
    [[nodiscard]] UndistortMapResult operator()(cv::Size image_size) const;

private:
    cv::Mat K_;
    cv::Mat dist_;
    cv::Mat new_K_;
    cv::Mat R_;
};

} // namespace improc::calib
