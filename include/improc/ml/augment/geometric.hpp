// include/improc/ml/augment/geometric.hpp
#pragma once

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>
#include <utility>
#include <vector>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/core/ops/axis.hpp"
#include "improc/ml/augment/detail.hpp"
#include "improc/ml/annotated.hpp"
#include "improc/exceptions.hpp"

namespace improc::ml {

using improc::core::AnyFormat;
using improc::core::Image;

namespace detail {

inline std::vector<cv::Point2f> bbox_corners(const cv::Rect2f& r) {
    return {
        {r.x,           r.y},
        {r.x + r.width, r.y},
        {r.x + r.width, r.y + r.height},
        {r.x,           r.y + r.height}
    };
}

inline cv::Rect2f corners_to_aabb(const std::vector<cv::Point2f>& corners) {
    float xmin = corners[0].x, xmax = corners[0].x;
    float ymin = corners[0].y, ymax = corners[0].y;
    for (const auto& c : corners) {
        xmin = std::min(xmin, c.x); xmax = std::max(xmax, c.x);
        ymin = std::min(ymin, c.y); ymax = std::max(ymax, c.y);
    }
    return {xmin, ymin, xmax - xmin, ymax - ymin};
}

inline cv::Rect2f clip_to_image(cv::Rect2f r, int W, int H) {
    float x1 = std::max(0.f, r.x);
    float y1 = std::max(0.f, r.y);
    float x2 = std::min(static_cast<float>(W), r.x + r.width);
    float y2 = std::min(static_cast<float>(H), r.y + r.height);
    if (x2 <= x1 || y2 <= y1) return {};
    return {x1, y1, x2 - x1, y2 - y1};
}

inline bool keep_box(cv::Rect2f clipped, cv::Rect2f original, float threshold) {
    float orig_area = original.area();
    if (orig_area <= 0.f || clipped.empty()) return false;
    return (clipped.area() / orig_area) >= threshold;
}

} // namespace detail

struct RandomFlip : detail::BindMixin<RandomFlip> {
    RandomFlip& p(float prob) {
        if (prob < 0.0f || prob > 1.0f)
            throw ParameterError{"p", std::format("must be in [0, 1], got {}", prob), "RandomFlip"};
        p_ = prob; return *this;
    }
    RandomFlip& axis(core::Axis a) { axis_ = a; return *this; }
    RandomFlip& min_area_ratio(float r) {
        if (r < 0.0f || r > 1.0f)
            throw ParameterError{"min_area_ratio", "must be in [0, 1]", "RandomFlip"};
        min_area_ratio_ = r; return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        std::bernoulli_distribution d(p_);
        if (!d(rng)) return img;
        int flip_code;
        switch (axis_) {
            case core::Axis::Horizontal: flip_code =  1; break;
            case core::Axis::Vertical:   flip_code =  0; break;
            case core::Axis::Both:       flip_code = -1; break;
            default: std::unreachable();
        }
        cv::Mat dst;
        cv::flip(img.mat(), dst, flip_code);
        return Image<Format>(std::move(dst));
    }

    template<AnyFormat Format>
    AnnotatedImage<Format> operator()(AnnotatedImage<Format> ann, std::mt19937& rng) const {
        std::bernoulli_distribution d(p_);
        if (!d(rng)) return ann;
        int W = ann.image.cols(), H = ann.image.rows();
        int flip_code;
        switch (axis_) {
            case core::Axis::Horizontal: flip_code =  1; break;
            case core::Axis::Vertical:   flip_code =  0; break;
            case core::Axis::Both:       flip_code = -1; break;
            default: std::unreachable();
        }
        cv::Mat dst;
        cv::flip(ann.image.mat(), dst, flip_code);
        ann.image = Image<Format>(std::move(dst));
        for (auto& bb : ann.boxes) {
            float x = bb.box.x, y = bb.box.y, w = bb.box.width, h = bb.box.height;
            if (flip_code == 1 || flip_code == -1) x = static_cast<float>(W) - x - w;
            if (flip_code == 0 || flip_code == -1) y = static_cast<float>(H) - y - h;
            bb.box = {x, y, w, h};
        }
        return ann;
    }

private:
    float      p_               = 0.5f;
    core::Axis axis_            = core::Axis::Horizontal;
    float      min_area_ratio_  = 0.1f;
};

struct RandomRotate : detail::BindMixin<RandomRotate> {
    RandomRotate& range(float min_deg, float max_deg) {
        if (min_deg > max_deg)
            throw ParameterError{"min_deg", "must be <= max_deg", "RandomRotate"};
        min_deg_ = min_deg; max_deg_ = max_deg; return *this;
    }
    RandomRotate& scale(float s) {
        if (s <= 0.0f)
            throw ParameterError{"scale", "must be > 0", "RandomRotate"};
        scale_ = s; return *this;
    }
    RandomRotate& min_area_ratio(float r) {
        if (r < 0.0f || r > 1.0f)
            throw ParameterError{"min_area_ratio", "must be in [0, 1]", "RandomRotate"};
        min_area_ratio_ = r; return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        std::uniform_real_distribution<float> d(min_deg_, max_deg_);
        float angle = d(rng);
        cv::Point2f center(img.cols() / 2.0f, img.rows() / 2.0f);
        cv::Mat M = cv::getRotationMatrix2D(center, angle, scale_);
        cv::Mat dst;
        try {
            cv::warpAffine(img.mat(), dst, M, img.mat().size());
        } catch (const cv::Exception& e) {
            throw AugmentError{"RandomRotate: " + std::string(e.what())};
        }
        return Image<Format>(std::move(dst));
    }

private:
    float min_deg_         = -15.0f;
    float max_deg_         =  15.0f;
    float scale_           =   1.0f;
    float min_area_ratio_  =   0.1f;
};

struct RandomCrop : detail::BindMixin<RandomCrop> {
    RandomCrop& width(int w) {
        if (w <= 0) throw ParameterError{"width", "must be positive", "RandomCrop"};
        width_ = w; return *this;
    }
    RandomCrop& height(int h) {
        if (h <= 0) throw ParameterError{"height", "must be positive", "RandomCrop"};
        height_ = h; return *this;
    }
    RandomCrop& min_area_ratio(float r) {
        if (r < 0.0f || r > 1.0f)
            throw ParameterError{"min_area_ratio", "must be in [0, 1]", "RandomCrop"};
        min_area_ratio_ = r; return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        if (width_ <= 0 || height_ <= 0)
            throw ParameterError{"width/height", "both must be set", "RandomCrop"};
        if (width_ > img.cols() || height_ > img.rows())
            throw ParameterError{"width/height",
                std::format("crop {}x{} exceeds image {}x{}",
                    width_, height_, img.cols(), img.rows()),
                "RandomCrop"};
        std::uniform_int_distribution<int> x_d(0, img.cols() - width_);
        std::uniform_int_distribution<int> y_d(0, img.rows() - height_);
        int x = x_d(rng);
        int y = y_d(rng);
        cv::Mat dst = img.mat()(cv::Rect(x, y, width_, height_)).clone();
        return Image<Format>(std::move(dst));
    }

    template<AnyFormat Format>
    AnnotatedImage<Format> operator()(AnnotatedImage<Format> ann, std::mt19937& rng) const {
        if (width_ <= 0 || height_ <= 0)
            throw ParameterError{"width/height", "both must be set", "RandomCrop"};
        if (width_ > ann.image.cols() || height_ > ann.image.rows())
            throw ParameterError{"width/height",
                std::format("crop {}x{} exceeds image {}x{}",
                    width_, height_, ann.image.cols(), ann.image.rows()),
                "RandomCrop"};
        std::uniform_int_distribution<int> x_d(0, ann.image.cols() - width_);
        std::uniform_int_distribution<int> y_d(0, ann.image.rows() - height_);
        int cx = x_d(rng), cy = y_d(rng);
        cv::Mat dst = ann.image.mat()(cv::Rect(cx, cy, width_, height_)).clone();
        ann.image = Image<Format>(std::move(dst));
        std::vector<BBox> kept;
        for (auto& bb : ann.boxes) {
            cv::Rect2f shifted{bb.box.x - cx, bb.box.y - cy, bb.box.width, bb.box.height};
            cv::Rect2f clipped = detail::clip_to_image(shifted, width_, height_);
            if (detail::keep_box(clipped, bb.box, min_area_ratio_)) {
                bb.box = clipped;
                kept.push_back(std::move(bb));
            }
        }
        ann.boxes = std::move(kept);
        return ann;
    }

private:
    int   width_           = 0;
    int   height_          = 0;
    float min_area_ratio_  = 0.1f;
};

struct RandomResize : detail::BindMixin<RandomResize> {
    RandomResize& range(int min_side, int max_side) {
        if (min_side <= 0)
            throw ParameterError{"min_side", "must be > 0", "RandomResize"};
        if (min_side > max_side)
            throw ParameterError{"min_side", "must be <= max_side", "RandomResize"};
        min_side_ = min_side; max_side_ = max_side; return *this;
    }
    RandomResize& min_area_ratio(float r) {
        if (r < 0.0f || r > 1.0f)
            throw ParameterError{"min_area_ratio", "must be in [0, 1]", "RandomResize"};
        min_area_ratio_ = r; return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        std::uniform_int_distribution<int> d(min_side_, max_side_);
        int target = d(rng);
        const int h = img.rows();
        const int w = img.cols();
        int new_w, new_h;
        if (h <= w) {
            new_h = target;
            new_w = static_cast<int>(std::round(static_cast<double>(w) * target / h));
        } else {
            new_w = target;
            new_h = static_cast<int>(std::round(static_cast<double>(h) * target / w));
        }
        cv::Mat dst;
        cv::resize(img.mat(), dst, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
        return Image<Format>(std::move(dst));
    }

    template<AnyFormat Format>
    AnnotatedImage<Format> operator()(AnnotatedImage<Format> ann, std::mt19937& rng) const {
        std::uniform_int_distribution<int> d(min_side_, max_side_);
        int target = d(rng);
        const int h = ann.image.rows(), w = ann.image.cols();
        int new_w, new_h;
        if (h <= w) {
            new_h = target;
            new_w = static_cast<int>(std::round(static_cast<double>(w) * target / h));
        } else {
            new_w = target;
            new_h = static_cast<int>(std::round(static_cast<double>(h) * target / w));
        }
        float sx = static_cast<float>(new_w) / static_cast<float>(w);
        float sy = static_cast<float>(new_h) / static_cast<float>(h);
        cv::Mat dst;
        cv::resize(ann.image.mat(), dst, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
        ann.image = Image<Format>(std::move(dst));
        for (auto& bb : ann.boxes)
            bb.box = {bb.box.x * sx, bb.box.y * sy, bb.box.width * sx, bb.box.height * sy};
        return ann;
    }

private:
    int   min_side_        = 224;
    int   max_side_        = 256;
    float min_area_ratio_  = 0.1f;
};

struct RandomZoom : detail::BindMixin<RandomZoom> {
    RandomZoom& range(float min_scale, float max_scale) {
        if (min_scale <= 0.0f || max_scale <= 0.0f || min_scale > 1.0f || max_scale > 1.0f)
            throw ParameterError{"min_scale/max_scale", "must be in (0, 1]", "RandomZoom"};
        if (min_scale > max_scale)
            throw ParameterError{"min_scale", "must be <= max_scale", "RandomZoom"};
        min_scale_ = min_scale; max_scale_ = max_scale; return *this;
    }
    RandomZoom& min_area_ratio(float r) {
        if (r < 0.0f || r > 1.0f)
            throw ParameterError{"min_area_ratio", "must be in [0, 1]", "RandomZoom"};
        min_area_ratio_ = r; return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        int W = img.cols(), H = img.rows();
        std::uniform_real_distribution<float> d(min_scale_, max_scale_);
        float scale = d(rng);
        int cw = std::max(1, static_cast<int>(W * scale));
        int ch = std::max(1, static_cast<int>(H * scale));
        std::uniform_int_distribution<int> xd(0, W - cw);
        std::uniform_int_distribution<int> yd(0, H - ch);
        int x = xd(rng), y = yd(rng);
        cv::Mat crop = img.mat()(cv::Rect(x, y, cw, ch));
        cv::Mat dst;
        cv::resize(crop, dst, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
        return Image<Format>(std::move(dst));
    }

    template<AnyFormat Format>
    AnnotatedImage<Format> operator()(AnnotatedImage<Format> ann, std::mt19937& rng) const {
        int W = ann.image.cols(), H = ann.image.rows();
        std::uniform_real_distribution<float> d(min_scale_, max_scale_);
        float scale = d(rng);
        int cw = std::max(1, static_cast<int>(W * scale));
        int ch = std::max(1, static_cast<int>(H * scale));
        std::uniform_int_distribution<int> xd(0, W - cw);
        std::uniform_int_distribution<int> yd(0, H - ch);
        int zx = xd(rng), zy = yd(rng);
        float sx = static_cast<float>(W) / static_cast<float>(cw);
        float sy = static_cast<float>(H) / static_cast<float>(ch);
        cv::Mat crop = ann.image.mat()(cv::Rect(zx, zy, cw, ch));
        cv::Mat dst;
        cv::resize(crop, dst, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
        ann.image = Image<Format>(std::move(dst));
        std::vector<BBox> kept;
        for (auto& bb : ann.boxes) {
            cv::Rect2f transformed{
                (bb.box.x - zx) * sx,
                (bb.box.y - zy) * sy,
                bb.box.width  * sx,
                bb.box.height * sy
            };
            cv::Rect2f clipped = detail::clip_to_image(transformed, W, H);
            if (detail::keep_box(clipped, transformed, min_area_ratio_)) {
                bb.box = clipped;
                kept.push_back(std::move(bb));
            }
        }
        ann.boxes = std::move(kept);
        return ann;
    }

private:
    float min_scale_       = 0.7f;
    float max_scale_       = 1.0f;
    float min_area_ratio_  = 0.1f;
};

struct RandomShear : detail::BindMixin<RandomShear> {
    RandomShear& range(float min_deg, float max_deg) {
        if (min_deg > max_deg)
            throw ParameterError{"min_deg", "must be <= max_deg", "RandomShear"};
        min_deg_ = min_deg; max_deg_ = max_deg; return *this;
    }
    RandomShear& axis(core::Axis a) {
        if (a == core::Axis::Both)
            throw ParameterError{"axis", "Axis::Both is not supported; use Horizontal or Vertical", "RandomShear"};
        axis_ = a; return *this;
    }
    RandomShear& min_area_ratio(float r) {
        if (r < 0.0f || r > 1.0f)
            throw ParameterError{"min_area_ratio", "must be in [0, 1]", "RandomShear"};
        min_area_ratio_ = r; return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        std::uniform_real_distribution<float> d(min_deg_, max_deg_);
        float angle_rad = d(rng) * std::numbers::pi_v<float> / 180.0f;
        float shear = std::tan(angle_rad);
        int W = img.cols(), H = img.rows();
        cv::Mat M = cv::Mat::eye(2, 3, CV_32F);
        if (axis_ == core::Axis::Horizontal)
            M.at<float>(0, 1) = shear;
        else
            M.at<float>(1, 0) = shear;
        cv::Mat dst;
        cv::warpAffine(img.mat(), dst, M, cv::Size(W, H),
                       cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        return Image<Format>(std::move(dst));
    }

private:
    float      min_deg_        = -15.0f;
    float      max_deg_        =  15.0f;
    core::Axis axis_           = core::Axis::Horizontal;
    float      min_area_ratio_ =   0.1f;
};

struct RandomPerspective : detail::BindMixin<RandomPerspective> {
    RandomPerspective& distortion_scale(float s) {
        if (s < 0.0f || s > 1.0f)
            throw ParameterError{"distortion_scale", "must be in [0, 1]", "RandomPerspective"};
        distortion_scale_ = s; return *this;
    }
    RandomPerspective& min_area_ratio(float r) {
        if (r < 0.0f || r > 1.0f)
            throw ParameterError{"min_area_ratio", "must be in [0, 1]", "RandomPerspective"};
        min_area_ratio_ = r; return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        int W = img.cols(), H = img.rows();
        float half_range = distortion_scale_ * std::min(W, H) / 2.0f;
        std::uniform_real_distribution<float> d(-half_range, half_range);
        std::vector<cv::Point2f> src_pts = {
            {0.f, 0.f},
            {static_cast<float>(W), 0.f},
            {0.f, static_cast<float>(H)},
            {static_cast<float>(W), static_cast<float>(H)}
        };
        std::vector<cv::Point2f> dst_pts;
        dst_pts.reserve(4);
        for (const auto& p : src_pts)
            dst_pts.push_back({p.x + d(rng), p.y + d(rng)});
        cv::Mat M = cv::getPerspectiveTransform(src_pts, dst_pts);
        cv::Mat dst;
        cv::warpPerspective(img.mat(), dst, M, cv::Size(W, H));
        return Image<Format>(std::move(dst));
    }

private:
    float distortion_scale_ = 0.5f;
    float min_area_ratio_   = 0.1f;
};

} // namespace improc::ml
