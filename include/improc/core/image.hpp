// include/improc/core/image.hpp
#pragma once

#include <opencv2/core.hpp>
#include "improc/core/format_traits.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/**
 * @brief Type-safe image wrapper over `cv::Mat`.
 *
 * Enforces format at construction: passing a `cv::Mat` of the wrong type
 * throws `FormatError`. Shallow-copy semantics match `cv::Mat`; use `.clone()`
 * for a deep copy.
 *
 * @tparam Format  A format tag (`BGR`, `Gray`, `BGRA`, `HSV`, `Float32`, `Float32C3`).
 *
 * @throws ParameterError if the mat is empty.
 * @throws FormatError    if `mat.type()` does not match `FormatTraits<Format>::cv_type`.
 *
 * @code
 * cv::Mat raw = cv::imread("photo.png");
 * Image<BGR> img(raw);
 * Image<Gray> gray = img | ToGray{};
 * @endcode
 */
template<AnyFormat Format>
class Image {
public:
    explicit Image(cv::Mat mat) : mat_(std::move(mat)) {
        if (mat_.empty())
            throw ParameterError{"mat", "must not be empty", "Image constructor"};
        if (mat_.type() != FormatTraits<Format>::cv_type)
            throw FormatError{
                std::string(FormatTraits<Format>::name),
                cv_type_to_name(mat_.type()),
                "Image constructor"};
    }

    Image clone() const { return Image(mat_.clone()); }

    [[nodiscard]] cv::Mat&       mat()       { return mat_; }
    [[nodiscard]] const cv::Mat& mat() const { return mat_; }

    [[nodiscard]] int  rows()  const { return mat_.rows; }
    [[nodiscard]] int  cols()  const { return mat_.cols; }
    [[nodiscard]] bool empty() const { return mat_.empty(); }

    Image(const Image&)            = default;
    Image& operator=(const Image&) = default;
    Image(Image&&)                 = default;
    Image& operator=(Image&&)      = default;

private:
    cv::Mat mat_;
};

} // namespace improc::core
