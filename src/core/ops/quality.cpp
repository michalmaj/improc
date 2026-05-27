// src/core/ops/quality.cpp
// Quality metrics implemented using only standard OpenCV (no contrib required).
#include "improc/core/ops/quality.hpp"
#include <cmath>

namespace improc::core {

namespace {

void check_size(cv::Size a, cv::Size b, const char* op) {
    if (a != b)
        throw improc::ParameterError{"cmp", "must have same size as ref", op};
}

double compute_mse(const cv::Mat& ref, const cv::Mat& cmp) {
    cv::Mat diff;
    cv::absdiff(ref, cmp, diff);
    diff.convertTo(diff, CV_64F);
    cv::multiply(diff, diff, diff);
    cv::Scalar s = cv::mean(diff);
    int channels = ref.channels();
    double sum = 0;
    for (int i = 0; i < channels; ++i) sum += s[i];
    return sum / channels;
}

// SSIM for single-channel CV_64F images.
double ssim_channel(const cv::Mat& a, const cv::Mat& b) {
    static const double C1 = 6.5025;   // (0.01 * 255)^2
    static const double C2 = 58.5225;  // (0.03 * 255)^2
    static const cv::Size ksize{11, 11};
    static const double sigma = 1.5;

    cv::Mat mu_a, mu_b;
    cv::GaussianBlur(a, mu_a, ksize, sigma);
    cv::GaussianBlur(b, mu_b, ksize, sigma);

    cv::Mat mu_a2 = mu_a.mul(mu_a);
    cv::Mat mu_b2 = mu_b.mul(mu_b);
    cv::Mat mu_ab = mu_a.mul(mu_b);

    cv::Mat sigma_a2, sigma_b2, sigma_ab;
    cv::GaussianBlur(a.mul(a), sigma_a2, ksize, sigma);
    sigma_a2 -= mu_a2;
    cv::GaussianBlur(b.mul(b), sigma_b2, ksize, sigma);
    sigma_b2 -= mu_b2;
    cv::GaussianBlur(a.mul(b), sigma_ab, ksize, sigma);
    sigma_ab -= mu_ab;

    cv::Mat num = (2.0 * mu_ab + C1).mul(2.0 * sigma_ab + C2);
    cv::Mat den = (mu_a2 + mu_b2 + C1).mul(sigma_a2 + sigma_b2 + C2);
    cv::Mat ssim_map;
    cv::divide(num, den, ssim_map);
    return cv::mean(ssim_map)[0];
}

double compute_ssim(const cv::Mat& ref, const cv::Mat& cmp) {
    std::vector<cv::Mat> ref_ch, cmp_ch;
    cv::split(ref, ref_ch);
    cv::split(cmp, cmp_ch);
    double sum = 0;
    for (size_t i = 0; i < ref_ch.size(); ++i) {
        cv::Mat a, b;
        ref_ch[i].convertTo(a, CV_64F);
        cmp_ch[i].convertTo(b, CV_64F);
        sum += ssim_channel(a, b);
    }
    return sum / static_cast<double>(ref_ch.size());
}

// GMSD: Gradient Magnitude Similarity Deviation.
// Computed on grayscale. Lower = better.
double compute_gmsd(const cv::Mat& ref, const cv::Mat& cmp) {
    cv::Mat ref_gray, cmp_gray;
    if (ref.channels() == 3) {
        cv::cvtColor(ref, ref_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(cmp, cmp_gray, cv::COLOR_BGR2GRAY);
    } else {
        ref_gray = ref;
        cmp_gray = cmp;
    }
    cv::Mat ref_f, cmp_f;
    ref_gray.convertTo(ref_f, CV_64F);
    cmp_gray.convertTo(cmp_f, CV_64F);

    // Prewitt filters
    cv::Mat kx = (cv::Mat_<double>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1) / 3.0;
    cv::Mat ky = (cv::Mat_<double>(3, 3) << -1, -1, -1,  0, 0, 0,  1, 1, 1) / 3.0;

    cv::Mat gx_r, gy_r, gx_c, gy_c;
    cv::filter2D(ref_f, gx_r, CV_64F, kx);
    cv::filter2D(ref_f, gy_r, CV_64F, ky);
    cv::filter2D(cmp_f, gx_c, CV_64F, kx);
    cv::filter2D(cmp_f, gy_c, CV_64F, ky);

    cv::Mat gm_r, gm_c;
    cv::sqrt(gx_r.mul(gx_r) + gy_r.mul(gy_r), gm_r);
    cv::sqrt(gx_c.mul(gx_c) + gy_c.mul(gy_c), gm_c);

    static const double C = 0.0026;
    cv::Mat gmsm = (2.0 * gm_r.mul(gm_c) + C) / (gm_r.mul(gm_r) + gm_c.mul(gm_c) + C);

    cv::Scalar mean_v, stddev_v;
    cv::meanStdDev(gmsm, mean_v, stddev_v);
    return stddev_v[0];
}

} // namespace

// ── PSNR ─────────────────────────────────────────────────────────────────────

double PSNR::operator()(const Image<BGR>& ref, const Image<BGR>& cmp) const {
    check_size(ref.mat().size(), cmp.mat().size(), "PSNR");
    double mse = compute_mse(ref.mat(), cmp.mat());
    if (mse < 1e-10) return std::numeric_limits<double>::infinity();
    return 10.0 * std::log10(255.0 * 255.0 / mse);
}
double PSNR::operator()(const Image<Gray>& ref, const Image<Gray>& cmp) const {
    check_size(ref.mat().size(), cmp.mat().size(), "PSNR");
    double mse = compute_mse(ref.mat(), cmp.mat());
    if (mse < 1e-10) return std::numeric_limits<double>::infinity();
    return 10.0 * std::log10(255.0 * 255.0 / mse);
}

// ── SSIM ─────────────────────────────────────────────────────────────────────

double SSIM::operator()(const Image<BGR>& ref, const Image<BGR>& cmp) const {
    check_size(ref.mat().size(), cmp.mat().size(), "SSIM");
    return compute_ssim(ref.mat(), cmp.mat());
}
double SSIM::operator()(const Image<Gray>& ref, const Image<Gray>& cmp) const {
    check_size(ref.mat().size(), cmp.mat().size(), "SSIM");
    return compute_ssim(ref.mat(), cmp.mat());
}

// ── GMSD ─────────────────────────────────────────────────────────────────────

double GMSD::operator()(const Image<BGR>& ref, const Image<BGR>& cmp) const {
    check_size(ref.mat().size(), cmp.mat().size(), "GMSD");
    return compute_gmsd(ref.mat(), cmp.mat());
}
double GMSD::operator()(const Image<Gray>& ref, const Image<Gray>& cmp) const {
    check_size(ref.mat().size(), cmp.mat().size(), "GMSD");
    return compute_gmsd(ref.mat(), cmp.mat());
}

// ── MSE ──────────────────────────────────────────────────────────────────────

double MSE::operator()(const Image<BGR>& ref, const Image<BGR>& cmp) const {
    check_size(ref.mat().size(), cmp.mat().size(), "MSE");
    return compute_mse(ref.mat(), cmp.mat());
}
double MSE::operator()(const Image<Gray>& ref, const Image<Gray>& cmp) const {
    check_size(ref.mat().size(), cmp.mat().size(), "MSE");
    return compute_mse(ref.mat(), cmp.mat());
}

} // namespace improc::core
