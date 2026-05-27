// src/core/ops/img_hash.cpp
// Perceptual hash algorithms implemented using only standard OpenCV (no contrib).
#include "improc/core/ops/img_hash.hpp"
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace improc::core {

namespace {

// Hamming distance: count differing bits between two CV_8U hash mats.
double hamming(const cv::Mat& a, const cv::Mat& b) {
    CV_Assert(a.type() == CV_8U && b.type() == CV_8U && a.total() == b.total());
    cv::Mat diff;
    cv::bitwise_xor(a, b, diff);
    int bits = 0;
    for (size_t i = 0; i < diff.total(); ++i)
        bits += __builtin_popcount(diff.at<uint8_t>(i));
    return static_cast<double>(bits);
}

// Pack a flat bool vector into a CV_8U row mat (little-endian per byte).
cv::Mat pack_bits(const std::vector<bool>& bits) {
    int bytes = static_cast<int>((bits.size() + 7) / 8);
    cv::Mat hash(1, bytes, CV_8U, cv::Scalar(0));
    for (size_t i = 0; i < bits.size(); ++i)
        if (bits[i]) hash.at<uint8_t>(0, i / 8) |= (1u << (i % 8));
    return hash;
}

cv::Mat to_gray_resized(const Image<BGR>& img, int size) {
    cv::Mat gray, resized;
    cv::cvtColor(img.mat(), gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, resized, {size, size}, 0, 0, cv::INTER_AREA);
    return resized;
}

} // namespace

// ── AverageHash ───────────────────────────────────────────────────────────────
// Resize to 8×8 gray. Bit = pixel > mean. 64 bits = 8 bytes.

cv::Mat AverageHash::operator()(const Image<BGR>& img) const {
    cv::Mat small = to_gray_resized(img, 8);
    double mean = cv::mean(small)[0];
    std::vector<bool> bits;
    bits.reserve(64);
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c)
            bits.push_back(small.at<uint8_t>(r, c) > mean);
    return pack_bits(bits);
}
double AverageHash::distance(const cv::Mat& a, const cv::Mat& b) {
    return hamming(a, b);
}

// ── PHash ─────────────────────────────────────────────────────────────────────
// Resize to 32×32, apply DCT, take 8×8 top-left (skip DC), bit = value > mean.
// 63 meaningful bits packed into 8 bytes.

cv::Mat PHash::operator()(const Image<BGR>& img) const {
    cv::Mat small = to_gray_resized(img, 32);
    cv::Mat floatMat, dctMat;
    small.convertTo(floatMat, CV_32F);
    cv::dct(floatMat, dctMat);

    cv::Mat top = dctMat(cv::Rect(0, 0, 8, 8)).clone();
    // Mean of 63 values (skip DC at 0,0)
    double sum = cv::sum(top)[0] - top.at<float>(0, 0);
    double mean = sum / 63.0;

    std::vector<bool> bits;
    bits.reserve(64);
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c) {
            if (r == 0 && c == 0) continue;
            bits.push_back(top.at<float>(r, c) > mean);
        }
    bits.push_back(false); // pad to 64 bits
    return pack_bits(bits);
}
double PHash::distance(const cv::Mat& a, const cv::Mat& b) {
    return hamming(a, b);
}

// ── MarrHildrethHash ──────────────────────────────────────────────────────────
// Apply Laplacian of Gaussian, encode zero-crossing sign pattern.
// 576 bits (24×24 sign map) = 72 bytes.

cv::Mat MarrHildrethHash::operator()(const Image<BGR>& img) const {
    cv::Mat gray, resized, blurred, laplacian;
    cv::cvtColor(img.mat(), gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, resized, {24, 24}, 0, 0, cv::INTER_AREA);
    resized.convertTo(resized, CV_64F);
    // LoG: Gaussian blur then Laplacian
    cv::GaussianBlur(resized, blurred, {5, 5}, 1.0);
    cv::Laplacian(blurred, laplacian, CV_64F, 3);

    // Encode sign of each pixel as a bit
    std::vector<bool> bits;
    bits.reserve(576);
    for (int r = 0; r < 24; ++r)
        for (int c = 0; c < 24; ++c)
            bits.push_back(laplacian.at<double>(r, c) >= 0);
    return pack_bits(bits);
}
double MarrHildrethHash::distance(const cv::Mat& a, const cv::Mat& b) {
    return hamming(a, b);
}

// ── RadialVarianceHash ────────────────────────────────────────────────────────
// Sample 40 radial lines from the center, compute variance per line.
// Result: 1×40 CV_64F. Distance: L2 norm.

cv::Mat RadialVarianceHash::operator()(const Image<BGR>& img) const {
    cv::Mat gray, resized;
    cv::cvtColor(img.mat(), gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, resized, {64, 64}, 0, 0, cv::INTER_AREA);
    resized.convertTo(resized, CV_64F);
    resized /= 255.0;

    cv::Mat hash(1, 40, CV_64F);
    const double cx = 32.0, cy = 32.0, radius = 32.0;

    for (int i = 0; i < 40; ++i) {
        double angle = M_PI * i / 40.0;
        double dx = std::cos(angle), dy = std::sin(angle);
        std::vector<double> samples;
        samples.reserve(64);
        for (double r = 0; r <= radius; r += 1.0) {
            int x = static_cast<int>(cx + r * dx + 0.5);
            int y = static_cast<int>(cy + r * dy + 0.5);
            if (x >= 0 && x < 64 && y >= 0 && y < 64)
                samples.push_back(resized.at<double>(y, x));
        }
        double mean = 0;
        for (double v : samples) mean += v;
        mean /= static_cast<double>(samples.size());
        double var = 0;
        for (double v : samples) var += (v - mean) * (v - mean);
        var /= static_cast<double>(samples.size());
        hash.at<double>(0, i) = var;
    }
    return hash;
}
double RadialVarianceHash::distance(const cv::Mat& a, const cv::Mat& b) {
    return cv::norm(a, b, cv::NORM_L2);
}

// ── ColorMomentHash ───────────────────────────────────────────────────────────
// Compute 14 statistical values per channel of YCrCb (3 × 14 = 42).
// Result: 1×42 CV_64F. Distance: L2 norm.

cv::Mat ColorMomentHash::operator()(const Image<BGR>& img) const {
    cv::Mat resized, ycrcb;
    cv::resize(img.mat(), resized, {16, 16}, 0, 0, cv::INTER_AREA);
    cv::cvtColor(resized, ycrcb, cv::COLOR_BGR2YCrCb);

    cv::Mat hash(1, 42, CV_64F, cv::Scalar(0));
    std::vector<cv::Mat> channels;
    cv::split(ycrcb, channels);

    for (int ch = 0; ch < 3; ++ch) {
        cv::Mat ch_f;
        channels[ch].convertTo(ch_f, CV_64F);
        ch_f /= 255.0;

        cv::Scalar mu_s, sigma_s;
        cv::meanStdDev(ch_f, mu_s, sigma_s);
        double mu = mu_s[0];
        double sigma = sigma_s[0];

        // Central moments (3rd–6th)
        double m3 = 0, m4 = 0, m5 = 0, m6 = 0;
        int n = static_cast<int>(ch_f.total());
        ch_f = ch_f.reshape(1, 1);
        for (int i = 0; i < n; ++i) {
            double d = ch_f.at<double>(0, i) - mu;
            double d2 = d * d;
            m3 += d2 * d;
            m4 += d2 * d2;
            m5 += d2 * d2 * d;
            m6 += d2 * d2 * d2;
        }
        m3 /= n; m4 /= n; m5 /= n; m6 /= n;
        double s3 = (sigma > 1e-10) ? sigma * sigma * sigma : 1.0;
        double s4 = s3 * sigma;

        // 8-bin histogram
        float rng[] = {0.f, 1.f};
        const float* ranges[] = {rng};
        int bins = 8;
        cv::Mat hist, ch_32f;
        ch_f.convertTo(ch_32f, CV_32F);
        cv::calcHist(&ch_32f, 1, nullptr, cv::Mat(), hist, 1, &bins, ranges);
        hist /= static_cast<float>(n);

        int base = ch * 14;
        hash.at<double>(0, base + 0) = mu;
        hash.at<double>(0, base + 1) = sigma;
        hash.at<double>(0, base + 2) = (sigma > 1e-10) ? m3 / s3 : 0.0;
        hash.at<double>(0, base + 3) = (sigma > 1e-10) ? m4 / s4 : 0.0;
        hash.at<double>(0, base + 4) = m5;
        hash.at<double>(0, base + 5) = m6;
        for (int b = 0; b < 8; ++b)
            hash.at<double>(0, base + 6 + b) = hist.at<float>(b);
    }
    return hash;
}
double ColorMomentHash::distance(const cv::Mat& a, const cv::Mat& b) {
    return cv::norm(a, b, cv::NORM_L2);
}

// ── BlockMeanHash ─────────────────────────────────────────────────────────────
// Resize to 256×256, divide into 16×16 blocks of 16×16 pixels.
// Bit = block mean > overall mean. 256 bits = 32 bytes.

cv::Mat BlockMeanHash::operator()(const Image<BGR>& img) const {
    cv::Mat gray, resized;
    cv::cvtColor(img.mat(), gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, resized, {256, 256}, 0, 0, cv::INTER_AREA);
    resized.convertTo(resized, CV_64F);

    double overall_mean = cv::mean(resized)[0];

    std::vector<bool> bits;
    bits.reserve(256);
    for (int br = 0; br < 16; ++br) {
        for (int bc = 0; bc < 16; ++bc) {
            cv::Mat block = resized(cv::Rect(bc * 16, br * 16, 16, 16));
            bits.push_back(cv::mean(block)[0] > overall_mean);
        }
    }
    return pack_bits(bits);
}
double BlockMeanHash::distance(const cv::Mat& a, const cv::Mat& b) {
    return hamming(a, b);
}

} // namespace improc::core
