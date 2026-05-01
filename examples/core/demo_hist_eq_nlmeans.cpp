// examples/core/demo_hist_eq_nlmeans.cpp
// Demonstrates HistogramEqualization (contrast normalization) and NLMeansDenoising (noise reduction).

#include <format>
#include <iostream>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

int main() {
    // ── HistogramEqualization on Gray ────────────────────────────────────────
    cv::Mat dark_gray(64, 64, CV_8UC1, cv::Scalar(50));
    Image<Gray> gray{dark_gray};
    cv::Scalar mean_before = cv::mean(dark_gray);
    Image<Gray> eq_gray = gray | HistogramEqualization{};
    cv::Scalar mean_after = cv::mean(eq_gray.mat());
    std::cout << std::format("HistogramEqualization Gray: mean {:.1f} → {:.1f}\n",
        mean_before[0], mean_after[0]);

    // ── HistogramEqualization on BGR ─────────────────────────────────────────
    cv::Mat dark_bgr(64, 64, CV_8UC3, cv::Scalar(30, 40, 50));
    Image<BGR> bgr{dark_bgr};
    Image<BGR> eq_bgr = bgr | HistogramEqualization{};
    std::cout << std::format("HistogramEqualization BGR: {}x{} BGR\n",
        eq_bgr.cols(), eq_bgr.rows());

    // ── NLMeansDenoising on Gray ─────────────────────────────────────────────
    cv::Mat base_gray(64, 64, CV_32FC1, cv::Scalar(128.0f));
    cv::Mat noise_gray(64, 64, CV_32FC1);
    cv::randn(noise_gray, 0.0f, 30.0f);
    cv::Mat noisy_gray_f;
    cv::add(base_gray, noise_gray, noisy_gray_f);
    cv::Mat noisy_gray;
    noisy_gray_f.convertTo(noisy_gray, CV_8UC1);
    Image<Gray> noisy_gray_img{noisy_gray};

    cv::Scalar m_before, sd_before, m_after, sd_after;
    cv::meanStdDev(noisy_gray, m_before, sd_before);
    Image<Gray> denoised_gray = noisy_gray_img | NLMeansDenoising{}.h(10.0f);
    cv::meanStdDev(denoised_gray.mat(), m_after, sd_after);
    std::cout << std::format("NLMeansDenoising Gray: stddev {:.2f} → {:.2f}\n",
        sd_before[0], sd_after[0]);

    // ── NLMeansDenoising on BGR ──────────────────────────────────────────────
    cv::Mat base_bgr(64, 64, CV_32FC3, cv::Scalar(100.0f, 120.0f, 80.0f));
    cv::Mat noise_bgr(64, 64, CV_32FC3);
    cv::randn(noise_bgr, 0.0f, 30.0f);
    cv::Mat noisy_bgr_f;
    cv::add(base_bgr, noise_bgr, noisy_bgr_f);
    cv::Mat noisy_bgr;
    noisy_bgr_f.convertTo(noisy_bgr, CV_8UC3);
    Image<BGR> noisy_bgr_img{noisy_bgr};

    cv::Scalar m_bgr_b, sd_bgr_b;
    cv::meanStdDev(noisy_bgr, m_bgr_b, sd_bgr_b);
    Image<BGR> denoised_bgr = noisy_bgr_img | NLMeansDenoising{}.h(10.0f).h_color(10.0f);
    cv::Scalar m_bgr_a, sd_bgr_a;
    cv::meanStdDev(denoised_bgr.mat(), m_bgr_a, sd_bgr_a);
    std::cout << std::format("NLMeansDenoising BGR: stddev {:.2f} → {:.2f}\n",
        sd_bgr_b[0], sd_bgr_a[0]);

    // ── Pipeline: HistogramEqualization | NLMeansDenoising ───────────────────
    Image<Gray> smoothed = noisy_gray_img
        | HistogramEqualization{}
        | NLMeansDenoising{}.h(5.0f);
    std::cout << std::format("HistogramEqualization | NLMeansDenoising pipeline: {}x{}\n",
        smoothed.cols(), smoothed.rows());

    std::cout << "Done.\n";
    return 0;
}
