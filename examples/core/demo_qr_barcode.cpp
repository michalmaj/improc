// examples/core/demo_qr_barcode.cpp
//
// Demo: DetectQR, DetectBarcode
//
// Generates a synthetic QR-like image using cv::QRCodeEncoder (if available).
// Detects and prints the decoded string and polygon corners.
// If QRCodeEncoder is unavailable, prints a skip notice.
// Press any key to close.

#include "improc/core/pipeline.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace improc::core;

int main() {
    // ── Generate a QR code image ──────────────────────────────────────────────
    const std::string content = "https://github.com/michalmaj/improc++";
    cv::Mat qr_mat;

    auto encoder = cv::QRCodeEncoder::create();
    encoder->encode(content, qr_mat);

    if (qr_mat.empty()) {
        std::cout << "cv::QRCodeEncoder unavailable — skipping QR demo\n";
    } else {
        // Embed QR code in a padded white canvas for better detection
        cv::Mat canvas(400, 400, CV_8UC1, cv::Scalar(255));
        cv::resize(qr_mat, qr_mat, {300, 300}, 0, 0, cv::INTER_NEAREST);
        qr_mat.copyTo(canvas(cv::Rect{50, 50, 300, 300}));

        cv::Mat canvas_bgr;
        cv::cvtColor(canvas, canvas_bgr, cv::COLOR_GRAY2BGR);
        Image<BGR> qr_img(canvas_bgr);

        cv::imshow("1 — Generated QR code", canvas_bgr);
        cv::waitKey(0);

        // ── DetectQR ─────────────────────────────────────────────────────────
        QRResult qr = qr_img | DetectQR{};
        std::cout << "QR codes detected: " << qr.size() << "\n";
        for (std::size_t i = 0; i < qr.size(); ++i) {
            std::cout << "  [" << i << "] decoded: \"" << qr.decoded[i] << "\"\n";

            // Draw the 4-corner polygon
            if (!qr.points[i].empty()) {
                std::vector<cv::Point> corners;
                for (int k = 0; k < qr.points[i].rows; ++k) {
                    auto pt = qr.points[i].at<cv::Point2f>(k, 0);
                    corners.push_back({(int)pt.x, (int)pt.y});
                }
                cv::polylines(canvas_bgr, corners, true, {0, 0, 255}, 2);
            }
        }
        cv::imshow("2 — QR detection result", canvas_bgr);
        cv::waitKey(0);
    }

    // ── DetectBarcode — works with ISBN, EAN, UPC, etc. ──────────────────────
    // (barcode detection on a plain grey image will find no barcodes — that is expected)
    cv::Mat plain(200, 400, CV_8UC3, cv::Scalar(200, 200, 200));
    Image<BGR> plain_img(plain);
    BarcodeResult bc = plain_img | DetectBarcode{};
    std::cout << "Barcodes in plain grey image: " << bc.size()
              << " (expected: 0)\n";

    std::cout << "Done.\n";
    return 0;
}
