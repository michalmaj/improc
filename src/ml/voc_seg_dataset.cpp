// src/ml/voc_seg_dataset.cpp
#include "improc/ml/voc_seg_dataset.hpp"
#include <algorithm>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace improc::ml {

namespace {

cv::Mat load_mask(const std::filesystem::path& p) {
    cv::Mat raw = cv::imread(p.string(), cv::IMREAD_UNCHANGED);
    if (raw.empty()) return {};
    if (raw.channels() == 1) return raw;

    // VOC palette: index i has r,g,b computed by bit-interleaving of i
    // Build reverse LUT: packed BGR uint32 → class_id
    std::unordered_map<uint32_t, uint8_t> lut;
    lut.reserve(256);
    for (int i = 0; i < 256; ++i) {
        int r = 0, g = 0, b = 0, c = i;
        for (int j = 0; j < 8; ++j) {
            r |= ((c >> 0) & 1) << (7 - j);
            g |= ((c >> 1) & 1) << (7 - j);
            b |= ((c >> 2) & 1) << (7 - j);
            c >>= 3;
        }
        uint32_t key = static_cast<uint32_t>(b) |
                       (static_cast<uint32_t>(g) << 8) |
                       (static_cast<uint32_t>(r) << 16);
        lut[key] = static_cast<uint8_t>(i);
    }
    cv::Mat result(raw.size(), CV_8UC1, cv::Scalar(0));
    for (int row = 0; row < raw.rows; ++row) {
        for (int col = 0; col < raw.cols; ++col) {
            const auto& px = raw.at<cv::Vec3b>(row, col);
            uint32_t key = static_cast<uint32_t>(px[0]) |
                           (static_cast<uint32_t>(px[1]) << 8) |
                           (static_cast<uint32_t>(px[2]) << 16);
            auto it = lut.find(key);
            result.at<uint8_t>(row, col) = (it != lut.end()) ? it->second : 0;
        }
    }
    return result;
}

} // namespace

std::expected<SegmentedImage<BGR>, improc::Error>
parse_voc_seg(const std::filesystem::path& stem,
              const std::filesystem::path& images_dir,
              const std::filesystem::path& class_masks_dir,
              const std::filesystem::path& instance_masks_dir) {
    using namespace std::filesystem;

    // 1. Load image (.jpg first, .png fallback)
    path img_path = images_dir / (stem.string() + ".jpg");
    cv::Mat img = cv::imread(img_path.string(), cv::IMREAD_COLOR);
    if (img.empty()) {
        img_path = images_dir / (stem.string() + ".png");
        img = cv::imread(img_path.string(), cv::IMREAD_COLOR);
    }
    if (img.empty())
        return std::unexpected(improc::Error::voc_seg_parse_failed(
            img_path.string(), "image not found"));

    // 2. Load class mask
    path mask_path = class_masks_dir / (stem.string() + ".png");
    cv::Mat class_mat = load_mask(mask_path);
    if (class_mat.empty())
        return std::unexpected(improc::Error::voc_seg_parse_failed(
            mask_path.string(), "class mask not found"));

    // 3. Load instance mask (optional)
    std::optional<improc::core::Image<improc::core::Gray>> instance_mask;
    if (!instance_masks_dir.empty()) {
        path inst_path = instance_masks_dir / (stem.string() + ".png");
        cv::Mat inst_mat = load_mask(inst_path);
        if (inst_mat.empty())
            return std::unexpected(improc::Error::voc_seg_parse_failed(
                inst_path.string(), "instance mask not found"));
        instance_mask = improc::core::Image<improc::core::Gray>(std::move(inst_mat));
    }

    return SegmentedImage<BGR>{
        improc::core::Image<BGR>(std::move(img)),
        improc::core::Image<improc::core::Gray>(std::move(class_mat)),
        std::move(instance_mask)
    };
}

VocSegDataset& VocSegDataset::classes(std::vector<std::string> cls) {
    id_to_class_.clear();
    for (int i = 0; i < static_cast<int>(cls.size()); ++i)
        id_to_class_[i] = cls[i];
    return *this;
}

std::expected<std::vector<SegmentedImage<BGR>>, improc::Error>
VocSegDataset::load_stems(const std::vector<std::string>& stems,
                          const std::filesystem::path& root) {
    std::vector<SegmentedImage<BGR>> result;
    result.reserve(stems.size());
    auto inst_dir = load_instance_ ? root / "SegmentationObject"
                                   : std::filesystem::path{};
    for (const auto& stem : stems) {
        auto r = parse_voc_seg(stem,
                               root / "JPEGImages",
                               root / "SegmentationClass",
                               inst_dir);
        if (!r) return std::unexpected(r.error());
        result.push_back(std::move(*r));
    }
    return result;
}

std::expected<void, improc::Error>
VocSegDataset::load_from_directory(const std::filesystem::path& root) {
    namespace fs = std::filesystem;

    if (!fs::is_directory(root / "SegmentationClass"))
        return std::unexpected(improc::Error::voc_seg_parse_failed(
            root.string(), "SegmentationClass/ directory not found"));

    auto read_txt = [](const fs::path& p) {
        std::vector<std::string> stems;
        std::ifstream f(p);
        std::string line;
        while (std::getline(f, line)) {
            while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back())))
                line.pop_back();
            if (!line.empty()) stems.push_back(line);
        }
        return stems;
    };

    auto split_dir = root / "ImageSets" / "Segmentation";
    std::vector<std::string> train_stems, val_stems, test_stems;

    if (fs::exists(split_dir / "train.txt")) {
        train_stems = read_txt(split_dir / "train.txt");
        if (fs::exists(split_dir / "val.txt"))
            val_stems = read_txt(split_dir / "val.txt");
        if (fs::exists(split_dir / "test.txt"))
            test_stems = read_txt(split_dir / "test.txt");
    } else {
        for (const auto& entry : fs::directory_iterator(root / "SegmentationClass")) {
            if (entry.path().extension() == ".png")
                train_stems.push_back(entry.path().stem().string());
        }
        std::ranges::sort(train_stems);
        std::size_t n = train_stems.size();
        std::size_t n_test = std::max<std::size_t>(1, n / 10);
        std::size_t n_val  = std::max<std::size_t>(1, n / 10);
        if (n_test + n_val >= n) { n_test = 0; n_val = 0; }
        test_stems.assign(train_stems.end() - n_test, train_stems.end());
        val_stems.assign(train_stems.end() - n_test - n_val, train_stems.end() - n_test);
        train_stems.resize(n - n_test - n_val);
    }

    auto r = load_stems(train_stems, root);
    if (!r) return std::unexpected(r.error());
    train_ = std::move(*r);

    r = load_stems(val_stems, root);
    if (!r) return std::unexpected(r.error());
    val_ = std::move(*r);

    r = load_stems(test_stems, root);
    if (!r) return std::unexpected(r.error());
    test_ = std::move(*r);

    return {};
}

} // namespace improc::ml
