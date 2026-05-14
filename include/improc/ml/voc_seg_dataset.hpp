// include/improc/ml/voc_seg_dataset.hpp
#pragma once

#include <expected>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>
#include "improc/ml/segmented.hpp"
#include "improc/error.hpp"

namespace improc::ml {

using improc::core::BGR;

/**
 * @brief Parses one VOC segmentation entry (image + class mask + optional instance mask).
 *
 * @param stem               Filename stem (e.g. "2007_000032"), without extension.
 * @param images_dir         Directory containing JPEG/PNG images.
 * @param class_masks_dir    Directory containing class segmentation PNGs.
 * @param instance_masks_dir Directory containing instance segmentation PNGs, or empty to skip.
 * @return Parsed `SegmentedImage<BGR>` or an `improc::Error`.
 */
std::expected<SegmentedImage<BGR>, improc::Error>
parse_voc_seg(const std::filesystem::path& stem,
              const std::filesystem::path& images_dir,
              const std::filesystem::path& class_masks_dir,
              const std::filesystem::path& instance_masks_dir);

/**
 * @brief Loads a Pascal VOC segmentation dataset.
 */
class VocSegDataset {
public:
    /// @brief Map index → class name (index = class_id pixel value).
    /// Optional — if not called, class_name_for() throws for any id.
    VocSegDataset& classes(std::vector<std::string> cls);

    /// @brief Load instance masks from SegmentationObject/ (default: false).
    VocSegDataset& load_instance_masks(bool v) { load_instance_ = v; return *this; }

    /// @brief Load from a VOC-structured directory root.
    std::expected<void, improc::Error>
    load_from_directory(const std::filesystem::path& root);

    const std::vector<SegmentedImage<BGR>>& train() const { return train_; }
    const std::vector<SegmentedImage<BGR>>& val()   const { return val_;   }
    const std::vector<SegmentedImage<BGR>>& test()  const { return test_;  }

    const std::unordered_map<int, std::string>& class_mapping() const { return id_to_class_; }

    /// @throws std::out_of_range if id has no name.
    std::string class_name_for(int id) const { return id_to_class_.at(id); }

private:
    bool load_instance_ = false;
    std::vector<SegmentedImage<BGR>>   train_, val_, test_;
    std::unordered_map<int, std::string> id_to_class_;

    std::expected<std::vector<SegmentedImage<BGR>>, improc::Error>
    load_stems(const std::vector<std::string>& stems,
               const std::filesystem::path& root);
};

} // namespace improc::ml
