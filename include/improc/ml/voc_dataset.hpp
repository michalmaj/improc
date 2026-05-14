// include/improc/ml/voc_dataset.hpp
#pragma once

#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/ml/annotated.hpp"
#include "improc/error.hpp"
#include "improc/exceptions.hpp"

namespace improc::ml {

using improc::core::Image;
using improc::core::BGR;

/**
 * @brief Parses one Pascal VOC XML annotation file and loads the corresponding image.
 *
 * @param xml_path       Path to the `.xml` annotation file.
 * @param images_dir     Directory containing the image files.
 * @param class_map      String-to-int class mapping, mutated in-place.
 *                       If `filter_unknown` is false (default), unknown class names are
 *                       auto-assigned new ids. If true, unknown classes are silently dropped.
 * @param skip_difficult Skip objects with `<difficult>1</difficult>` (default: true).
 * @param filter_unknown Drop objects whose class name is not already in `class_map` (default: false).
 * @return Parsed `AnnotatedImage<BGR>` or an `improc::Error`.
 */
std::expected<AnnotatedImage<BGR>, improc::Error>
parse_voc_xml(const std::filesystem::path& xml_path,
              const std::filesystem::path& images_dir,
              std::unordered_map<std::string, int>& class_map,
              bool skip_difficult  = true,
              bool filter_unknown  = false);

/**
 * @brief Loads a Pascal VOC annotation dataset into train/val/test splits of `AnnotatedImage<BGR>`.
 *
 * Supports two split modes:
 * - **VOC split**: uses `ImageSets/Main/train.txt`, `val.txt`, `test.txt` when present.
 * - **Random split**: shuffles all XMLs in `Annotations/` and splits by ratio when `ImageSets/` is absent.
 *
 * @code
 * VocDataset ds;
 * ds.classes({"cat", "dog"}).skip_difficult(true);
 * auto ok = ds.load_from_directory("data/VOC2012");
 * for (const auto& ann : ds.train()) { ... }
 * @endcode
 */
class VocDataset {
public:
    /// @brief Fix class order and filter unknown classes. Optional — auto-assign if not called.
    VocDataset& classes(std::vector<std::string> cls);

    /// @brief Skip objects with `<difficult>1</difficult>` (default: true).
    VocDataset& skip_difficult(bool v) { skip_difficult_ = v; return *this; }

    /// @brief Proportion of data used as test set in random-split mode (default: 0.2).
    /// @throws improc::ParameterError if not in [0, 1).
    VocDataset& test_ratio(float r);

    /// @brief Proportion of data used as validation set in random-split mode (default: 0.1).
    /// @throws improc::ParameterError if not in [0, 1).
    VocDataset& val_ratio(float r);

    /// @brief Seed for the RNG used in random-split mode.
    VocDataset& shuffle_seed(unsigned int s) { shuffle_seed_ = s; return *this; }

    /// @brief Loads the dataset from a VOC-structured directory.
    std::expected<void, improc::Error>
    load_from_directory(const std::filesystem::path& root);

    const std::vector<AnnotatedImage<BGR>>& train() const { return train_; }
    const std::vector<AnnotatedImage<BGR>>& val()   const { return val_;   }
    const std::vector<AnnotatedImage<BGR>>& test()  const { return test_;  }

    /// @brief Returns the string→int class mapping built during loading.
    const std::unordered_map<std::string, int>& class_mapping() const { return class_map_; }

    /// @brief Returns the class name for the given integer id.
    /// @throws std::out_of_range if the id is unknown.
    std::string class_name_for(int id) const { return id_to_class_.at(id); }

private:
    std::vector<std::string>              user_classes_;
    bool                                  skip_difficult_ = true;
    float                                 test_ratio_     = 0.2f;
    float                                 val_ratio_      = 0.1f;
    std::optional<unsigned int>           shuffle_seed_;

    std::vector<AnnotatedImage<BGR>>      train_, val_, test_;
    std::unordered_map<std::string, int>  class_map_;
    std::unordered_map<int, std::string>  id_to_class_;

    std::expected<std::vector<AnnotatedImage<BGR>>, improc::Error>
    load_stems(const std::vector<std::string>& stems,
               const std::filesystem::path& images_dir,
               const std::filesystem::path& annotations_dir);
};

} // namespace improc::ml
