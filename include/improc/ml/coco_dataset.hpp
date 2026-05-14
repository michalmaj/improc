// include/improc/ml/coco_dataset.hpp
#pragma once

#include <expected>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/ml/annotated.hpp"
#include "improc/error.hpp"

namespace improc::ml {

using improc::core::Image;
using improc::core::BGR;

/**
 * @brief Parses one COCO-format JSON annotation file and loads all referenced images.
 *
 * @param json_path      Path to the COCO `.json` file.
 * @param images_dir     Directory containing the image files.
 * @param class_map      String-to-int class mapping, mutated in-place.
 *                       If `filter_unknown` is false (default), unknown class names are
 *                       auto-assigned new ids. If true, unknown classes are silently dropped.
 * @param skip_crowd     Skip annotations with `"iscrowd": 1` (default: true).
 * @param filter_unknown Drop annotations whose class name is not already in `class_map`.
 * @return Vector of `AnnotatedImage<BGR>` (one per image in the JSON), or an `improc::Error`.
 */
std::expected<std::vector<AnnotatedImage<BGR>>, improc::Error>
parse_coco_json(const std::filesystem::path& json_path,
                const std::filesystem::path& images_dir,
                std::unordered_map<std::string, int>& class_map,
                bool skip_crowd     = true,
                bool filter_unknown = false);

/**
 * @brief Loads COCO annotation datasets into train/val/test splits of `AnnotatedImage<BGR>`.
 *
 * Each split is loaded independently via an explicit `load_train/val/test` call.
 * All three splits share the same class mapping — ids are consistent across calls.
 *
 * @code
 * CocoDataset ds;
 * ds.classes({"cat", "dog"}).skip_crowd(true);
 * ds.load_train("instances_train.json", "images/train/");
 * ds.load_val("instances_val.json", "images/val/");
 * for (const auto& ann : ds.train()) { ... }
 * @endcode
 */
class CocoDataset {
public:
    /// @brief Fix class order and filter unknown classes. Optional — auto-assign if not called.
    /// MUST be called before the first load_* call; calling after a load has no effect.
    CocoDataset& classes(std::vector<std::string> cls);

    /// @brief Skip annotations with `"iscrowd": 1` (default: true).
    CocoDataset& skip_crowd(bool v) { skip_crowd_ = v; return *this; }

    /// @brief Load (or replace) the train split from a COCO JSON file.
    std::expected<void, improc::Error>
    load_train(const std::filesystem::path& json_path,
               const std::filesystem::path& images_dir);

    /// @brief Load (or replace) the val split from a COCO JSON file.
    std::expected<void, improc::Error>
    load_val(const std::filesystem::path& json_path,
             const std::filesystem::path& images_dir);

    /// @brief Load (or replace) the test split from a COCO JSON file.
    std::expected<void, improc::Error>
    load_test(const std::filesystem::path& json_path,
              const std::filesystem::path& images_dir);

    const std::vector<AnnotatedImage<BGR>>& train() const { return train_; }
    const std::vector<AnnotatedImage<BGR>>& val()   const { return val_;   }
    const std::vector<AnnotatedImage<BGR>>& test()  const { return test_;  }

    /// @brief Returns the string→int class mapping built during loading.
    const std::unordered_map<std::string, int>& class_mapping() const { return class_map_; }

    /// @brief Returns the class name for a given integer id.
    /// @throws std::out_of_range if the id is unknown.
    std::string class_name_for(int id) const { return id_to_class_.at(id); }

private:
    std::vector<std::string>              user_classes_;
    bool                                  skip_crowd_ = true;

    std::vector<AnnotatedImage<BGR>>      train_, val_, test_;
    std::unordered_map<std::string, int>  class_map_;
    std::unordered_map<int, std::string>  id_to_class_;

    std::expected<void, improc::Error>
    load_split(std::vector<AnnotatedImage<BGR>>& split,
               const std::filesystem::path& json_path,
               const std::filesystem::path& images_dir);
};

} // namespace improc::ml
