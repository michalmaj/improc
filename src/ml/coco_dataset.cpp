// src/ml/coco_dataset.cpp
#include "improc/ml/coco_dataset.hpp"

#include <filesystem>
#include <format>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

namespace improc::ml {

using json = nlohmann::json;

std::expected<std::vector<AnnotatedImage<BGR>>, improc::Error>
parse_coco_json(const std::filesystem::path& json_path,
                const std::filesystem::path& images_dir,
                std::unordered_map<std::string, int>& class_map,
                bool skip_crowd,
                bool filter_unknown)
{
    if (!std::filesystem::exists(json_path))
        return std::unexpected(improc::Error::coco_json_parse_failed(
            json_path.string(), "file not found"));

    std::ifstream f(json_path);
    if (!f.is_open())
        return std::unexpected(improc::Error::coco_json_parse_failed(
            json_path.string(), "cannot open file"));

    json doc;
    try {
        doc = json::parse(f);
    } catch (const json::exception& e) {
        return std::unexpected(improc::Error::coco_json_parse_failed(
            json_path.string(), std::format("JSON parse error: {}", e.what())));
    }

    for (const char* key : {"images", "annotations", "categories"}) {
        if (!doc.contains(key))
            return std::unexpected(improc::Error::coco_json_parse_failed(
                json_path.string(), std::format("missing '{}' key", key)));
    }

    // categories: coco_id → name
    std::unordered_map<int, std::string> coco_id_to_name;
    std::unordered_map<int, std::string> img_id_to_filename;
    std::unordered_map<int, std::vector<json>> img_id_to_anns;

    try {
        for (const auto& cat : doc["categories"])
            coco_id_to_name[cat["id"].get<int>()] = cat["name"].get<std::string>();

        // images: coco_img_id → filename
        for (const auto& img : doc["images"])
            img_id_to_filename[img["id"].get<int>()] = img["file_name"].get<std::string>();

        // annotations grouped by image_id
        for (const auto& ann : doc["annotations"])
            img_id_to_anns[ann["image_id"].get<int>()].push_back(ann);
    } catch (const json::exception& e) {
        return std::unexpected(improc::Error::coco_json_parse_failed(
            json_path.string(), std::format("malformed JSON: {}", e.what())));
    }

    // Sort by image id for deterministic result ordering
    std::vector<int> sorted_ids;
    sorted_ids.reserve(img_id_to_filename.size());
    for (const auto& [id, _] : img_id_to_filename)
        sorted_ids.push_back(id);
    std::ranges::sort(sorted_ids);

    std::vector<AnnotatedImage<BGR>> result;
    result.reserve(sorted_ids.size());

    for (int img_id : sorted_ids) {
        const auto& filename = img_id_to_filename.at(img_id);
        auto img_path = images_dir / filename;
        if (!std::filesystem::exists(img_path)) {
            auto stem = std::filesystem::path(filename).stem();
            img_path  = images_dir / (stem.string() + ".png");
        }
        cv::Mat mat = cv::imread(img_path.string(), cv::IMREAD_COLOR);
        if (mat.empty())
            return std::unexpected(improc::Error::image_read_failed(img_path.string()));

        std::vector<BBox> boxes;
        try {
            for (const auto& ann : img_id_to_anns[img_id]) {
                if (skip_crowd && ann.value("iscrowd", 0) == 1) continue;

                auto cat_it = coco_id_to_name.find(ann["category_id"].get<int>());
                if (cat_it == coco_id_to_name.end()) continue;
                const std::string& name = cat_it->second;

                if (filter_unknown && !class_map.contains(name)) continue;

                auto [it, inserted] = class_map.try_emplace(name, static_cast<int>(class_map.size()));
                int class_id = it->second;

                const auto& bbox = ann["bbox"];
                if (bbox.size() < 4)
                    return std::unexpected(improc::Error::coco_json_parse_failed(
                        json_path.string(), "annotation bbox has fewer than 4 elements"));
                boxes.push_back(BBox{
                    cv::Rect2f{bbox[0].get<float>(), bbox[1].get<float>(),
                               bbox[2].get<float>(), bbox[3].get<float>()},
                    class_id,
                    name
                });
            }
        } catch (const json::exception& e) {
            return std::unexpected(improc::Error::coco_json_parse_failed(
                json_path.string(), std::format("malformed JSON: {}", e.what())));
        }

        result.push_back(AnnotatedImage<BGR>{Image<BGR>(mat), std::move(boxes)});
    }

    return result;
}

CocoDataset& CocoDataset::classes(std::vector<std::string> cls) {
    user_classes_ = std::move(cls);
    return *this;
}

std::expected<void, improc::Error>
CocoDataset::load_split(std::vector<AnnotatedImage<BGR>>& split,
                        const std::filesystem::path& json_path,
                        const std::filesystem::path& images_dir)
{
    // Pre-fill class map from user list on first load call
    if (!user_classes_.empty() && class_map_.empty()) {
        for (int i = 0; i < static_cast<int>(user_classes_.size()); ++i) {
            class_map_[user_classes_[i]] = i;
            id_to_class_[i] = user_classes_[i];
        }
    }

    bool filter = !user_classes_.empty();
    auto result = parse_coco_json(json_path, images_dir, class_map_, skip_crowd_, filter);
    if (!result) return std::unexpected(result.error());

    split = std::move(*result);

    id_to_class_.clear();
    for (const auto& [name, id] : class_map_)
        id_to_class_[id] = name;

    return {};
}

std::expected<void, improc::Error>
CocoDataset::load_train(const std::filesystem::path& json_path,
                        const std::filesystem::path& images_dir) {
    return load_split(train_, json_path, images_dir);
}

std::expected<void, improc::Error>
CocoDataset::load_val(const std::filesystem::path& json_path,
                      const std::filesystem::path& images_dir) {
    return load_split(val_, json_path, images_dir);
}

std::expected<void, improc::Error>
CocoDataset::load_test(const std::filesystem::path& json_path,
                       const std::filesystem::path& images_dir) {
    return load_split(test_, json_path, images_dir);
}

} // namespace improc::ml
