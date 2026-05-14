// src/ml/voc_dataset.cpp
#include "improc/ml/voc_dataset.hpp"

#include <algorithm>
#include <format>
#include <fstream>
#include <random>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <opencv2/imgcodecs.hpp>

namespace improc::ml {

// ── Minimal XML helpers ──────────────────────────────────────────────────────

namespace {

// Returns the trimmed text content between the FIRST occurrence of <tag>…</tag>.
std::string extract_tag(const std::string& xml, const std::string& tag) {
    const std::string open  = "<"  + tag + ">";
    const std::string close = "</" + tag + ">";
    auto s = xml.find(open);
    if (s == std::string::npos) return {};
    s += open.size();
    auto e = xml.find(close, s);
    if (e == std::string::npos) return {};
    auto v = xml.substr(s, e - s);
    auto f = v.find_first_not_of(" \t\n\r");
    if (f == std::string::npos) return {};
    auto l = v.find_last_not_of(" \t\n\r");
    return v.substr(f, l - f + 1);
}

// Returns the text content between every occurrence of <tag>…</tag>.
std::vector<std::string> extract_all(const std::string& xml, const std::string& tag) {
    std::vector<std::string> out;
    const std::string open  = "<"  + tag + ">";
    const std::string close = "</" + tag + ">";
    std::size_t pos = 0;
    while (true) {
        auto s = xml.find(open, pos);
        if (s == std::string::npos) break;
        s += open.size();
        auto e = xml.find(close, s);
        if (e == std::string::npos) break;
        out.push_back(xml.substr(s, e - s));
        pos = e + close.size();
    }
    return out;
}

} // anonymous namespace

// ── parse_voc_xml ────────────────────────────────────────────────────────────

std::expected<AnnotatedImage<BGR>, improc::Error>
parse_voc_xml(const std::filesystem::path& xml_path,
              const std::filesystem::path& images_dir,
              std::unordered_map<std::string, int>& class_map,
              bool skip_difficult,
              bool filter_unknown)
{
    if (!std::filesystem::exists(xml_path))
        return std::unexpected(improc::Error::voc_xml_parse_failed(
            xml_path.string(), "file not found"));

    std::ifstream f(xml_path);
    if (!f.is_open())
        return std::unexpected(improc::Error::voc_xml_parse_failed(
            xml_path.string(), "cannot open file"));
    std::string xml((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());

    const std::string filename = extract_tag(xml, "filename");
    if (filename.empty())
        return std::unexpected(improc::Error::voc_xml_parse_failed(
            xml_path.string(), "missing <filename> tag"));

    // Try exact filename, then swap extension to .png
    auto img_path = images_dir / filename;
    if (!std::filesystem::exists(img_path)) {
        auto stem = std::filesystem::path(filename).stem();
        img_path  = images_dir / (stem.string() + ".png");
    }

    cv::Mat mat = cv::imread(img_path.string(), cv::IMREAD_COLOR);
    if (mat.empty())
        return std::unexpected(improc::Error::image_read_failed(img_path.string()));

    std::vector<BBox> boxes;
    for (const auto& obj : extract_all(xml, "object")) {
        if (skip_difficult && extract_tag(obj, "difficult") == "1") continue;

        const std::string name = extract_tag(obj, "name");
        if (name.empty())
            return std::unexpected(improc::Error::voc_xml_parse_failed(
                xml_path.string(), "object missing <name>"));

        if (filter_unknown && !class_map.contains(name)) continue;

        if (!class_map.contains(name)) {
            int new_id = static_cast<int>(class_map.size());
            class_map[name] = new_id;
        }

        const std::string bndbox = extract_tag(obj, "bndbox");
        if (bndbox.empty())
            return std::unexpected(improc::Error::voc_xml_parse_failed(
                xml_path.string(), "object missing <bndbox>"));

        try {
            float xmin = std::stof(extract_tag(bndbox, "xmin"));
            float ymin = std::stof(extract_tag(bndbox, "ymin"));
            float xmax = std::stof(extract_tag(bndbox, "xmax"));
            float ymax = std::stof(extract_tag(bndbox, "ymax"));
            boxes.push_back(BBox{
                cv::Rect2f{xmin, ymin, xmax - xmin, ymax - ymin},
                class_map[name],
                name
            });
        } catch (const std::exception& e) {
            return std::unexpected(improc::Error::voc_xml_parse_failed(
                xml_path.string(),
                std::format("invalid bndbox coordinates: {}", e.what())));
        }
    }

    return AnnotatedImage<BGR>{Image<BGR>(mat), std::move(boxes)};
}

// ── VocDataset helpers ───────────────────────────────────────────────────────

VocDataset& VocDataset::classes(std::vector<std::string> cls) {
    user_classes_ = std::move(cls);
    return *this;
}

VocDataset& VocDataset::test_ratio(float r) {
    if (r < 0.0f || r >= 1.0f)
        throw ParameterError{"test_ratio", "must be in [0, 1)", "VocDataset"};
    test_ratio_ = r; return *this;
}

VocDataset& VocDataset::val_ratio(float r) {
    if (r < 0.0f || r >= 1.0f)
        throw ParameterError{"val_ratio", "must be in [0, 1)", "VocDataset"};
    val_ratio_ = r; return *this;
}

std::expected<std::vector<AnnotatedImage<BGR>>, improc::Error>
VocDataset::load_stems(const std::vector<std::string>& stems,
                       const std::filesystem::path& images_dir,
                       const std::filesystem::path& annotations_dir)
{
    bool filter = !user_classes_.empty();
    std::vector<AnnotatedImage<BGR>> result;
    for (const auto& stem : stems) {
        auto xml_path = annotations_dir / (stem + ".xml");
        auto ann = parse_voc_xml(xml_path, images_dir, class_map_,
                                 skip_difficult_, filter);
        if (!ann) return std::unexpected(ann.error());
        result.push_back(std::move(*ann));
    }
    return result;
}

// ── VocDataset::load_from_directory ─────────────────────────────────────────

std::expected<void, improc::Error>
VocDataset::load_from_directory(const std::filesystem::path& root) {
    if (!std::filesystem::exists(root) || !std::filesystem::is_directory(root))
        return std::unexpected(improc::Error::directory_not_found(root.string()));

    const auto annotations_dir = root / "Annotations";
    if (!std::filesystem::exists(annotations_dir))
        return std::unexpected(improc::Error::directory_not_found(
            annotations_dir.string()));

    const auto images_dir = root / "JPEGImages";

    // Reset state
    class_map_.clear();
    id_to_class_.clear();
    train_.clear(); val_.clear(); test_.clear();

    // Pre-fill class map from user list if provided
    if (!user_classes_.empty()) {
        for (int i = 0; i < static_cast<int>(user_classes_.size()); ++i) {
            class_map_[user_classes_[i]] = i;
            id_to_class_[i] = user_classes_[i];
        }
    }

    const auto imageset_train = root / "ImageSets" / "Main" / "train.txt";
    const bool use_voc_split  = std::filesystem::exists(imageset_train);

    auto read_stems = [](const std::filesystem::path& p) {
        std::vector<std::string> stems;
        if (!std::filesystem::exists(p)) return stems;
        std::ifstream f(p);
        std::string line;
        while (std::getline(f, line)) {
            auto first = line.find_first_not_of(" \t\r\n");
            auto last  = line.find_last_not_of(" \t\r\n");
            if (first != std::string::npos)
                stems.push_back(line.substr(first, last - first + 1));
        }
        return stems;
    };

    if (use_voc_split) {
        auto tr = load_stems(read_stems(root / "ImageSets" / "Main" / "train.txt"),
                             images_dir, annotations_dir);
        if (!tr) return std::unexpected(tr.error());
        train_ = std::move(*tr);

        auto va = load_stems(read_stems(root / "ImageSets" / "Main" / "val.txt"),
                             images_dir, annotations_dir);
        if (!va) return std::unexpected(va.error());
        val_ = std::move(*va);

        auto te = load_stems(read_stems(root / "ImageSets" / "Main" / "test.txt"),
                             images_dir, annotations_dir);
        if (!te) return std::unexpected(te.error());
        test_ = std::move(*te);
    } else {
        // Collect all XML stems, sorted for determinism
        std::vector<std::string> stems;
        for (const auto& entry : std::filesystem::directory_iterator(annotations_dir))
            if (entry.path().extension() == ".xml")
                stems.push_back(entry.path().stem().string());
        std::ranges::sort(stems);

        auto all = load_stems(stems, images_dir, annotations_dir);
        if (!all) return std::unexpected(all.error());

        // Shuffle
        if (shuffle_seed_) {
            std::mt19937 rng(*shuffle_seed_);
            std::ranges::shuffle(*all, rng);
        } else {
            std::mt19937 rng(std::random_device{}());
            std::ranges::shuffle(*all, rng);
        }

        const std::size_t total      = all->size();
        const std::size_t test_count = static_cast<std::size_t>(total * test_ratio_);
        const std::size_t val_count  = static_cast<std::size_t>(total * val_ratio_);
        const std::size_t train_count = total - test_count - val_count;

        train_.assign(all->begin(),                            all->begin() + train_count);
        val_.assign  (all->begin() + train_count,             all->begin() + train_count + val_count);
        test_.assign (all->begin() + train_count + val_count, all->end());
    }

    // Build id_to_class_ from final class_map_
    for (const auto& [name, id] : class_map_)
        id_to_class_[id] = name;

    return {};
}

} // namespace improc::ml
