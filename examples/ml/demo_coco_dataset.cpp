// examples/ml/demo_coco_dataset.cpp
#include <iostream>
#include "improc/ml/ml.hpp"

int main(int argc, char* argv[]) {
    using namespace improc::ml;

    if (argc < 4) {
        std::cout << "Usage: demo_coco_dataset <train_json> <val_json> <images_dir>\n"
                  << "  Example: demo_coco_dataset annotations/train.json "
                     "annotations/val.json images/\n";
        return 0;
    }

    CocoDataset ds;
    ds.skip_crowd(true);

    auto result = ds.load_train(argv[1], argv[3]);
    if (!result) {
        std::cerr << "Train load failed: " << result.error().message << "\n";
        return 1;
    }

    result = ds.load_val(argv[2], argv[3]);
    if (!result) {
        std::cerr << "Val load failed: " << result.error().message << "\n";
        return 1;
    }

    std::cout << "Classes: " << ds.class_mapping().size() << "\n";
    for (const auto& [name, id] : ds.class_mapping())
        std::cout << "  [" << id << "] " << name << "\n";

    std::cout << "Train: " << ds.train().size()
              << "  Val: "  << ds.val().size() << "\n";

    if (!ds.train().empty()) {
        const auto& first = ds.train()[0];
        std::cout << "First train image: " << first.image.cols()
                  << "x" << first.image.rows()
                  << ", " << first.boxes.size() << " object(s)\n";
        for (const auto& b : first.boxes)
            std::cout << "  " << b.label << " @ [" << b.box << "]\n";
    }

    return 0;
}
