// examples/ml/demo_voc_dataset.cpp
#include <iostream>
#include "improc/ml/ml.hpp"

int main(int argc, char* argv[]) {
    using namespace improc::ml;

    if (argc < 2) {
        std::cout << "Usage: demo_voc_dataset <path/to/voc_root>\n"
                  << "  Expected structure: JPEGImages/, Annotations/, "
                     "optionally ImageSets/Main/\n";
        return 0;
    }

    VocDataset ds;
    ds.skip_difficult(true);

    auto result = ds.load_from_directory(argv[1]);
    if (!result) {
        std::cerr << "Failed: " << result.error().message << "\n";
        return 1;
    }

    std::cout << "Classes: " << ds.class_mapping().size() << "\n";
    for (const auto& [name, id] : ds.class_mapping())
        std::cout << "  [" << id << "] " << name << "\n";

    std::cout << "Train: " << ds.train().size()
              << "  Val: "  << ds.val().size()
              << "  Test: " << ds.test().size() << "\n";

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
