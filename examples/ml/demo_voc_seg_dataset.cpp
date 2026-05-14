// examples/ml/demo_voc_seg_dataset.cpp
#include <iostream>
#include "improc/ml/ml.hpp"

int main(int argc, char* argv[]) {
    using namespace improc::ml;

    if (argc < 2) {
        std::cout << "Usage: demo_voc_seg_dataset <path/to/voc_root>\n"
                  << "  Expected structure: JPEGImages/, SegmentationClass/,\n"
                  << "  optionally SegmentationObject/, ImageSets/Segmentation/\n";
        return 0;
    }

    VocSegDataset ds;
    ds.classes({"background","aeroplane","bicycle","bird","boat",
                "bottle","bus","car","cat","chair","cow",
                "diningtable","dog","horse","motorbike","person",
                "pottedplant","sheep","sofa","train","tvmonitor"})
      .load_instance_masks(false);

    auto result = ds.load_from_directory(argv[1]);
    if (!result) {
        std::cerr << "Failed: " << result.error().message << "\n";
        return 1;
    }

    std::cout << "Train: " << ds.train().size()
              << "  Val: "  << ds.val().size()
              << "  Test: " << ds.test().size() << "\n";

    if (!ds.train().empty()) {
        const auto& first = ds.train()[0];
        std::cout << "First train image: "
                  << first.image.cols() << "x" << first.image.rows()
                  << "  class_mask: "
                  << first.class_mask.cols() << "x" << first.class_mask.rows()
                  << "  instance_mask: "
                  << (first.instance_mask ? "present" : "absent") << "\n";
    }

    return 0;
}
