// tests/ml/test_dnn_forward.cpp
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include "improc/ml/dnn_forward.hpp"

using namespace improc::ml;
namespace fs = std::filesystem;

TEST(DnnForwardTest, NonExistentPathThrows) {
    EXPECT_THROW(DnnForward{"nonexistent/model.onnx"}, std::runtime_error);
}

TEST(DnnForwardTest, InvalidExtensionThrows) {
    fs::path p = fs::temp_directory_path() / "improc_fwd_dummy.txt";
    { std::ofstream f(p); f << "not a model"; }
    EXPECT_THROW(DnnForward{p.string()}, std::runtime_error);
    fs::remove(p);
}

TEST(DnnForwardTest, CorruptModelThrows) {
    fs::path p = fs::temp_directory_path() / "improc_fwd_dummy.onnx";
    { std::ofstream f(p); f << "not a real onnx file"; }
    EXPECT_THROW(DnnForward{p.string()}, std::runtime_error);
    fs::remove(p);
}

TEST(DnnForwardTest, ZeroScaleThrows) {
    EXPECT_THROW(DnnForward{"x.onnx"}.scale(0.0f), std::exception);
}

TEST(DnnForwardTest, ZeroInputWidthThrows) {
    EXPECT_THROW(DnnForward{"x.onnx"}.input_size(0, 224), std::exception);
}

TEST(DnnForwardTest, ZeroInputHeightThrows) {
    EXPECT_THROW(DnnForward{"x.onnx"}.input_size(224, 0), std::exception);
}
