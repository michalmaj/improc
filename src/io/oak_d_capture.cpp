// src/io/oak_d_capture.cpp
#include "improc/io/oak_d_capture.hpp"

#ifdef IMPROC_WITH_DEPTHAI

#include <opencv2/core.hpp>
#include <depthai/depthai.hpp>

namespace improc::io {

OakDCapture::OakDCapture() : source_id_("oak-d") {}
OakDCapture::OakDCapture(std::string mx_id)
    : mx_id_(mx_id), source_id_("oak-d:" + mx_id) {}
OakDCapture::~OakDCapture() { stop(); }

void OakDCapture::start() {
    if (started_.exchange(true)) return;
    try {
        dai::Pipeline pipeline;

        auto color_cam = pipeline.create<dai::node::ColorCamera>();
        color_cam->setBoardSocket(dai::CameraBoardSocket::CAM_A);
        color_cam->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
        color_cam->setColorOrder(dai::ColorCameraProperties::ColorOrder::BGR);
        color_cam->setInterleaved(true);

        auto left  = pipeline.create<dai::node::MonoCamera>();
        auto right = pipeline.create<dai::node::MonoCamera>();
        left->setBoardSocket(dai::CameraBoardSocket::CAM_B);
        right->setBoardSocket(dai::CameraBoardSocket::CAM_C);
        left->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
        right->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);

        auto stereo = pipeline.create<dai::node::StereoDepth>();
        stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_DENSITY);
        stereo->setDepthAlign(dai::CameraBoardSocket::CAM_A);  // align to RGB
        stereo->setOutputSize(1920, 1080);
        left->out.link(stereo->left);
        right->out.link(stereo->right);

        auto xout_rgb   = pipeline.create<dai::node::XLinkOut>();
        auto xout_depth = pipeline.create<dai::node::XLinkOut>();
        xout_rgb->setStreamName("rgb");
        xout_depth->setStreamName("depth");
        color_cam->video.link(xout_rgb->input);
        stereo->depth.link(xout_depth->input);

        dai::DeviceInfo device_info;
        if (!mx_id_.empty()) {
            device_info.mxid = mx_id_;
            device_ = std::make_shared<dai::Device>(pipeline, device_info);
        } else {
            device_ = std::make_shared<dai::Device>(pipeline);
        }

        rgb_queue_   = device_->getOutputQueue("rgb",   4, false);
        depth_queue_ = device_->getOutputQueue("depth", 4, false);
    } catch (...) {
        started_ = false;
        throw;
    }
}

void OakDCapture::stop() {
    if (!started_.exchange(false)) return;
    rgb_queue_.reset();
    depth_queue_.reset();
    device_.reset();
}

std::expected<CameraFrame, improc::Error> OakDCapture::getFrame() {
    if (!started_) {
        return std::unexpected(improc::Error::camera_unavailable("oak-d"));
    }

    auto rgb_data = rgb_queue_->get<dai::ImgFrame>(std::chrono::milliseconds(timeout_ms_));
    if (!rgb_data) return std::unexpected(improc::Error::timeout("oak-d:rgb"));

    auto depth_data = depth_queue_->get<dai::ImgFrame>(std::chrono::milliseconds(timeout_ms_));
    if (!depth_data) return std::unexpected(improc::Error::timeout("oak-d:depth"));

    // RGB: convert BGR ImgFrame to Image<BGR>
    cv::Mat bgr_mat = rgb_data->getCvFrame();
    core::Image<core::BGR> rgb_img(bgr_mat.clone());

    // Depth: uint16 mm → float32 metres
    cv::Mat depth_u16 = depth_data->getCvFrame();
    cv::Mat depth_f32;
    depth_u16.convertTo(depth_f32, CV_32F, 1.0 / 1000.0);
    core::Image<core::Float32> depth_img(depth_f32);

    return CameraFrame{
        .rgb       = std::move(rgb_img),
        .depth     = std::move(depth_img),
        .timestamp = std::chrono::steady_clock::now(),
        .source_id = source_id_,
    };
}

}  // namespace improc::io

#endif  // IMPROC_WITH_DEPTHAI
