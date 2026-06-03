// examples/io/demo_webcam_capture.cpp
#include "improc/io/webcam_capture.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

int main() {
    improc::io::WebcamCapture camera(0);
    camera.start();
    std::cout << "Starting camera. Press ESC to exit.\n";
    while (true) {
        auto frame = camera.getFrame();
        if (frame.has_value() && frame->rgb.has_value())
            cv::imshow(camera.getWindowName(), frame->rgb->mat());
        if (cv::waitKey(30) == 27) break;
    }
    camera.stop();
    return 0;
}
