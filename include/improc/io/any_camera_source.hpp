// include/improc/io/any_camera_source.hpp
#pragma once

#include <memory>
#include "improc/io/camera_source.hpp"

namespace improc::io {

namespace detail {

struct ICameraSourceErased {
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual std::expected<CameraFrame, improc::Error> getFrame() = 0;
    virtual ~ICameraSourceErased() = default;
};

template<CameraSourceType T>
struct CameraSourceImpl final : ICameraSourceErased {
    template<typename... Args>
    explicit CameraSourceImpl(Args&&... args)
        : source_(std::forward<Args>(args)...) {}

    void start() override { source_.start(); }
    void stop()  override { source_.stop(); }
    std::expected<CameraFrame, improc::Error> getFrame() override {
        return source_.getFrame();
    }

    T source_;
};

} // namespace detail

class AnyCameraSource {
public:
    AnyCameraSource() = default;

    template<CameraSourceType T, typename... Args>
    static AnyCameraSource make(Args&&... args) {
        AnyCameraSource any;
        any.impl_ = std::make_shared<detail::CameraSourceImpl<T>>(
            std::forward<Args>(args)...);
        return any;
    }

    void start() { impl_->start(); }
    void stop()  { impl_->stop(); }
    std::expected<CameraFrame, improc::Error> getFrame() { return impl_->getFrame(); }

    explicit operator bool() const noexcept { return impl_ != nullptr; }

private:
    std::shared_ptr<detail::ICameraSourceErased> impl_;
};

} // namespace improc::io
