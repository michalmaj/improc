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

/**
 * @brief Type-erased wrapper for any `CameraSourceType` — holds a webcam, IP cam, OAK-D, or other source.
 *
 * The wrapped source is heap-allocated via `make<T>(args...)`. Copying is disabled; move is allowed.
 *
 * @code
 * auto src = AnyCameraSource::make<WebcamCapture>(0);
 * src.start();
 * while (auto frame = src.getFrame()) { ... }
 * src.stop();
 * @endcode
 */
class AnyCameraSource {
public:
    /// @brief Constructs an empty (no-op) source. getFrame() returns CameraUnavailable.
    AnyCameraSource() = default;

    /// @brief Constructs an AnyCameraSource wrapping a new instance of T built from args.
    template<CameraSourceType T, typename... Args>
    static AnyCameraSource make(Args&&... args) {
        AnyCameraSource any;
        any.impl_ = std::make_shared<detail::CameraSourceImpl<T>>(
            std::forward<Args>(args)...);
        return any;
    }

    /// @brief Starts the wrapped source. No-op if empty.
    void start() { if (impl_) impl_->start(); }
    /// @brief Stops the wrapped source. No-op if empty.
    void stop()  { if (impl_) impl_->stop(); }
    /// @brief Returns the next frame from the wrapped source.
    /// @return CameraUnavailable error if the source is empty; otherwise delegates to the wrapped source.
    std::expected<CameraFrame, improc::Error> getFrame() {
        if (!impl_)
            return std::unexpected(improc::Error{
                improc::Error::Code::CameraUnavailable,
                "AnyCameraSource is empty (no camera source wrapped)"});
        return impl_->getFrame();
    }

    /// @brief Returns true if a camera source has been wrapped.
    explicit operator bool() const noexcept { return impl_ != nullptr; }

private:
    std::shared_ptr<detail::ICameraSourceErased> impl_;
};

} // namespace improc::io
