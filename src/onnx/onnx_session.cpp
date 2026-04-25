// src/onnx/onnx_session.cpp
#include "improc/onnx/onnx_session.hpp"

#include <algorithm>
#include <onnxruntime_cxx_api.h>

namespace improc::onnx {

// ── Pimpl ────────────────────────────────────────────────────────────────────

struct OnnxSession::Impl {
    Ort::Env                      env{ORT_LOGGING_LEVEL_WARNING, "improc"};
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string>      input_names;
    std::vector<std::string>      output_names;
};

// ── Lifecycle ────────────────────────────────────────────────────────────────

OnnxSession::OnnxSession()  : impl_(std::make_unique<Impl>()) {}
OnnxSession::~OnnxSession() = default;

OnnxSession::OnnxSession(OnnxSession&&) noexcept            = default;
OnnxSession& OnnxSession::operator=(OnnxSession&&) noexcept = default;

// ── load ─────────────────────────────────────────────────────────────────────

std::expected<void, improc::Error>
OnnxSession::load(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path))
        return std::unexpected(improc::Error::invalid_model_file(path.string(), "file not found"));

    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext != ".onnx")
        return std::unexpected(improc::Error::invalid_model_file(path.string(),
            "unsupported extension '" + ext + "' — only .onnx is accepted"));

    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef __APPLE__
        // CoreML EP: accelerates compatible ops on Apple Silicon; falls back to CPU otherwise.
        try { opts.AppendExecutionProvider("CoreML"); } catch (...) {}
#endif

        auto path_str    = path.string();
        impl_->session   = std::make_unique<Ort::Session>(impl_->env, path_str.c_str(), opts);

        Ort::AllocatorWithDefaultOptions alloc;
        impl_->input_names.clear();
        for (size_t i = 0; i < impl_->session->GetInputCount(); ++i)
            impl_->input_names.push_back(impl_->session->GetInputNameAllocated(i, alloc).get());

        impl_->output_names.clear();
        for (size_t i = 0; i < impl_->session->GetOutputCount(); ++i)
            impl_->output_names.push_back(impl_->session->GetOutputNameAllocated(i, alloc).get());

        return {};
    } catch (const Ort::Exception& e) {
        return std::unexpected(improc::Error::onnx_model_load_failed(path.string(), e.what()));
    }
}

// ── run ──────────────────────────────────────────────────────────────────────

std::expected<std::vector<TensorInfo>, improc::Error>
OnnxSession::run(const std::vector<TensorInfo>& inputs) const {
    if (!impl_->session)
        return std::unexpected(improc::Error::onnx_session_not_loaded());

    try {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<Ort::Value>    input_tensors;
        std::vector<const char*>   input_names_c;
        input_tensors.reserve(inputs.size());
        input_names_c.reserve(inputs.size());

        for (const auto& t : inputs) {
            input_names_c.push_back(t.name.c_str());
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info,
                const_cast<float*>(t.data.data()),
                t.data.size(),
                t.shape.data(),
                t.shape.size()
            ));
        }

        std::vector<const char*> output_names_c;
        output_names_c.reserve(impl_->output_names.size());
        for (const auto& n : impl_->output_names)
            output_names_c.push_back(n.c_str());

        auto ort_outputs = impl_->session->Run(
            Ort::RunOptions{nullptr},
            input_names_c.data(),   input_tensors.data(), input_tensors.size(),
            output_names_c.data(),  output_names_c.size()
        );

        std::vector<TensorInfo> results;
        results.reserve(ort_outputs.size());
        for (size_t i = 0; i < ort_outputs.size(); ++i) {
            auto& v     = ort_outputs[i];
            auto  info  = v.GetTensorTypeAndShapeInfo();
            auto  shape = info.GetShape();
            auto  count = info.GetElementCount();
            const float* ptr = v.GetTensorData<float>();
            results.push_back({impl_->output_names[i], shape, {ptr, ptr + count}});
        }

        return results;

    } catch (const Ort::Exception& e) {
        return std::unexpected(improc::Error::onnx_inference_failed(e.what()));
    }
}

// ── introspection ────────────────────────────────────────────────────────────

std::vector<std::string> OnnxSession::input_names() const {
    return impl_->input_names;
}

std::vector<std::string> OnnxSession::output_names() const {
    return impl_->output_names;
}

bool OnnxSession::is_loaded() const noexcept {
    return impl_->session != nullptr;
}

} // namespace improc::onnx
