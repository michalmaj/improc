from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, CMakeDeps, cmake_layout
from conan.tools.files import copy, get
from conan.tools.build import check_min_cppstd
import os


class ImprocConan(ConanFile):
    name = "improc"
    description = (
        "Type-safe C++23 image processing library built on OpenCV. "
        "Provides a composable pipeline API, geometric/filter/morphological ops, "
        "camera I/O, ML model loaders, camera calibration, and optional ONNX Runtime inference."
    )
    license = "MIT"
    url = "https://github.com/conan-io/conan-center-index"
    homepage = "https://github.com/michalmaj/improc"
    topics = ("image-processing", "opencv", "computer-vision", "cpp23", "pipeline")
    package_type = "library"
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared":    [True, False],
        "fPIC":      [True, False],
        "with_onnx": [True, False],
    }
    default_options = {
        "shared":    False,
        "fPIC":      True,
        "with_onnx": True,
    }

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")
        self.options["opencv"].with_protobuf = False
        try:
            self.options["opencv"].with_eigen = False
        except Exception:
            pass

    def layout(self):
        cmake_layout(self, src_folder=".")

    def validate(self):
        check_min_cppstd(self, "23")

    def requirements(self):
        self.requires("opencv/4.10.0")
        self.requires("nlohmann_json/3.11.3")
        if self.options.with_onnx:
            self.requires("onnxruntime/1.24.4")
            # Force Eigen >= 5.0.1 to satisfy onnxruntime; opencv defaults to 3.4.0
            self.requires("eigen/5.0.1", override=True)

    def source(self):
        get(self, **self.conan_data["sources"][self.version], strip_root=True)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        tc.variables["BUILD_SHARED_LIBS"]               = self.options.shared
        tc.variables["CMAKE_POSITION_INDEPENDENT_CODE"] = self.options.get_safe("fPIC", True)
        tc.variables["IMPROC_WITH_ONNX"]                = self.options.with_onnx
        tc.variables["IMPROC_BUILD_TESTS"]              = False
        tc.variables["IMPROC_BUILD_EXAMPLES"]           = False
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        copy(self, "LICENSE",
             src=self.source_folder,
             dst=os.path.join(self.package_folder, "licenses"))
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "improc")
        self.cpp_info.set_property("cmake_target_name", "improc::improc")
        self.cpp_info.libs = ["improc"]
        if self.options.with_onnx:
            self.cpp_info.defines = ["IMPROC_WITH_ONNX"]
