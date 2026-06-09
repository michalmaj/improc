from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.cmake import CMake, CMakeToolchain, CMakeDeps, cmake_layout
from conan.tools.files import copy, get
from conan.tools.build import check_min_cppstd
from conan.tools.scm import Version
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
        if self.options.with_onnx:
            # onnxruntime bundles its own protobuf; enabling OpenCV's protobuf causes
            # duplicate symbol errors at link time.
            self.options["opencv"].with_protobuf = False
            # onnxruntime requires Eigen >= 5.x; OpenCV defaults to bundled 3.4.x,
            # which produces a Conan graph conflict. Disabling opencv's bundled Eigen
            # lets the explicit eigen/5.x requirement in requirements() win.
            try:
                self.options["opencv"].with_eigen = False
            except Exception:
                pass

    def layout(self):
        cmake_layout(self, src_folder=".")

    def validate(self):
        check_min_cppstd(self, "23")

        compiler = str(self.settings.compiler)
        version = Version(str(self.settings.compiler.version))

        if compiler == "gcc" and version < "14":
            raise ConanInvalidConfiguration("improc requires GCC >= 14 for full C++23 support")
        if compiler == "clang" and version < "18":
            raise ConanInvalidConfiguration("improc requires Clang >= 18 for full C++23 support")
        if compiler == "apple-clang" and version < "16":
            raise ConanInvalidConfiguration(
                "improc requires Apple-Clang >= 16 (Xcode 16); "
                "older Apple-Clang versions are not validated for this C++23 codebase"
            )
        if compiler == "msvc":
            raise ConanInvalidConfiguration(
                "improc C++23 support with MSVC is not validated in this recipe"
            )

    def build_requirements(self):
        self.tool_requires("cmake/[>=3.30 <4]")

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
        tc.variables["IMPROC_WITH_DEPTHAI"]             = False
        tc.variables["IMPROC_BENCHMARKS"]               = False
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
