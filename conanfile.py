from conan import ConanFile
from conan.tools.cmake import cmake_layout, CMakeToolchain

# Developer/consumer recipe — installs dependencies needed to BUILD this repo locally.
# This is NOT the library packaging recipe.
# The CCI-ready producer recipe lives in recipes/improc/all/conanfile.py.
class ConanApplication(ConanFile):
    package_type = "application"
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps"

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.user_presets_path = False
        tc.generate()

    def requirements(self):
        requirements = self.conan_data.get('requirements', [])
        for requirement in requirements:
            self.requires(requirement)

    def configure(self):
        self.options["opencv"].with_protobuf = False
        # Disable opencv's bundled Eigen to resolve Conan graph conflict:
        # onnxruntime requires Eigen >= 5.0.1; OpenCV defaults to 3.4.0.
        # eigen/5.0.1 is listed in conandata.yml as an explicit dep to force
        # the correct version — improc++ itself does not use Eigen directly.
        try:
            self.options["opencv"].with_eigen = False
        except Exception:
            pass  # Option might not exist in all versions