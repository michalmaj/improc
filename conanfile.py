from conan import ConanFile
from conan.tools.cmake import cmake_layout, CMakeToolchain

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
        # Disable opencv's eigen to resolve conflict with onnxruntime which needs eigen >= 5.0.1
        # We use eigen/5.0.1 as an explicit requirement instead
        try:
            self.options["opencv"].with_eigen = False
        except:
            pass  # Option might not exist in all versions