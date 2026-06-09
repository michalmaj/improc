#include <improc/version.hpp>
#include <improc/core/pipeline.hpp>

int main() {
    static_assert(IMPROC_VERSION >= 10001, "improc >= 1.0.1 required");
    return 0;
}
