#include <improc/version.hpp>
#include <improc/core/pipeline.hpp>

int main() {
    static_assert(IMPROC_VERSION >= 10002, "improc >= 1.0.2 required");
    return 0;
}
