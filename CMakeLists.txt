cmake_minimum_required(VERSION 3.20)
project(improc__)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Required dependencies
find_package(Matplot++ CONFIG REQUIRED)
find_package(OpenCV REQUIRED)

# Force Armadillo to use raw LAPACK/BLAS (no wrapper functions)
add_compile_definitions(ARMA_DONT_USE_WRAPPER)

add_executable(improc__ main.cpp)

# Platform-specific LAPACK/BLAS linking
if(APPLE)
    # macOS: use Accelerate framework (includes BLAS/LAPACK)
    target_link_libraries(improc__ PRIVATE
            opencv::opencv
            Matplot++::cimg
            Matplot++::matplot
            "-framework Accelerate"
    )
elseif(WIN32)
    # Windows: use OpenBLAS (via vcpkg)
    find_package(OpenBLAS CONFIG REQUIRED)
    target_link_libraries(improc__ PRIVATE
            opencv::opencv
            Matplot++::cimg
            Matplot++::matplot
            OpenBLAS::OpenBLAS
    )
elseif(UNIX)
    # Linux: use OpenBLAS (via vcpkg or system)
    find_package(OpenBLAS CONFIG REQUIRED)
    target_link_libraries(improc__ PRIVATE
            opencv::opencv
            Matplot++::cimg
            Matplot++::matplot
            OpenBLAS::OpenBLAS
    )
endif()
