# Conan automatically generated toolchain file
# DO NOT EDIT MANUALLY, it will be overwritten

# Avoid including toolchain file several times (bad if appending to variables like
#   CMAKE_CXX_FLAGS. See https://github.com/android/ndk/issues/323
include_guard()
message(STATUS "Using Conan toolchain: ${CMAKE_CURRENT_LIST_FILE}")
if(${CMAKE_VERSION} VERSION_LESS "3.15")
    message(FATAL_ERROR "The 'CMakeToolchain' generator only works with CMake >= 3.15")
endif()

########## 'user_toolchain' block #############
# Include one or more CMake user toolchain from tools.cmake.cmaketoolchain:user_toolchain



########## 'generic_system' block #############
# Definition of system, platform and toolset





########## 'compilers' block #############

set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")


########## 'apple_system' block #############
# Define Apple architectures, sysroot, deployment target, bitcode, etc

# Set the architectures for which to build.
set(CMAKE_OSX_ARCHITECTURES arm64 CACHE STRING "" FORCE)
# Setting CMAKE_OSX_SYSROOT SDK, when using Xcode generator the name is enough
# but full path is necessary for others
set(CMAKE_OSX_SYSROOT macosx CACHE STRING "" FORCE)
set(BITCODE "")
set(FOBJC_ARC "")
set(VISIBILITY "")
#Check if Xcode generator is used, since that will handle these flags automagically
if(CMAKE_GENERATOR MATCHES "Xcode")
  message(DEBUG "Not setting any manual command-line buildflags, since Xcode is selected as generator.")
else()
    string(APPEND CONAN_C_FLAGS " ${BITCODE} ${VISIBILITY} ${FOBJC_ARC}")
    string(APPEND CONAN_CXX_FLAGS " ${BITCODE} ${VISIBILITY} ${FOBJC_ARC}")
endif()


########## 'libcxx' block #############
# Definition of libcxx from 'compiler.libcxx' setting, defining the
# right CXX_FLAGS for that libcxx

message(STATUS "Conan toolchain: Defining libcxx as C++ flags: -stdlib=libc++")
string(APPEND CONAN_CXX_FLAGS " -stdlib=libc++")


########## 'cppstd' block #############
# Define the C++ and C standards from 'compiler.cppstd' and 'compiler.cstd'

function(conan_modify_std_watch variable access value current_list_file stack)
    set(conan_watched_std_variable "23")
    if (${variable} STREQUAL "CMAKE_C_STANDARD")
        set(conan_watched_std_variable "")
    endif()
    if ("${access}" STREQUAL "MODIFIED_ACCESS" AND NOT "${value}" STREQUAL "${conan_watched_std_variable}")
        message(STATUS "Warning: Standard ${variable} value defined in conan_toolchain.cmake to ${conan_watched_std_variable} has been modified to ${value} by ${current_list_file}")
    endif()
    unset(conan_watched_std_variable)
endfunction()

message(STATUS "Conan toolchain: C++ Standard 23 with extensions OFF")
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
variable_watch(CMAKE_CXX_STANDARD conan_modify_std_watch)


########## 'extra_flags' block #############
# Include extra C++, C and linker flags from configuration tools.build:<type>flags
# and from CMakeToolchain.extra_<type>_flags

# Conan conf flags start: 
# Conan conf flags end


########## 'cmake_flags_init' block #############
# Define CMAKE_<XXX>_FLAGS from CONAN_<XXX>_FLAGS

foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER ${config} config)
    if(DEFINED CONAN_CXX_FLAGS_${config})
      string(APPEND CMAKE_CXX_FLAGS_${config}_INIT " ${CONAN_CXX_FLAGS_${config}}")
    endif()
    if(DEFINED CONAN_C_FLAGS_${config})
      string(APPEND CMAKE_C_FLAGS_${config}_INIT " ${CONAN_C_FLAGS_${config}}")
    endif()
    if(DEFINED CONAN_SHARED_LINKER_FLAGS_${config})
      string(APPEND CMAKE_SHARED_LINKER_FLAGS_${config}_INIT " ${CONAN_SHARED_LINKER_FLAGS_${config}}")
    endif()
    if(DEFINED CONAN_EXE_LINKER_FLAGS_${config})
      string(APPEND CMAKE_EXE_LINKER_FLAGS_${config}_INIT " ${CONAN_EXE_LINKER_FLAGS_${config}}")
    endif()
endforeach()

if(DEFINED CONAN_CXX_FLAGS)
  string(APPEND CMAKE_CXX_FLAGS_INIT " ${CONAN_CXX_FLAGS}")
endif()
if(DEFINED CONAN_C_FLAGS)
  string(APPEND CMAKE_C_FLAGS_INIT " ${CONAN_C_FLAGS}")
endif()
if(DEFINED CONAN_SHARED_LINKER_FLAGS)
  string(APPEND CMAKE_SHARED_LINKER_FLAGS_INIT " ${CONAN_SHARED_LINKER_FLAGS}")
endif()
if(DEFINED CONAN_EXE_LINKER_FLAGS)
  string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT " ${CONAN_EXE_LINKER_FLAGS}")
endif()


########## 'extra_variables' block #############
# Definition of extra CMake variables from tools.cmake.cmaketoolchain:extra_variables



########## 'try_compile' block #############
# Blocks after this one will not be added when running CMake try/checks

get_property( _CMAKE_IN_TRY_COMPILE GLOBAL PROPERTY IN_TRY_COMPILE )
if(_CMAKE_IN_TRY_COMPILE)
    message(STATUS "Running toolchain IN_TRY_COMPILE")
    return()
endif()


########## 'find_paths' block #############
# Define paths to find packages, programs, libraries, etc.
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/conan_cmakedeps_paths.cmake")
  message(STATUS "Conan toolchain: Including CMakeDeps generated conan_find_paths.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/conan_cmakedeps_paths.cmake")
else()

set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)

# Definition of CMAKE_MODULE_PATH
list(PREPEND CMAKE_MODULE_PATH "/Users/michalmaj/.conan2/p/b/proto54fdac56de2ea/p/lib/cmake/protobuf" "/Users/michalmaj/.conan2/p/b/openja99b17d74a5ef/p/lib/cmake" "/Users/michalmaj/.conan2/p/b/opens9bf56e2ebc65d/p/lib/cmake")
# the generators folder (where conan generates files, like this toolchain)
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})

# Definition of CMAKE_PREFIX_PATH, CMAKE_XXXXX_PATH
# The explicitly defined "builddirs" of "host" context dependencies must be in PREFIX_PATH
list(PREPEND CMAKE_PREFIX_PATH "/Users/michalmaj/.conan2/p/b/proto54fdac56de2ea/p/lib/cmake/protobuf" "/Users/michalmaj/.conan2/p/b/openja99b17d74a5ef/p/lib/cmake" "/Users/michalmaj/.conan2/p/b/opens9bf56e2ebc65d/p/lib/cmake")
# The Conan local "generators" folder, where this toolchain is saved.
list(PREPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_LIST_DIR} )
list(PREPEND CMAKE_LIBRARY_PATH "/Users/michalmaj/.conan2/p/b/openccca6a0a7c2059/p/lib" "lib" "/Users/michalmaj/.conan2/p/b/proto54fdac56de2ea/p/lib" "/Users/michalmaj/.conan2/p/b/ade08e8b03f86db1/p/lib" "/Users/michalmaj/.conan2/p/b/opene8abc43ba2bf05/p/lib" "/Users/michalmaj/.conan2/p/b/imath09f5b1ba5d337/p/lib" "/Users/michalmaj/.conan2/p/b/libti9d6c2c398a00f/p/lib" "/Users/michalmaj/.conan2/p/b/libde0eff2c1b38ac4/p/lib" "/Users/michalmaj/.conan2/p/b/libjp3013fbffe5d14/p/lib" "/Users/michalmaj/.conan2/p/b/jbig9b08dbdbc0f2e/p/lib" "/Users/michalmaj/.conan2/p/b/zstd39aeefd0022b0/p/lib" "/Users/michalmaj/.conan2/p/b/quirc877de30ad1087/p/lib" "/Users/michalmaj/.conan2/p/b/ffmpe147b26bf1ae86/p/lib" "/Users/michalmaj/.conan2/p/b/xz_utcba4be3eb8f04/p/lib" "/Users/michalmaj/.conan2/p/b/libicee588c702423d/p/lib" "/Users/michalmaj/.conan2/p/b/freet0decf104691c0/p/lib" "/Users/michalmaj/.conan2/p/b/libpn8f5988011ee78/p/lib" "/Users/michalmaj/.conan2/p/b/bzip2aec6cc871a76d/p/lib" "/Users/michalmaj/.conan2/p/b/brotl1017876a3b95f/p/lib" "/Users/michalmaj/.conan2/p/b/openja99b17d74a5ef/p/lib" "/Users/michalmaj/.conan2/p/b/openhd9e0c2f6b0367/p/lib" "/Users/michalmaj/.conan2/p/b/vorbia97ce2e270bac/p/lib" "/Users/michalmaj/.conan2/p/b/ogga9eb13d871b96/p/lib" "/Users/michalmaj/.conan2/p/b/opus9e379e32bcda2/p/lib" "/Users/michalmaj/.conan2/p/b/libx27eaae947448b2/p/lib" "/Users/michalmaj/.conan2/p/b/libx241a5aaf089156/p/lib" "/Users/michalmaj/.conan2/p/b/libvp15e6422a53151/p/lib" "/Users/michalmaj/.conan2/p/b/libmpb0020dcd28c41/p/lib" "/Users/michalmaj/.conan2/p/b/libfde76d64b2c73cd/p/lib" "/Users/michalmaj/.conan2/p/b/libwe3a7f9d95b97a6/p/lib" "/Users/michalmaj/.conan2/p/b/opens9bf56e2ebc65d/p/lib" "/Users/michalmaj/.conan2/p/b/zliba7921f306428d/p/lib" "/Users/michalmaj/.conan2/p/b/libao65a345ab06376/p/lib" "/Users/michalmaj/.conan2/p/b/dav1d698460c4e0706/p/lib")
list(PREPEND CMAKE_INCLUDE_PATH "/Users/michalmaj/.conan2/p/b/openccca6a0a7c2059/p/include" "/Users/michalmaj/.conan2/p/b/openccca6a0a7c2059/p/include/opencv4" "include" "/Users/michalmaj/.conan2/p/b/proto54fdac56de2ea/p/include" "/Users/michalmaj/.conan2/p/b/ade08e8b03f86db1/p/include" "/Users/michalmaj/.conan2/p/b/opene8abc43ba2bf05/p/include" "/Users/michalmaj/.conan2/p/b/opene8abc43ba2bf05/p/include/OpenEXR" "/Users/michalmaj/.conan2/p/b/imath09f5b1ba5d337/p/include" "/Users/michalmaj/.conan2/p/b/imath09f5b1ba5d337/p/include/Imath" "/Users/michalmaj/.conan2/p/b/libti9d6c2c398a00f/p/include" "/Users/michalmaj/.conan2/p/b/libde0eff2c1b38ac4/p/include" "/Users/michalmaj/.conan2/p/b/libjp3013fbffe5d14/p/include" "/Users/michalmaj/.conan2/p/b/jbig9b08dbdbc0f2e/p/include" "/Users/michalmaj/.conan2/p/b/zstd39aeefd0022b0/p/include" "/Users/michalmaj/.conan2/p/b/quirc877de30ad1087/p/include" "/Users/michalmaj/.conan2/p/b/ffmpe147b26bf1ae86/p/include" "/Users/michalmaj/.conan2/p/b/xz_utcba4be3eb8f04/p/include" "/Users/michalmaj/.conan2/p/b/libicee588c702423d/p/include" "/Users/michalmaj/.conan2/p/b/freet0decf104691c0/p/include" "/Users/michalmaj/.conan2/p/b/freet0decf104691c0/p/include/freetype2" "/Users/michalmaj/.conan2/p/b/libpn8f5988011ee78/p/include" "/Users/michalmaj/.conan2/p/b/bzip2aec6cc871a76d/p/include" "/Users/michalmaj/.conan2/p/b/brotl1017876a3b95f/p/include" "/Users/michalmaj/.conan2/p/b/brotl1017876a3b95f/p/include/brotli" "/Users/michalmaj/.conan2/p/b/openja99b17d74a5ef/p/include" "/Users/michalmaj/.conan2/p/b/openja99b17d74a5ef/p/include/openjpeg-2.5" "/Users/michalmaj/.conan2/p/b/openhd9e0c2f6b0367/p/include" "/Users/michalmaj/.conan2/p/b/vorbia97ce2e270bac/p/include" "/Users/michalmaj/.conan2/p/b/ogga9eb13d871b96/p/include" "/Users/michalmaj/.conan2/p/b/opus9e379e32bcda2/p/include" "/Users/michalmaj/.conan2/p/b/opus9e379e32bcda2/p/include/opus" "/Users/michalmaj/.conan2/p/b/libx27eaae947448b2/p/include" "/Users/michalmaj/.conan2/p/b/libx241a5aaf089156/p/include" "/Users/michalmaj/.conan2/p/b/libvp15e6422a53151/p/include" "/Users/michalmaj/.conan2/p/b/libmpb0020dcd28c41/p/include" "/Users/michalmaj/.conan2/p/b/libfde76d64b2c73cd/p/include" "/Users/michalmaj/.conan2/p/b/libwe3a7f9d95b97a6/p/include" "/Users/michalmaj/.conan2/p/b/opens9bf56e2ebc65d/p/include" "/Users/michalmaj/.conan2/p/b/zliba7921f306428d/p/include" "/Users/michalmaj/.conan2/p/b/libao65a345ab06376/p/include" "/Users/michalmaj/.conan2/p/b/dav1d698460c4e0706/p/include")
set(CONAN_RUNTIME_LIB_DIRS "/Users/michalmaj/.conan2/p/b/openccca6a0a7c2059/p/lib" "lib" "/Users/michalmaj/.conan2/p/b/proto54fdac56de2ea/p/lib" "/Users/michalmaj/.conan2/p/b/ade08e8b03f86db1/p/lib" "/Users/michalmaj/.conan2/p/b/opene8abc43ba2bf05/p/lib" "/Users/michalmaj/.conan2/p/b/imath09f5b1ba5d337/p/lib" "/Users/michalmaj/.conan2/p/b/libti9d6c2c398a00f/p/lib" "/Users/michalmaj/.conan2/p/b/libde0eff2c1b38ac4/p/lib" "/Users/michalmaj/.conan2/p/b/libjp3013fbffe5d14/p/lib" "/Users/michalmaj/.conan2/p/b/jbig9b08dbdbc0f2e/p/lib" "/Users/michalmaj/.conan2/p/b/zstd39aeefd0022b0/p/lib" "/Users/michalmaj/.conan2/p/b/quirc877de30ad1087/p/lib" "/Users/michalmaj/.conan2/p/b/ffmpe147b26bf1ae86/p/lib" "/Users/michalmaj/.conan2/p/b/xz_utcba4be3eb8f04/p/lib" "/Users/michalmaj/.conan2/p/b/libicee588c702423d/p/lib" "/Users/michalmaj/.conan2/p/b/freet0decf104691c0/p/lib" "/Users/michalmaj/.conan2/p/b/libpn8f5988011ee78/p/lib" "/Users/michalmaj/.conan2/p/b/bzip2aec6cc871a76d/p/lib" "/Users/michalmaj/.conan2/p/b/brotl1017876a3b95f/p/lib" "/Users/michalmaj/.conan2/p/b/openja99b17d74a5ef/p/lib" "/Users/michalmaj/.conan2/p/b/openhd9e0c2f6b0367/p/lib" "/Users/michalmaj/.conan2/p/b/vorbia97ce2e270bac/p/lib" "/Users/michalmaj/.conan2/p/b/ogga9eb13d871b96/p/lib" "/Users/michalmaj/.conan2/p/b/opus9e379e32bcda2/p/lib" "/Users/michalmaj/.conan2/p/b/libx27eaae947448b2/p/lib" "/Users/michalmaj/.conan2/p/b/libx241a5aaf089156/p/lib" "/Users/michalmaj/.conan2/p/b/libvp15e6422a53151/p/lib" "/Users/michalmaj/.conan2/p/b/libmpb0020dcd28c41/p/lib" "/Users/michalmaj/.conan2/p/b/libfde76d64b2c73cd/p/lib" "/Users/michalmaj/.conan2/p/b/libwe3a7f9d95b97a6/p/lib" "/Users/michalmaj/.conan2/p/b/opens9bf56e2ebc65d/p/lib" "/Users/michalmaj/.conan2/p/b/zliba7921f306428d/p/lib" "/Users/michalmaj/.conan2/p/b/libao65a345ab06376/p/lib" "/Users/michalmaj/.conan2/p/b/dav1d698460c4e0706/p/lib" )

endif()


########## 'pkg_config' block #############
# Define pkg-config from 'tools.gnu:pkg_config' executable and paths

if (DEFINED ENV{PKG_CONFIG_PATH})
set(ENV{PKG_CONFIG_PATH} "${CMAKE_CURRENT_LIST_DIR}:$ENV{PKG_CONFIG_PATH}")
else()
set(ENV{PKG_CONFIG_PATH} "${CMAKE_CURRENT_LIST_DIR}:")
endif()


########## 'rpath' block #############
# Defining CMAKE_SKIP_RPATH



########## 'output_dirs' block #############
# Definition of CMAKE_INSTALL_XXX folders

set(CMAKE_INSTALL_BINDIR "bin")
set(CMAKE_INSTALL_SBINDIR "bin")
set(CMAKE_INSTALL_LIBEXECDIR "bin")
set(CMAKE_INSTALL_LIBDIR "lib")
set(CMAKE_INSTALL_INCLUDEDIR "include")
set(CMAKE_INSTALL_OLDINCLUDEDIR "include")


########## 'variables' block #############
# Definition of CMake variables from CMakeToolchain.variables values

# Variables
# Variables  per configuration



########## 'preprocessor' block #############
# Preprocessor definitions from CMakeToolchain.preprocessor_definitions values

# Preprocessor definitions per configuration



if(CMAKE_POLICY_DEFAULT_CMP0091)  # Avoid unused and not-initialized warnings
endif()
