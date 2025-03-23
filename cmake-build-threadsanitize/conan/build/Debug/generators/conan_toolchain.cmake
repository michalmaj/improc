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
list(PREPEND CMAKE_MODULE_PATH "/Users/michalmaj/.conan2/p/b/proto16863157713de/p/lib/cmake/protobuf" "/Users/michalmaj/.conan2/p/b/openj590fc4ad4b2c0/p/lib/cmake" "/Users/michalmaj/.conan2/p/b/opens72225a7ead871/p/lib/cmake")
# the generators folder (where conan generates files, like this toolchain)
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})

# Definition of CMAKE_PREFIX_PATH, CMAKE_XXXXX_PATH
# The explicitly defined "builddirs" of "host" context dependencies must be in PREFIX_PATH
list(PREPEND CMAKE_PREFIX_PATH "/Users/michalmaj/.conan2/p/b/proto16863157713de/p/lib/cmake/protobuf" "/Users/michalmaj/.conan2/p/b/openj590fc4ad4b2c0/p/lib/cmake" "/Users/michalmaj/.conan2/p/b/opens72225a7ead871/p/lib/cmake")
# The Conan local "generators" folder, where this toolchain is saved.
list(PREPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_LIST_DIR} )
list(PREPEND CMAKE_LIBRARY_PATH "/Users/michalmaj/.conan2/p/b/openc5777460f09040/p/lib" "lib" "/Users/michalmaj/.conan2/p/b/proto16863157713de/p/lib" "/Users/michalmaj/.conan2/p/b/aded89f0a8d410ce/p/lib" "/Users/michalmaj/.conan2/p/b/opene025c9f70d1191/p/lib" "/Users/michalmaj/.conan2/p/b/imath51b4bacc7eef7/p/lib" "/Users/michalmaj/.conan2/p/b/libtid42f93cdfe043/p/lib" "/Users/michalmaj/.conan2/p/b/libde64719ff66a324/p/lib" "/Users/michalmaj/.conan2/p/b/libjp3d654569a44d4/p/lib" "/Users/michalmaj/.conan2/p/b/jbig516d4816be6b9/p/lib" "/Users/michalmaj/.conan2/p/b/zstd8a1f28025d875/p/lib" "/Users/michalmaj/.conan2/p/b/quirc275dbee08385e/p/lib" "/Users/michalmaj/.conan2/p/b/ffmpe6d9d61c457f1f/p/lib" "/Users/michalmaj/.conan2/p/b/xz_ut8d204187d32c2/p/lib" "/Users/michalmaj/.conan2/p/b/libic641e44fb5a28b/p/lib" "/Users/michalmaj/.conan2/p/b/freet83b112aea9552/p/lib" "/Users/michalmaj/.conan2/p/b/libpndc59ec7b95fe7/p/lib" "/Users/michalmaj/.conan2/p/b/bzip22228d1cced555/p/lib" "/Users/michalmaj/.conan2/p/b/brotl9cc0e986fdfd9/p/lib" "/Users/michalmaj/.conan2/p/b/openj590fc4ad4b2c0/p/lib" "/Users/michalmaj/.conan2/p/b/openh1378742f09d05/p/lib" "/Users/michalmaj/.conan2/p/b/vorbi2f9200ea5102f/p/lib" "/Users/michalmaj/.conan2/p/b/oggdea2728db8451/p/lib" "/Users/michalmaj/.conan2/p/b/opus4dbbb31bf0594/p/lib" "/Users/michalmaj/.conan2/p/b/libx239d4ea55f45ac/p/lib" "/Users/michalmaj/.conan2/p/b/libx24905fd99732ba/p/lib" "/Users/michalmaj/.conan2/p/b/libvpafd42c3a4fae7/p/lib" "/Users/michalmaj/.conan2/p/b/libmpfd83ff08acb45/p/lib" "/Users/michalmaj/.conan2/p/b/libfd9db9e95ba1ef8/p/lib" "/Users/michalmaj/.conan2/p/b/libweb3c8e47b43282/p/lib" "/Users/michalmaj/.conan2/p/b/opens72225a7ead871/p/lib" "/Users/michalmaj/.conan2/p/b/zlib34ea3696bc2f9/p/lib" "/Users/michalmaj/.conan2/p/b/libaoc05c759761d60/p/lib" "/Users/michalmaj/.conan2/p/b/dav1dd728572d65e54/p/lib")
list(PREPEND CMAKE_INCLUDE_PATH "/Users/michalmaj/.conan2/p/b/openc5777460f09040/p/include" "/Users/michalmaj/.conan2/p/b/openc5777460f09040/p/include/opencv4" "include" "/Users/michalmaj/.conan2/p/b/proto16863157713de/p/include" "/Users/michalmaj/.conan2/p/b/aded89f0a8d410ce/p/include" "/Users/michalmaj/.conan2/p/b/opene025c9f70d1191/p/include" "/Users/michalmaj/.conan2/p/b/opene025c9f70d1191/p/include/OpenEXR" "/Users/michalmaj/.conan2/p/b/imath51b4bacc7eef7/p/include" "/Users/michalmaj/.conan2/p/b/imath51b4bacc7eef7/p/include/Imath" "/Users/michalmaj/.conan2/p/b/libtid42f93cdfe043/p/include" "/Users/michalmaj/.conan2/p/b/libde64719ff66a324/p/include" "/Users/michalmaj/.conan2/p/b/libjp3d654569a44d4/p/include" "/Users/michalmaj/.conan2/p/b/jbig516d4816be6b9/p/include" "/Users/michalmaj/.conan2/p/b/zstd8a1f28025d875/p/include" "/Users/michalmaj/.conan2/p/b/quirc275dbee08385e/p/include" "/Users/michalmaj/.conan2/p/b/ffmpe6d9d61c457f1f/p/include" "/Users/michalmaj/.conan2/p/b/xz_ut8d204187d32c2/p/include" "/Users/michalmaj/.conan2/p/b/libic641e44fb5a28b/p/include" "/Users/michalmaj/.conan2/p/b/freet83b112aea9552/p/include" "/Users/michalmaj/.conan2/p/b/freet83b112aea9552/p/include/freetype2" "/Users/michalmaj/.conan2/p/b/libpndc59ec7b95fe7/p/include" "/Users/michalmaj/.conan2/p/b/bzip22228d1cced555/p/include" "/Users/michalmaj/.conan2/p/b/brotl9cc0e986fdfd9/p/include" "/Users/michalmaj/.conan2/p/b/brotl9cc0e986fdfd9/p/include/brotli" "/Users/michalmaj/.conan2/p/b/openj590fc4ad4b2c0/p/include" "/Users/michalmaj/.conan2/p/b/openj590fc4ad4b2c0/p/include/openjpeg-2.5" "/Users/michalmaj/.conan2/p/b/openh1378742f09d05/p/include" "/Users/michalmaj/.conan2/p/b/vorbi2f9200ea5102f/p/include" "/Users/michalmaj/.conan2/p/b/oggdea2728db8451/p/include" "/Users/michalmaj/.conan2/p/b/opus4dbbb31bf0594/p/include" "/Users/michalmaj/.conan2/p/b/opus4dbbb31bf0594/p/include/opus" "/Users/michalmaj/.conan2/p/b/libx239d4ea55f45ac/p/include" "/Users/michalmaj/.conan2/p/b/libx24905fd99732ba/p/include" "/Users/michalmaj/.conan2/p/b/libvpafd42c3a4fae7/p/include" "/Users/michalmaj/.conan2/p/b/libmpfd83ff08acb45/p/include" "/Users/michalmaj/.conan2/p/b/libfd9db9e95ba1ef8/p/include" "/Users/michalmaj/.conan2/p/b/libweb3c8e47b43282/p/include" "/Users/michalmaj/.conan2/p/b/opens72225a7ead871/p/include" "/Users/michalmaj/.conan2/p/b/zlib34ea3696bc2f9/p/include" "/Users/michalmaj/.conan2/p/b/libaoc05c759761d60/p/include" "/Users/michalmaj/.conan2/p/b/dav1dd728572d65e54/p/include")
set(CONAN_RUNTIME_LIB_DIRS "/Users/michalmaj/.conan2/p/b/openc5777460f09040/p/lib" "lib" "/Users/michalmaj/.conan2/p/b/proto16863157713de/p/lib" "/Users/michalmaj/.conan2/p/b/aded89f0a8d410ce/p/lib" "/Users/michalmaj/.conan2/p/b/opene025c9f70d1191/p/lib" "/Users/michalmaj/.conan2/p/b/imath51b4bacc7eef7/p/lib" "/Users/michalmaj/.conan2/p/b/libtid42f93cdfe043/p/lib" "/Users/michalmaj/.conan2/p/b/libde64719ff66a324/p/lib" "/Users/michalmaj/.conan2/p/b/libjp3d654569a44d4/p/lib" "/Users/michalmaj/.conan2/p/b/jbig516d4816be6b9/p/lib" "/Users/michalmaj/.conan2/p/b/zstd8a1f28025d875/p/lib" "/Users/michalmaj/.conan2/p/b/quirc275dbee08385e/p/lib" "/Users/michalmaj/.conan2/p/b/ffmpe6d9d61c457f1f/p/lib" "/Users/michalmaj/.conan2/p/b/xz_ut8d204187d32c2/p/lib" "/Users/michalmaj/.conan2/p/b/libic641e44fb5a28b/p/lib" "/Users/michalmaj/.conan2/p/b/freet83b112aea9552/p/lib" "/Users/michalmaj/.conan2/p/b/libpndc59ec7b95fe7/p/lib" "/Users/michalmaj/.conan2/p/b/bzip22228d1cced555/p/lib" "/Users/michalmaj/.conan2/p/b/brotl9cc0e986fdfd9/p/lib" "/Users/michalmaj/.conan2/p/b/openj590fc4ad4b2c0/p/lib" "/Users/michalmaj/.conan2/p/b/openh1378742f09d05/p/lib" "/Users/michalmaj/.conan2/p/b/vorbi2f9200ea5102f/p/lib" "/Users/michalmaj/.conan2/p/b/oggdea2728db8451/p/lib" "/Users/michalmaj/.conan2/p/b/opus4dbbb31bf0594/p/lib" "/Users/michalmaj/.conan2/p/b/libx239d4ea55f45ac/p/lib" "/Users/michalmaj/.conan2/p/b/libx24905fd99732ba/p/lib" "/Users/michalmaj/.conan2/p/b/libvpafd42c3a4fae7/p/lib" "/Users/michalmaj/.conan2/p/b/libmpfd83ff08acb45/p/lib" "/Users/michalmaj/.conan2/p/b/libfd9db9e95ba1ef8/p/lib" "/Users/michalmaj/.conan2/p/b/libweb3c8e47b43282/p/lib" "/Users/michalmaj/.conan2/p/b/opens72225a7ead871/p/lib" "/Users/michalmaj/.conan2/p/b/zlib34ea3696bc2f9/p/lib" "/Users/michalmaj/.conan2/p/b/libaoc05c759761d60/p/lib" "/Users/michalmaj/.conan2/p/b/dav1dd728572d65e54/p/lib" )

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
