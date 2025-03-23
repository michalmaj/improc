# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(quirc_FRAMEWORKS_FOUND_DEBUG "") # Will be filled later
conan_find_apple_frameworks(quirc_FRAMEWORKS_FOUND_DEBUG "${quirc_FRAMEWORKS_DEBUG}" "${quirc_FRAMEWORK_DIRS_DEBUG}")

set(quirc_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET quirc_DEPS_TARGET)
    add_library(quirc_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET quirc_DEPS_TARGET
             APPEND PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${quirc_FRAMEWORKS_FOUND_DEBUG}>
             $<$<CONFIG:Debug>:${quirc_SYSTEM_LIBS_DEBUG}>
             $<$<CONFIG:Debug>:>)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### quirc_DEPS_TARGET to all of them
conan_package_library_targets("${quirc_LIBS_DEBUG}"    # libraries
                              "${quirc_LIB_DIRS_DEBUG}" # package_libdir
                              "${quirc_BIN_DIRS_DEBUG}" # package_bindir
                              "${quirc_LIBRARY_TYPE_DEBUG}"
                              "${quirc_IS_HOST_WINDOWS_DEBUG}"
                              quirc_DEPS_TARGET
                              quirc_LIBRARIES_TARGETS  # out_libraries_targets
                              "_DEBUG"
                              "quirc"    # package_name
                              "${quirc_NO_SONAME_MODE_DEBUG}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${quirc_BUILD_DIRS_DEBUG} ${CMAKE_MODULE_PATH})

########## GLOBAL TARGET PROPERTIES Debug ########################################
    set_property(TARGET quirc::quirc
                 APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                 $<$<CONFIG:Debug>:${quirc_OBJECTS_DEBUG}>
                 $<$<CONFIG:Debug>:${quirc_LIBRARIES_TARGETS}>
                 )

    if("${quirc_LIBS_DEBUG}" STREQUAL "")
        # If the package is not declaring any "cpp_info.libs" the package deps, system libs,
        # frameworks etc are not linked to the imported targets and we need to do it to the
        # global target
        set_property(TARGET quirc::quirc
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     quirc_DEPS_TARGET)
    endif()

    set_property(TARGET quirc::quirc
                 APPEND PROPERTY INTERFACE_LINK_OPTIONS
                 $<$<CONFIG:Debug>:${quirc_LINKER_FLAGS_DEBUG}>)
    set_property(TARGET quirc::quirc
                 APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                 $<$<CONFIG:Debug>:${quirc_INCLUDE_DIRS_DEBUG}>)
    # Necessary to find LINK shared libraries in Linux
    set_property(TARGET quirc::quirc
                 APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                 $<$<CONFIG:Debug>:${quirc_LIB_DIRS_DEBUG}>)
    set_property(TARGET quirc::quirc
                 APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                 $<$<CONFIG:Debug>:${quirc_COMPILE_DEFINITIONS_DEBUG}>)
    set_property(TARGET quirc::quirc
                 APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                 $<$<CONFIG:Debug>:${quirc_COMPILE_OPTIONS_DEBUG}>)

########## For the modules (FindXXX)
set(quirc_LIBRARIES_DEBUG quirc::quirc)
