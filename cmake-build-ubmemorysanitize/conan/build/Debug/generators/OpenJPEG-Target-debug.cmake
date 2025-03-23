# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(openjpeg_FRAMEWORKS_FOUND_DEBUG "") # Will be filled later
conan_find_apple_frameworks(openjpeg_FRAMEWORKS_FOUND_DEBUG "${openjpeg_FRAMEWORKS_DEBUG}" "${openjpeg_FRAMEWORK_DIRS_DEBUG}")

set(openjpeg_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET openjpeg_DEPS_TARGET)
    add_library(openjpeg_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET openjpeg_DEPS_TARGET
             APPEND PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${openjpeg_FRAMEWORKS_FOUND_DEBUG}>
             $<$<CONFIG:Debug>:${openjpeg_SYSTEM_LIBS_DEBUG}>
             $<$<CONFIG:Debug>:>)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### openjpeg_DEPS_TARGET to all of them
conan_package_library_targets("${openjpeg_LIBS_DEBUG}"    # libraries
                              "${openjpeg_LIB_DIRS_DEBUG}" # package_libdir
                              "${openjpeg_BIN_DIRS_DEBUG}" # package_bindir
                              "${openjpeg_LIBRARY_TYPE_DEBUG}"
                              "${openjpeg_IS_HOST_WINDOWS_DEBUG}"
                              openjpeg_DEPS_TARGET
                              openjpeg_LIBRARIES_TARGETS  # out_libraries_targets
                              "_DEBUG"
                              "openjpeg"    # package_name
                              "${openjpeg_NO_SONAME_MODE_DEBUG}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${openjpeg_BUILD_DIRS_DEBUG} ${CMAKE_MODULE_PATH})

########## GLOBAL TARGET PROPERTIES Debug ########################################
    set_property(TARGET openjp2
                 APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                 $<$<CONFIG:Debug>:${openjpeg_OBJECTS_DEBUG}>
                 $<$<CONFIG:Debug>:${openjpeg_LIBRARIES_TARGETS}>
                 )

    if("${openjpeg_LIBS_DEBUG}" STREQUAL "")
        # If the package is not declaring any "cpp_info.libs" the package deps, system libs,
        # frameworks etc are not linked to the imported targets and we need to do it to the
        # global target
        set_property(TARGET openjp2
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     openjpeg_DEPS_TARGET)
    endif()

    set_property(TARGET openjp2
                 APPEND PROPERTY INTERFACE_LINK_OPTIONS
                 $<$<CONFIG:Debug>:${openjpeg_LINKER_FLAGS_DEBUG}>)
    set_property(TARGET openjp2
                 APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                 $<$<CONFIG:Debug>:${openjpeg_INCLUDE_DIRS_DEBUG}>)
    # Necessary to find LINK shared libraries in Linux
    set_property(TARGET openjp2
                 APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                 $<$<CONFIG:Debug>:${openjpeg_LIB_DIRS_DEBUG}>)
    set_property(TARGET openjp2
                 APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                 $<$<CONFIG:Debug>:${openjpeg_COMPILE_DEFINITIONS_DEBUG}>)
    set_property(TARGET openjp2
                 APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                 $<$<CONFIG:Debug>:${openjpeg_COMPILE_OPTIONS_DEBUG}>)

########## For the modules (FindXXX)
set(openjpeg_LIBRARIES_DEBUG openjp2)
