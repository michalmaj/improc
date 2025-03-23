# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(ade_FRAMEWORKS_FOUND_DEBUG "") # Will be filled later
conan_find_apple_frameworks(ade_FRAMEWORKS_FOUND_DEBUG "${ade_FRAMEWORKS_DEBUG}" "${ade_FRAMEWORK_DIRS_DEBUG}")

set(ade_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET ade_DEPS_TARGET)
    add_library(ade_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET ade_DEPS_TARGET
             APPEND PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${ade_FRAMEWORKS_FOUND_DEBUG}>
             $<$<CONFIG:Debug>:${ade_SYSTEM_LIBS_DEBUG}>
             $<$<CONFIG:Debug>:>)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### ade_DEPS_TARGET to all of them
conan_package_library_targets("${ade_LIBS_DEBUG}"    # libraries
                              "${ade_LIB_DIRS_DEBUG}" # package_libdir
                              "${ade_BIN_DIRS_DEBUG}" # package_bindir
                              "${ade_LIBRARY_TYPE_DEBUG}"
                              "${ade_IS_HOST_WINDOWS_DEBUG}"
                              ade_DEPS_TARGET
                              ade_LIBRARIES_TARGETS  # out_libraries_targets
                              "_DEBUG"
                              "ade"    # package_name
                              "${ade_NO_SONAME_MODE_DEBUG}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${ade_BUILD_DIRS_DEBUG} ${CMAKE_MODULE_PATH})

########## GLOBAL TARGET PROPERTIES Debug ########################################
    set_property(TARGET ade
                 APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                 $<$<CONFIG:Debug>:${ade_OBJECTS_DEBUG}>
                 $<$<CONFIG:Debug>:${ade_LIBRARIES_TARGETS}>
                 )

    if("${ade_LIBS_DEBUG}" STREQUAL "")
        # If the package is not declaring any "cpp_info.libs" the package deps, system libs,
        # frameworks etc are not linked to the imported targets and we need to do it to the
        # global target
        set_property(TARGET ade
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     ade_DEPS_TARGET)
    endif()

    set_property(TARGET ade
                 APPEND PROPERTY INTERFACE_LINK_OPTIONS
                 $<$<CONFIG:Debug>:${ade_LINKER_FLAGS_DEBUG}>)
    set_property(TARGET ade
                 APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                 $<$<CONFIG:Debug>:${ade_INCLUDE_DIRS_DEBUG}>)
    # Necessary to find LINK shared libraries in Linux
    set_property(TARGET ade
                 APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                 $<$<CONFIG:Debug>:${ade_LIB_DIRS_DEBUG}>)
    set_property(TARGET ade
                 APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                 $<$<CONFIG:Debug>:${ade_COMPILE_DEFINITIONS_DEBUG}>)
    set_property(TARGET ade
                 APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                 $<$<CONFIG:Debug>:${ade_COMPILE_OPTIONS_DEBUG}>)

########## For the modules (FindXXX)
set(ade_LIBRARIES_DEBUG ade)
