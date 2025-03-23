# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(jbig_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
conan_find_apple_frameworks(jbig_FRAMEWORKS_FOUND_RELEASE "${jbig_FRAMEWORKS_RELEASE}" "${jbig_FRAMEWORK_DIRS_RELEASE}")

set(jbig_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET jbig_DEPS_TARGET)
    add_library(jbig_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET jbig_DEPS_TARGET
             APPEND PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Release>:${jbig_FRAMEWORKS_FOUND_RELEASE}>
             $<$<CONFIG:Release>:${jbig_SYSTEM_LIBS_RELEASE}>
             $<$<CONFIG:Release>:>)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### jbig_DEPS_TARGET to all of them
conan_package_library_targets("${jbig_LIBS_RELEASE}"    # libraries
                              "${jbig_LIB_DIRS_RELEASE}" # package_libdir
                              "${jbig_BIN_DIRS_RELEASE}" # package_bindir
                              "${jbig_LIBRARY_TYPE_RELEASE}"
                              "${jbig_IS_HOST_WINDOWS_RELEASE}"
                              jbig_DEPS_TARGET
                              jbig_LIBRARIES_TARGETS  # out_libraries_targets
                              "_RELEASE"
                              "jbig"    # package_name
                              "${jbig_NO_SONAME_MODE_RELEASE}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${jbig_BUILD_DIRS_RELEASE} ${CMAKE_MODULE_PATH})

########## GLOBAL TARGET PROPERTIES Release ########################################
    set_property(TARGET jbig::jbig
                 APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                 $<$<CONFIG:Release>:${jbig_OBJECTS_RELEASE}>
                 $<$<CONFIG:Release>:${jbig_LIBRARIES_TARGETS}>
                 )

    if("${jbig_LIBS_RELEASE}" STREQUAL "")
        # If the package is not declaring any "cpp_info.libs" the package deps, system libs,
        # frameworks etc are not linked to the imported targets and we need to do it to the
        # global target
        set_property(TARGET jbig::jbig
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     jbig_DEPS_TARGET)
    endif()

    set_property(TARGET jbig::jbig
                 APPEND PROPERTY INTERFACE_LINK_OPTIONS
                 $<$<CONFIG:Release>:${jbig_LINKER_FLAGS_RELEASE}>)
    set_property(TARGET jbig::jbig
                 APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                 $<$<CONFIG:Release>:${jbig_INCLUDE_DIRS_RELEASE}>)
    # Necessary to find LINK shared libraries in Linux
    set_property(TARGET jbig::jbig
                 APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                 $<$<CONFIG:Release>:${jbig_LIB_DIRS_RELEASE}>)
    set_property(TARGET jbig::jbig
                 APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                 $<$<CONFIG:Release>:${jbig_COMPILE_DEFINITIONS_RELEASE}>)
    set_property(TARGET jbig::jbig
                 APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                 $<$<CONFIG:Release>:${jbig_COMPILE_OPTIONS_RELEASE}>)

########## For the modules (FindXXX)
set(jbig_LIBRARIES_RELEASE jbig::jbig)
