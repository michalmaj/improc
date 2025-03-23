# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(libfdk_aac_FRAMEWORKS_FOUND_DEBUG "") # Will be filled later
conan_find_apple_frameworks(libfdk_aac_FRAMEWORKS_FOUND_DEBUG "${libfdk_aac_FRAMEWORKS_DEBUG}" "${libfdk_aac_FRAMEWORK_DIRS_DEBUG}")

set(libfdk_aac_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET libfdk_aac_DEPS_TARGET)
    add_library(libfdk_aac_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET libfdk_aac_DEPS_TARGET
             APPEND PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${libfdk_aac_FRAMEWORKS_FOUND_DEBUG}>
             $<$<CONFIG:Debug>:${libfdk_aac_SYSTEM_LIBS_DEBUG}>
             $<$<CONFIG:Debug>:>)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### libfdk_aac_DEPS_TARGET to all of them
conan_package_library_targets("${libfdk_aac_LIBS_DEBUG}"    # libraries
                              "${libfdk_aac_LIB_DIRS_DEBUG}" # package_libdir
                              "${libfdk_aac_BIN_DIRS_DEBUG}" # package_bindir
                              "${libfdk_aac_LIBRARY_TYPE_DEBUG}"
                              "${libfdk_aac_IS_HOST_WINDOWS_DEBUG}"
                              libfdk_aac_DEPS_TARGET
                              libfdk_aac_LIBRARIES_TARGETS  # out_libraries_targets
                              "_DEBUG"
                              "libfdk_aac"    # package_name
                              "${libfdk_aac_NO_SONAME_MODE_DEBUG}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${libfdk_aac_BUILD_DIRS_DEBUG} ${CMAKE_MODULE_PATH})

########## COMPONENTS TARGET PROPERTIES Debug ########################################

    ########## COMPONENT FDK-AAC::fdk-aac #############

        set(libfdk_aac_FDK-AAC_fdk-aac_FRAMEWORKS_FOUND_DEBUG "")
        conan_find_apple_frameworks(libfdk_aac_FDK-AAC_fdk-aac_FRAMEWORKS_FOUND_DEBUG "${libfdk_aac_FDK-AAC_fdk-aac_FRAMEWORKS_DEBUG}" "${libfdk_aac_FDK-AAC_fdk-aac_FRAMEWORK_DIRS_DEBUG}")

        set(libfdk_aac_FDK-AAC_fdk-aac_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET libfdk_aac_FDK-AAC_fdk-aac_DEPS_TARGET)
            add_library(libfdk_aac_FDK-AAC_fdk-aac_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET libfdk_aac_FDK-AAC_fdk-aac_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${libfdk_aac_FDK-AAC_fdk-aac_FRAMEWORKS_FOUND_DEBUG}>
                     $<$<CONFIG:Debug>:${libfdk_aac_FDK-AAC_fdk-aac_SYSTEM_LIBS_DEBUG}>
                     $<$<CONFIG:Debug>:${libfdk_aac_FDK-AAC_fdk-aac_DEPENDENCIES_DEBUG}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'libfdk_aac_FDK-AAC_fdk-aac_DEPS_TARGET' to all of them
        conan_package_library_targets("${libfdk_aac_FDK-AAC_fdk-aac_LIBS_DEBUG}"
                              "${libfdk_aac_FDK-AAC_fdk-aac_LIB_DIRS_DEBUG}"
                              "${libfdk_aac_FDK-AAC_fdk-aac_BIN_DIRS_DEBUG}" # package_bindir
                              "${libfdk_aac_FDK-AAC_fdk-aac_LIBRARY_TYPE_DEBUG}"
                              "${libfdk_aac_FDK-AAC_fdk-aac_IS_HOST_WINDOWS_DEBUG}"
                              libfdk_aac_FDK-AAC_fdk-aac_DEPS_TARGET
                              libfdk_aac_FDK-AAC_fdk-aac_LIBRARIES_TARGETS
                              "_DEBUG"
                              "libfdk_aac_FDK-AAC_fdk-aac"
                              "${libfdk_aac_FDK-AAC_fdk-aac_NO_SONAME_MODE_DEBUG}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET FDK-AAC::fdk-aac
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${libfdk_aac_FDK-AAC_fdk-aac_OBJECTS_DEBUG}>
                     $<$<CONFIG:Debug>:${libfdk_aac_FDK-AAC_fdk-aac_LIBRARIES_TARGETS}>
                     )

        if("${libfdk_aac_FDK-AAC_fdk-aac_LIBS_DEBUG}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET FDK-AAC::fdk-aac
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         libfdk_aac_FDK-AAC_fdk-aac_DEPS_TARGET)
        endif()

        set_property(TARGET FDK-AAC::fdk-aac APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Debug>:${libfdk_aac_FDK-AAC_fdk-aac_LINKER_FLAGS_DEBUG}>)
        set_property(TARGET FDK-AAC::fdk-aac APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Debug>:${libfdk_aac_FDK-AAC_fdk-aac_INCLUDE_DIRS_DEBUG}>)
        set_property(TARGET FDK-AAC::fdk-aac APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Debug>:${libfdk_aac_FDK-AAC_fdk-aac_LIB_DIRS_DEBUG}>)
        set_property(TARGET FDK-AAC::fdk-aac APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Debug>:${libfdk_aac_FDK-AAC_fdk-aac_COMPILE_DEFINITIONS_DEBUG}>)
        set_property(TARGET FDK-AAC::fdk-aac APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Debug>:${libfdk_aac_FDK-AAC_fdk-aac_COMPILE_OPTIONS_DEBUG}>)

    ########## AGGREGATED GLOBAL TARGET WITH THE COMPONENTS #####################
    set_property(TARGET FDK-AAC::fdk-aac APPEND PROPERTY INTERFACE_LINK_LIBRARIES FDK-AAC::fdk-aac)

########## For the modules (FindXXX)
set(libfdk_aac_LIBRARIES_DEBUG FDK-AAC::fdk-aac)
