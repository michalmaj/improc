# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(openexr_FRAMEWORKS_FOUND_DEBUG "") # Will be filled later
conan_find_apple_frameworks(openexr_FRAMEWORKS_FOUND_DEBUG "${openexr_FRAMEWORKS_DEBUG}" "${openexr_FRAMEWORK_DIRS_DEBUG}")

set(openexr_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET openexr_DEPS_TARGET)
    add_library(openexr_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET openexr_DEPS_TARGET
             APPEND PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${openexr_FRAMEWORKS_FOUND_DEBUG}>
             $<$<CONFIG:Debug>:${openexr_SYSTEM_LIBS_DEBUG}>
             $<$<CONFIG:Debug>:OpenEXR::IexConfig;OpenEXR::IlmThreadConfig;OpenEXR::Iex;OpenEXR::OpenEXRConfig;ZLIB::ZLIB;libdeflate::libdeflate_static;OpenEXR::OpenEXRCore;OpenEXR::IlmThread;Imath::Imath;OpenEXR::OpenEXR>)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### openexr_DEPS_TARGET to all of them
conan_package_library_targets("${openexr_LIBS_DEBUG}"    # libraries
                              "${openexr_LIB_DIRS_DEBUG}" # package_libdir
                              "${openexr_BIN_DIRS_DEBUG}" # package_bindir
                              "${openexr_LIBRARY_TYPE_DEBUG}"
                              "${openexr_IS_HOST_WINDOWS_DEBUG}"
                              openexr_DEPS_TARGET
                              openexr_LIBRARIES_TARGETS  # out_libraries_targets
                              "_DEBUG"
                              "openexr"    # package_name
                              "${openexr_NO_SONAME_MODE_DEBUG}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${openexr_BUILD_DIRS_DEBUG} ${CMAKE_MODULE_PATH})

########## COMPONENTS TARGET PROPERTIES Debug ########################################

    ########## COMPONENT OpenEXR::OpenEXRUtil #############

        set(openexr_OpenEXR_OpenEXRUtil_FRAMEWORKS_FOUND_DEBUG "")
        conan_find_apple_frameworks(openexr_OpenEXR_OpenEXRUtil_FRAMEWORKS_FOUND_DEBUG "${openexr_OpenEXR_OpenEXRUtil_FRAMEWORKS_DEBUG}" "${openexr_OpenEXR_OpenEXRUtil_FRAMEWORK_DIRS_DEBUG}")

        set(openexr_OpenEXR_OpenEXRUtil_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET openexr_OpenEXR_OpenEXRUtil_DEPS_TARGET)
            add_library(openexr_OpenEXR_OpenEXRUtil_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET openexr_OpenEXR_OpenEXRUtil_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRUtil_FRAMEWORKS_FOUND_DEBUG}>
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRUtil_SYSTEM_LIBS_DEBUG}>
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRUtil_DEPENDENCIES_DEBUG}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'openexr_OpenEXR_OpenEXRUtil_DEPS_TARGET' to all of them
        conan_package_library_targets("${openexr_OpenEXR_OpenEXRUtil_LIBS_DEBUG}"
                              "${openexr_OpenEXR_OpenEXRUtil_LIB_DIRS_DEBUG}"
                              "${openexr_OpenEXR_OpenEXRUtil_BIN_DIRS_DEBUG}" # package_bindir
                              "${openexr_OpenEXR_OpenEXRUtil_LIBRARY_TYPE_DEBUG}"
                              "${openexr_OpenEXR_OpenEXRUtil_IS_HOST_WINDOWS_DEBUG}"
                              openexr_OpenEXR_OpenEXRUtil_DEPS_TARGET
                              openexr_OpenEXR_OpenEXRUtil_LIBRARIES_TARGETS
                              "_DEBUG"
                              "openexr_OpenEXR_OpenEXRUtil"
                              "${openexr_OpenEXR_OpenEXRUtil_NO_SONAME_MODE_DEBUG}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET OpenEXR::OpenEXRUtil
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRUtil_OBJECTS_DEBUG}>
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRUtil_LIBRARIES_TARGETS}>
                     )

        if("${openexr_OpenEXR_OpenEXRUtil_LIBS_DEBUG}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET OpenEXR::OpenEXRUtil
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         openexr_OpenEXR_OpenEXRUtil_DEPS_TARGET)
        endif()

        set_property(TARGET OpenEXR::OpenEXRUtil APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRUtil_LINKER_FLAGS_DEBUG}>)
        set_property(TARGET OpenEXR::OpenEXRUtil APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRUtil_INCLUDE_DIRS_DEBUG}>)
        set_property(TARGET OpenEXR::OpenEXRUtil APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRUtil_LIB_DIRS_DEBUG}>)
        set_property(TARGET OpenEXR::OpenEXRUtil APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRUtil_COMPILE_DEFINITIONS_DEBUG}>)
        set_property(TARGET OpenEXR::OpenEXRUtil APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRUtil_COMPILE_OPTIONS_DEBUG}>)

    ########## COMPONENT OpenEXR::OpenEXR #############

        set(openexr_OpenEXR_OpenEXR_FRAMEWORKS_FOUND_DEBUG "")
        conan_find_apple_frameworks(openexr_OpenEXR_OpenEXR_FRAMEWORKS_FOUND_DEBUG "${openexr_OpenEXR_OpenEXR_FRAMEWORKS_DEBUG}" "${openexr_OpenEXR_OpenEXR_FRAMEWORK_DIRS_DEBUG}")

        set(openexr_OpenEXR_OpenEXR_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET openexr_OpenEXR_OpenEXR_DEPS_TARGET)
            add_library(openexr_OpenEXR_OpenEXR_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET openexr_OpenEXR_OpenEXR_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXR_FRAMEWORKS_FOUND_DEBUG}>
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXR_SYSTEM_LIBS_DEBUG}>
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXR_DEPENDENCIES_DEBUG}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'openexr_OpenEXR_OpenEXR_DEPS_TARGET' to all of them
        conan_package_library_targets("${openexr_OpenEXR_OpenEXR_LIBS_DEBUG}"
                              "${openexr_OpenEXR_OpenEXR_LIB_DIRS_DEBUG}"
                              "${openexr_OpenEXR_OpenEXR_BIN_DIRS_DEBUG}" # package_bindir
                              "${openexr_OpenEXR_OpenEXR_LIBRARY_TYPE_DEBUG}"
                              "${openexr_OpenEXR_OpenEXR_IS_HOST_WINDOWS_DEBUG}"
                              openexr_OpenEXR_OpenEXR_DEPS_TARGET
                              openexr_OpenEXR_OpenEXR_LIBRARIES_TARGETS
                              "_DEBUG"
                              "openexr_OpenEXR_OpenEXR"
                              "${openexr_OpenEXR_OpenEXR_NO_SONAME_MODE_DEBUG}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET OpenEXR::OpenEXR
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXR_OBJECTS_DEBUG}>
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXR_LIBRARIES_TARGETS}>
                     )

        if("${openexr_OpenEXR_OpenEXR_LIBS_DEBUG}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET OpenEXR::OpenEXR
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         openexr_OpenEXR_OpenEXR_DEPS_TARGET)
        endif()

        set_property(TARGET OpenEXR::OpenEXR APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXR_LINKER_FLAGS_DEBUG}>)
        set_property(TARGET OpenEXR::OpenEXR APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXR_INCLUDE_DIRS_DEBUG}>)
        set_property(TARGET OpenEXR::OpenEXR APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXR_LIB_DIRS_DEBUG}>)
        set_property(TARGET OpenEXR::OpenEXR APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXR_COMPILE_DEFINITIONS_DEBUG}>)
        set_property(TARGET OpenEXR::OpenEXR APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXR_COMPILE_OPTIONS_DEBUG}>)

    ########## COMPONENT OpenEXR::IlmThread #############

        set(openexr_OpenEXR_IlmThread_FRAMEWORKS_FOUND_DEBUG "")
        conan_find_apple_frameworks(openexr_OpenEXR_IlmThread_FRAMEWORKS_FOUND_DEBUG "${openexr_OpenEXR_IlmThread_FRAMEWORKS_DEBUG}" "${openexr_OpenEXR_IlmThread_FRAMEWORK_DIRS_DEBUG}")

        set(openexr_OpenEXR_IlmThread_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET openexr_OpenEXR_IlmThread_DEPS_TARGET)
            add_library(openexr_OpenEXR_IlmThread_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET openexr_OpenEXR_IlmThread_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IlmThread_FRAMEWORKS_FOUND_DEBUG}>
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IlmThread_SYSTEM_LIBS_DEBUG}>
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IlmThread_DEPENDENCIES_DEBUG}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'openexr_OpenEXR_IlmThread_DEPS_TARGET' to all of them
        conan_package_library_targets("${openexr_OpenEXR_IlmThread_LIBS_DEBUG}"
                              "${openexr_OpenEXR_IlmThread_LIB_DIRS_DEBUG}"
                              "${openexr_OpenEXR_IlmThread_BIN_DIRS_DEBUG}" # package_bindir
                              "${openexr_OpenEXR_IlmThread_LIBRARY_TYPE_DEBUG}"
                              "${openexr_OpenEXR_IlmThread_IS_HOST_WINDOWS_DEBUG}"
                              openexr_OpenEXR_IlmThread_DEPS_TARGET
                              openexr_OpenEXR_IlmThread_LIBRARIES_TARGETS
                              "_DEBUG"
                              "openexr_OpenEXR_IlmThread"
                              "${openexr_OpenEXR_IlmThread_NO_SONAME_MODE_DEBUG}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET OpenEXR::IlmThread
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IlmThread_OBJECTS_DEBUG}>
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IlmThread_LIBRARIES_TARGETS}>
                     )

        if("${openexr_OpenEXR_IlmThread_LIBS_DEBUG}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET OpenEXR::IlmThread
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         openexr_OpenEXR_IlmThread_DEPS_TARGET)
        endif()

        set_property(TARGET OpenEXR::IlmThread APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IlmThread_LINKER_FLAGS_DEBUG}>)
        set_property(TARGET OpenEXR::IlmThread APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IlmThread_INCLUDE_DIRS_DEBUG}>)
        set_property(TARGET OpenEXR::IlmThread APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IlmThread_LIB_DIRS_DEBUG}>)
        set_property(TARGET OpenEXR::IlmThread APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IlmThread_COMPILE_DEFINITIONS_DEBUG}>)
        set_property(TARGET OpenEXR::IlmThread APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IlmThread_COMPILE_OPTIONS_DEBUG}>)

    ########## COMPONENT OpenEXR::OpenEXRCore #############

        set(openexr_OpenEXR_OpenEXRCore_FRAMEWORKS_FOUND_DEBUG "")
        conan_find_apple_frameworks(openexr_OpenEXR_OpenEXRCore_FRAMEWORKS_FOUND_DEBUG "${openexr_OpenEXR_OpenEXRCore_FRAMEWORKS_DEBUG}" "${openexr_OpenEXR_OpenEXRCore_FRAMEWORK_DIRS_DEBUG}")

        set(openexr_OpenEXR_OpenEXRCore_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET openexr_OpenEXR_OpenEXRCore_DEPS_TARGET)
            add_library(openexr_OpenEXR_OpenEXRCore_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET openexr_OpenEXR_OpenEXRCore_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRCore_FRAMEWORKS_FOUND_DEBUG}>
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRCore_SYSTEM_LIBS_DEBUG}>
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRCore_DEPENDENCIES_DEBUG}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'openexr_OpenEXR_OpenEXRCore_DEPS_TARGET' to all of them
        conan_package_library_targets("${openexr_OpenEXR_OpenEXRCore_LIBS_DEBUG}"
                              "${openexr_OpenEXR_OpenEXRCore_LIB_DIRS_DEBUG}"
                              "${openexr_OpenEXR_OpenEXRCore_BIN_DIRS_DEBUG}" # package_bindir
                              "${openexr_OpenEXR_OpenEXRCore_LIBRARY_TYPE_DEBUG}"
                              "${openexr_OpenEXR_OpenEXRCore_IS_HOST_WINDOWS_DEBUG}"
                              openexr_OpenEXR_OpenEXRCore_DEPS_TARGET
                              openexr_OpenEXR_OpenEXRCore_LIBRARIES_TARGETS
                              "_DEBUG"
                              "openexr_OpenEXR_OpenEXRCore"
                              "${openexr_OpenEXR_OpenEXRCore_NO_SONAME_MODE_DEBUG}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET OpenEXR::OpenEXRCore
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRCore_OBJECTS_DEBUG}>
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRCore_LIBRARIES_TARGETS}>
                     )

        if("${openexr_OpenEXR_OpenEXRCore_LIBS_DEBUG}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET OpenEXR::OpenEXRCore
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         openexr_OpenEXR_OpenEXRCore_DEPS_TARGET)
        endif()

        set_property(TARGET OpenEXR::OpenEXRCore APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRCore_LINKER_FLAGS_DEBUG}>)
        set_property(TARGET OpenEXR::OpenEXRCore APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRCore_INCLUDE_DIRS_DEBUG}>)
        set_property(TARGET OpenEXR::OpenEXRCore APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRCore_LIB_DIRS_DEBUG}>)
        set_property(TARGET OpenEXR::OpenEXRCore APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRCore_COMPILE_DEFINITIONS_DEBUG}>)
        set_property(TARGET OpenEXR::OpenEXRCore APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRCore_COMPILE_OPTIONS_DEBUG}>)

    ########## COMPONENT OpenEXR::Iex #############

        set(openexr_OpenEXR_Iex_FRAMEWORKS_FOUND_DEBUG "")
        conan_find_apple_frameworks(openexr_OpenEXR_Iex_FRAMEWORKS_FOUND_DEBUG "${openexr_OpenEXR_Iex_FRAMEWORKS_DEBUG}" "${openexr_OpenEXR_Iex_FRAMEWORK_DIRS_DEBUG}")

        set(openexr_OpenEXR_Iex_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET openexr_OpenEXR_Iex_DEPS_TARGET)
            add_library(openexr_OpenEXR_Iex_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET openexr_OpenEXR_Iex_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_Iex_FRAMEWORKS_FOUND_DEBUG}>
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_Iex_SYSTEM_LIBS_DEBUG}>
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_Iex_DEPENDENCIES_DEBUG}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'openexr_OpenEXR_Iex_DEPS_TARGET' to all of them
        conan_package_library_targets("${openexr_OpenEXR_Iex_LIBS_DEBUG}"
                              "${openexr_OpenEXR_Iex_LIB_DIRS_DEBUG}"
                              "${openexr_OpenEXR_Iex_BIN_DIRS_DEBUG}" # package_bindir
                              "${openexr_OpenEXR_Iex_LIBRARY_TYPE_DEBUG}"
                              "${openexr_OpenEXR_Iex_IS_HOST_WINDOWS_DEBUG}"
                              openexr_OpenEXR_Iex_DEPS_TARGET
                              openexr_OpenEXR_Iex_LIBRARIES_TARGETS
                              "_DEBUG"
                              "openexr_OpenEXR_Iex"
                              "${openexr_OpenEXR_Iex_NO_SONAME_MODE_DEBUG}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET OpenEXR::Iex
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_Iex_OBJECTS_DEBUG}>
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_Iex_LIBRARIES_TARGETS}>
                     )

        if("${openexr_OpenEXR_Iex_LIBS_DEBUG}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET OpenEXR::Iex
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         openexr_OpenEXR_Iex_DEPS_TARGET)
        endif()

        set_property(TARGET OpenEXR::Iex APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_Iex_LINKER_FLAGS_DEBUG}>)
        set_property(TARGET OpenEXR::Iex APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_Iex_INCLUDE_DIRS_DEBUG}>)
        set_property(TARGET OpenEXR::Iex APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_Iex_LIB_DIRS_DEBUG}>)
        set_property(TARGET OpenEXR::Iex APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_Iex_COMPILE_DEFINITIONS_DEBUG}>)
        set_property(TARGET OpenEXR::Iex APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_Iex_COMPILE_OPTIONS_DEBUG}>)

    ########## COMPONENT OpenEXR::IlmThreadConfig #############

        set(openexr_OpenEXR_IlmThreadConfig_FRAMEWORKS_FOUND_DEBUG "")
        conan_find_apple_frameworks(openexr_OpenEXR_IlmThreadConfig_FRAMEWORKS_FOUND_DEBUG "${openexr_OpenEXR_IlmThreadConfig_FRAMEWORKS_DEBUG}" "${openexr_OpenEXR_IlmThreadConfig_FRAMEWORK_DIRS_DEBUG}")

        set(openexr_OpenEXR_IlmThreadConfig_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET openexr_OpenEXR_IlmThreadConfig_DEPS_TARGET)
            add_library(openexr_OpenEXR_IlmThreadConfig_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET openexr_OpenEXR_IlmThreadConfig_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IlmThreadConfig_FRAMEWORKS_FOUND_DEBUG}>
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IlmThreadConfig_SYSTEM_LIBS_DEBUG}>
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IlmThreadConfig_DEPENDENCIES_DEBUG}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'openexr_OpenEXR_IlmThreadConfig_DEPS_TARGET' to all of them
        conan_package_library_targets("${openexr_OpenEXR_IlmThreadConfig_LIBS_DEBUG}"
                              "${openexr_OpenEXR_IlmThreadConfig_LIB_DIRS_DEBUG}"
                              "${openexr_OpenEXR_IlmThreadConfig_BIN_DIRS_DEBUG}" # package_bindir
                              "${openexr_OpenEXR_IlmThreadConfig_LIBRARY_TYPE_DEBUG}"
                              "${openexr_OpenEXR_IlmThreadConfig_IS_HOST_WINDOWS_DEBUG}"
                              openexr_OpenEXR_IlmThreadConfig_DEPS_TARGET
                              openexr_OpenEXR_IlmThreadConfig_LIBRARIES_TARGETS
                              "_DEBUG"
                              "openexr_OpenEXR_IlmThreadConfig"
                              "${openexr_OpenEXR_IlmThreadConfig_NO_SONAME_MODE_DEBUG}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET OpenEXR::IlmThreadConfig
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IlmThreadConfig_OBJECTS_DEBUG}>
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IlmThreadConfig_LIBRARIES_TARGETS}>
                     )

        if("${openexr_OpenEXR_IlmThreadConfig_LIBS_DEBUG}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET OpenEXR::IlmThreadConfig
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         openexr_OpenEXR_IlmThreadConfig_DEPS_TARGET)
        endif()

        set_property(TARGET OpenEXR::IlmThreadConfig APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IlmThreadConfig_LINKER_FLAGS_DEBUG}>)
        set_property(TARGET OpenEXR::IlmThreadConfig APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IlmThreadConfig_INCLUDE_DIRS_DEBUG}>)
        set_property(TARGET OpenEXR::IlmThreadConfig APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IlmThreadConfig_LIB_DIRS_DEBUG}>)
        set_property(TARGET OpenEXR::IlmThreadConfig APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IlmThreadConfig_COMPILE_DEFINITIONS_DEBUG}>)
        set_property(TARGET OpenEXR::IlmThreadConfig APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IlmThreadConfig_COMPILE_OPTIONS_DEBUG}>)

    ########## COMPONENT OpenEXR::IexConfig #############

        set(openexr_OpenEXR_IexConfig_FRAMEWORKS_FOUND_DEBUG "")
        conan_find_apple_frameworks(openexr_OpenEXR_IexConfig_FRAMEWORKS_FOUND_DEBUG "${openexr_OpenEXR_IexConfig_FRAMEWORKS_DEBUG}" "${openexr_OpenEXR_IexConfig_FRAMEWORK_DIRS_DEBUG}")

        set(openexr_OpenEXR_IexConfig_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET openexr_OpenEXR_IexConfig_DEPS_TARGET)
            add_library(openexr_OpenEXR_IexConfig_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET openexr_OpenEXR_IexConfig_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IexConfig_FRAMEWORKS_FOUND_DEBUG}>
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IexConfig_SYSTEM_LIBS_DEBUG}>
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IexConfig_DEPENDENCIES_DEBUG}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'openexr_OpenEXR_IexConfig_DEPS_TARGET' to all of them
        conan_package_library_targets("${openexr_OpenEXR_IexConfig_LIBS_DEBUG}"
                              "${openexr_OpenEXR_IexConfig_LIB_DIRS_DEBUG}"
                              "${openexr_OpenEXR_IexConfig_BIN_DIRS_DEBUG}" # package_bindir
                              "${openexr_OpenEXR_IexConfig_LIBRARY_TYPE_DEBUG}"
                              "${openexr_OpenEXR_IexConfig_IS_HOST_WINDOWS_DEBUG}"
                              openexr_OpenEXR_IexConfig_DEPS_TARGET
                              openexr_OpenEXR_IexConfig_LIBRARIES_TARGETS
                              "_DEBUG"
                              "openexr_OpenEXR_IexConfig"
                              "${openexr_OpenEXR_IexConfig_NO_SONAME_MODE_DEBUG}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET OpenEXR::IexConfig
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IexConfig_OBJECTS_DEBUG}>
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IexConfig_LIBRARIES_TARGETS}>
                     )

        if("${openexr_OpenEXR_IexConfig_LIBS_DEBUG}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET OpenEXR::IexConfig
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         openexr_OpenEXR_IexConfig_DEPS_TARGET)
        endif()

        set_property(TARGET OpenEXR::IexConfig APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IexConfig_LINKER_FLAGS_DEBUG}>)
        set_property(TARGET OpenEXR::IexConfig APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IexConfig_INCLUDE_DIRS_DEBUG}>)
        set_property(TARGET OpenEXR::IexConfig APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IexConfig_LIB_DIRS_DEBUG}>)
        set_property(TARGET OpenEXR::IexConfig APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IexConfig_COMPILE_DEFINITIONS_DEBUG}>)
        set_property(TARGET OpenEXR::IexConfig APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_IexConfig_COMPILE_OPTIONS_DEBUG}>)

    ########## COMPONENT OpenEXR::OpenEXRConfig #############

        set(openexr_OpenEXR_OpenEXRConfig_FRAMEWORKS_FOUND_DEBUG "")
        conan_find_apple_frameworks(openexr_OpenEXR_OpenEXRConfig_FRAMEWORKS_FOUND_DEBUG "${openexr_OpenEXR_OpenEXRConfig_FRAMEWORKS_DEBUG}" "${openexr_OpenEXR_OpenEXRConfig_FRAMEWORK_DIRS_DEBUG}")

        set(openexr_OpenEXR_OpenEXRConfig_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET openexr_OpenEXR_OpenEXRConfig_DEPS_TARGET)
            add_library(openexr_OpenEXR_OpenEXRConfig_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET openexr_OpenEXR_OpenEXRConfig_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRConfig_FRAMEWORKS_FOUND_DEBUG}>
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRConfig_SYSTEM_LIBS_DEBUG}>
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRConfig_DEPENDENCIES_DEBUG}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'openexr_OpenEXR_OpenEXRConfig_DEPS_TARGET' to all of them
        conan_package_library_targets("${openexr_OpenEXR_OpenEXRConfig_LIBS_DEBUG}"
                              "${openexr_OpenEXR_OpenEXRConfig_LIB_DIRS_DEBUG}"
                              "${openexr_OpenEXR_OpenEXRConfig_BIN_DIRS_DEBUG}" # package_bindir
                              "${openexr_OpenEXR_OpenEXRConfig_LIBRARY_TYPE_DEBUG}"
                              "${openexr_OpenEXR_OpenEXRConfig_IS_HOST_WINDOWS_DEBUG}"
                              openexr_OpenEXR_OpenEXRConfig_DEPS_TARGET
                              openexr_OpenEXR_OpenEXRConfig_LIBRARIES_TARGETS
                              "_DEBUG"
                              "openexr_OpenEXR_OpenEXRConfig"
                              "${openexr_OpenEXR_OpenEXRConfig_NO_SONAME_MODE_DEBUG}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET OpenEXR::OpenEXRConfig
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRConfig_OBJECTS_DEBUG}>
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRConfig_LIBRARIES_TARGETS}>
                     )

        if("${openexr_OpenEXR_OpenEXRConfig_LIBS_DEBUG}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET OpenEXR::OpenEXRConfig
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         openexr_OpenEXR_OpenEXRConfig_DEPS_TARGET)
        endif()

        set_property(TARGET OpenEXR::OpenEXRConfig APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRConfig_LINKER_FLAGS_DEBUG}>)
        set_property(TARGET OpenEXR::OpenEXRConfig APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRConfig_INCLUDE_DIRS_DEBUG}>)
        set_property(TARGET OpenEXR::OpenEXRConfig APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRConfig_LIB_DIRS_DEBUG}>)
        set_property(TARGET OpenEXR::OpenEXRConfig APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRConfig_COMPILE_DEFINITIONS_DEBUG}>)
        set_property(TARGET OpenEXR::OpenEXRConfig APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Debug>:${openexr_OpenEXR_OpenEXRConfig_COMPILE_OPTIONS_DEBUG}>)

    ########## AGGREGATED GLOBAL TARGET WITH THE COMPONENTS #####################
    set_property(TARGET openexr::openexr APPEND PROPERTY INTERFACE_LINK_LIBRARIES OpenEXR::OpenEXRUtil)
    set_property(TARGET openexr::openexr APPEND PROPERTY INTERFACE_LINK_LIBRARIES OpenEXR::OpenEXR)
    set_property(TARGET openexr::openexr APPEND PROPERTY INTERFACE_LINK_LIBRARIES OpenEXR::IlmThread)
    set_property(TARGET openexr::openexr APPEND PROPERTY INTERFACE_LINK_LIBRARIES OpenEXR::OpenEXRCore)
    set_property(TARGET openexr::openexr APPEND PROPERTY INTERFACE_LINK_LIBRARIES OpenEXR::Iex)
    set_property(TARGET openexr::openexr APPEND PROPERTY INTERFACE_LINK_LIBRARIES OpenEXR::IlmThreadConfig)
    set_property(TARGET openexr::openexr APPEND PROPERTY INTERFACE_LINK_LIBRARIES OpenEXR::IexConfig)
    set_property(TARGET openexr::openexr APPEND PROPERTY INTERFACE_LINK_LIBRARIES OpenEXR::OpenEXRConfig)

########## For the modules (FindXXX)
set(openexr_LIBRARIES_DEBUG openexr::openexr)
