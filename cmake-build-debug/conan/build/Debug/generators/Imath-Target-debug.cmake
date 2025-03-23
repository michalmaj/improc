# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(imath_FRAMEWORKS_FOUND_DEBUG "") # Will be filled later
conan_find_apple_frameworks(imath_FRAMEWORKS_FOUND_DEBUG "${imath_FRAMEWORKS_DEBUG}" "${imath_FRAMEWORK_DIRS_DEBUG}")

set(imath_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET imath_DEPS_TARGET)
    add_library(imath_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET imath_DEPS_TARGET
             APPEND PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${imath_FRAMEWORKS_FOUND_DEBUG}>
             $<$<CONFIG:Debug>:${imath_SYSTEM_LIBS_DEBUG}>
             $<$<CONFIG:Debug>:Imath::ImathConfig>)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### imath_DEPS_TARGET to all of them
conan_package_library_targets("${imath_LIBS_DEBUG}"    # libraries
                              "${imath_LIB_DIRS_DEBUG}" # package_libdir
                              "${imath_BIN_DIRS_DEBUG}" # package_bindir
                              "${imath_LIBRARY_TYPE_DEBUG}"
                              "${imath_IS_HOST_WINDOWS_DEBUG}"
                              imath_DEPS_TARGET
                              imath_LIBRARIES_TARGETS  # out_libraries_targets
                              "_DEBUG"
                              "imath"    # package_name
                              "${imath_NO_SONAME_MODE_DEBUG}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${imath_BUILD_DIRS_DEBUG} ${CMAKE_MODULE_PATH})

########## COMPONENTS TARGET PROPERTIES Debug ########################################

    ########## COMPONENT Imath::Imath #############

        set(imath_Imath_Imath_FRAMEWORKS_FOUND_DEBUG "")
        conan_find_apple_frameworks(imath_Imath_Imath_FRAMEWORKS_FOUND_DEBUG "${imath_Imath_Imath_FRAMEWORKS_DEBUG}" "${imath_Imath_Imath_FRAMEWORK_DIRS_DEBUG}")

        set(imath_Imath_Imath_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET imath_Imath_Imath_DEPS_TARGET)
            add_library(imath_Imath_Imath_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET imath_Imath_Imath_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${imath_Imath_Imath_FRAMEWORKS_FOUND_DEBUG}>
                     $<$<CONFIG:Debug>:${imath_Imath_Imath_SYSTEM_LIBS_DEBUG}>
                     $<$<CONFIG:Debug>:${imath_Imath_Imath_DEPENDENCIES_DEBUG}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'imath_Imath_Imath_DEPS_TARGET' to all of them
        conan_package_library_targets("${imath_Imath_Imath_LIBS_DEBUG}"
                              "${imath_Imath_Imath_LIB_DIRS_DEBUG}"
                              "${imath_Imath_Imath_BIN_DIRS_DEBUG}" # package_bindir
                              "${imath_Imath_Imath_LIBRARY_TYPE_DEBUG}"
                              "${imath_Imath_Imath_IS_HOST_WINDOWS_DEBUG}"
                              imath_Imath_Imath_DEPS_TARGET
                              imath_Imath_Imath_LIBRARIES_TARGETS
                              "_DEBUG"
                              "imath_Imath_Imath"
                              "${imath_Imath_Imath_NO_SONAME_MODE_DEBUG}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET Imath::Imath
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${imath_Imath_Imath_OBJECTS_DEBUG}>
                     $<$<CONFIG:Debug>:${imath_Imath_Imath_LIBRARIES_TARGETS}>
                     )

        if("${imath_Imath_Imath_LIBS_DEBUG}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET Imath::Imath
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         imath_Imath_Imath_DEPS_TARGET)
        endif()

        set_property(TARGET Imath::Imath APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Debug>:${imath_Imath_Imath_LINKER_FLAGS_DEBUG}>)
        set_property(TARGET Imath::Imath APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Debug>:${imath_Imath_Imath_INCLUDE_DIRS_DEBUG}>)
        set_property(TARGET Imath::Imath APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Debug>:${imath_Imath_Imath_LIB_DIRS_DEBUG}>)
        set_property(TARGET Imath::Imath APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Debug>:${imath_Imath_Imath_COMPILE_DEFINITIONS_DEBUG}>)
        set_property(TARGET Imath::Imath APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Debug>:${imath_Imath_Imath_COMPILE_OPTIONS_DEBUG}>)

    ########## COMPONENT Imath::ImathConfig #############

        set(imath_Imath_ImathConfig_FRAMEWORKS_FOUND_DEBUG "")
        conan_find_apple_frameworks(imath_Imath_ImathConfig_FRAMEWORKS_FOUND_DEBUG "${imath_Imath_ImathConfig_FRAMEWORKS_DEBUG}" "${imath_Imath_ImathConfig_FRAMEWORK_DIRS_DEBUG}")

        set(imath_Imath_ImathConfig_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET imath_Imath_ImathConfig_DEPS_TARGET)
            add_library(imath_Imath_ImathConfig_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET imath_Imath_ImathConfig_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${imath_Imath_ImathConfig_FRAMEWORKS_FOUND_DEBUG}>
                     $<$<CONFIG:Debug>:${imath_Imath_ImathConfig_SYSTEM_LIBS_DEBUG}>
                     $<$<CONFIG:Debug>:${imath_Imath_ImathConfig_DEPENDENCIES_DEBUG}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'imath_Imath_ImathConfig_DEPS_TARGET' to all of them
        conan_package_library_targets("${imath_Imath_ImathConfig_LIBS_DEBUG}"
                              "${imath_Imath_ImathConfig_LIB_DIRS_DEBUG}"
                              "${imath_Imath_ImathConfig_BIN_DIRS_DEBUG}" # package_bindir
                              "${imath_Imath_ImathConfig_LIBRARY_TYPE_DEBUG}"
                              "${imath_Imath_ImathConfig_IS_HOST_WINDOWS_DEBUG}"
                              imath_Imath_ImathConfig_DEPS_TARGET
                              imath_Imath_ImathConfig_LIBRARIES_TARGETS
                              "_DEBUG"
                              "imath_Imath_ImathConfig"
                              "${imath_Imath_ImathConfig_NO_SONAME_MODE_DEBUG}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET Imath::ImathConfig
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Debug>:${imath_Imath_ImathConfig_OBJECTS_DEBUG}>
                     $<$<CONFIG:Debug>:${imath_Imath_ImathConfig_LIBRARIES_TARGETS}>
                     )

        if("${imath_Imath_ImathConfig_LIBS_DEBUG}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET Imath::ImathConfig
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         imath_Imath_ImathConfig_DEPS_TARGET)
        endif()

        set_property(TARGET Imath::ImathConfig APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Debug>:${imath_Imath_ImathConfig_LINKER_FLAGS_DEBUG}>)
        set_property(TARGET Imath::ImathConfig APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Debug>:${imath_Imath_ImathConfig_INCLUDE_DIRS_DEBUG}>)
        set_property(TARGET Imath::ImathConfig APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Debug>:${imath_Imath_ImathConfig_LIB_DIRS_DEBUG}>)
        set_property(TARGET Imath::ImathConfig APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Debug>:${imath_Imath_ImathConfig_COMPILE_DEFINITIONS_DEBUG}>)
        set_property(TARGET Imath::ImathConfig APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Debug>:${imath_Imath_ImathConfig_COMPILE_OPTIONS_DEBUG}>)

    ########## AGGREGATED GLOBAL TARGET WITH THE COMPONENTS #####################
    set_property(TARGET Imath::Imath APPEND PROPERTY INTERFACE_LINK_LIBRARIES Imath::Imath)
    set_property(TARGET Imath::Imath APPEND PROPERTY INTERFACE_LINK_LIBRARIES Imath::ImathConfig)

########## For the modules (FindXXX)
set(imath_LIBRARIES_DEBUG Imath::Imath)
