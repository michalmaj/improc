########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

list(APPEND libfdk_aac_COMPONENT_NAMES FDK-AAC::fdk-aac)
list(REMOVE_DUPLICATES libfdk_aac_COMPONENT_NAMES)
if(DEFINED libfdk_aac_FIND_DEPENDENCY_NAMES)
  list(APPEND libfdk_aac_FIND_DEPENDENCY_NAMES )
  list(REMOVE_DUPLICATES libfdk_aac_FIND_DEPENDENCY_NAMES)
else()
  set(libfdk_aac_FIND_DEPENDENCY_NAMES )
endif()

########### VARIABLES #######################################################################
#############################################################################################
set(libfdk_aac_PACKAGE_FOLDER_DEBUG "/Users/michalmaj/.conan2/p/b/libfd9db9e95ba1ef8/p")
set(libfdk_aac_BUILD_MODULES_PATHS_DEBUG )


set(libfdk_aac_INCLUDE_DIRS_DEBUG )
set(libfdk_aac_RES_DIRS_DEBUG )
set(libfdk_aac_DEFINITIONS_DEBUG )
set(libfdk_aac_SHARED_LINK_FLAGS_DEBUG )
set(libfdk_aac_EXE_LINK_FLAGS_DEBUG )
set(libfdk_aac_OBJECTS_DEBUG )
set(libfdk_aac_COMPILE_DEFINITIONS_DEBUG )
set(libfdk_aac_COMPILE_OPTIONS_C_DEBUG )
set(libfdk_aac_COMPILE_OPTIONS_CXX_DEBUG )
set(libfdk_aac_LIB_DIRS_DEBUG "${libfdk_aac_PACKAGE_FOLDER_DEBUG}/lib")
set(libfdk_aac_BIN_DIRS_DEBUG )
set(libfdk_aac_LIBRARY_TYPE_DEBUG STATIC)
set(libfdk_aac_IS_HOST_WINDOWS_DEBUG 0)
set(libfdk_aac_LIBS_DEBUG fdk-aac)
set(libfdk_aac_SYSTEM_LIBS_DEBUG )
set(libfdk_aac_FRAMEWORK_DIRS_DEBUG )
set(libfdk_aac_FRAMEWORKS_DEBUG )
set(libfdk_aac_BUILD_DIRS_DEBUG )
set(libfdk_aac_NO_SONAME_MODE_DEBUG FALSE)


# COMPOUND VARIABLES
set(libfdk_aac_COMPILE_OPTIONS_DEBUG
    "$<$<COMPILE_LANGUAGE:CXX>:${libfdk_aac_COMPILE_OPTIONS_CXX_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:C>:${libfdk_aac_COMPILE_OPTIONS_C_DEBUG}>")
set(libfdk_aac_LINKER_FLAGS_DEBUG
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${libfdk_aac_SHARED_LINK_FLAGS_DEBUG}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${libfdk_aac_SHARED_LINK_FLAGS_DEBUG}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${libfdk_aac_EXE_LINK_FLAGS_DEBUG}>")


set(libfdk_aac_COMPONENTS_DEBUG FDK-AAC::fdk-aac)
########### COMPONENT FDK-AAC::fdk-aac VARIABLES ############################################

set(libfdk_aac_FDK-AAC_fdk-aac_INCLUDE_DIRS_DEBUG )
set(libfdk_aac_FDK-AAC_fdk-aac_LIB_DIRS_DEBUG "${libfdk_aac_PACKAGE_FOLDER_DEBUG}/lib")
set(libfdk_aac_FDK-AAC_fdk-aac_BIN_DIRS_DEBUG )
set(libfdk_aac_FDK-AAC_fdk-aac_LIBRARY_TYPE_DEBUG STATIC)
set(libfdk_aac_FDK-AAC_fdk-aac_IS_HOST_WINDOWS_DEBUG 0)
set(libfdk_aac_FDK-AAC_fdk-aac_RES_DIRS_DEBUG )
set(libfdk_aac_FDK-AAC_fdk-aac_DEFINITIONS_DEBUG )
set(libfdk_aac_FDK-AAC_fdk-aac_OBJECTS_DEBUG )
set(libfdk_aac_FDK-AAC_fdk-aac_COMPILE_DEFINITIONS_DEBUG )
set(libfdk_aac_FDK-AAC_fdk-aac_COMPILE_OPTIONS_C_DEBUG "")
set(libfdk_aac_FDK-AAC_fdk-aac_COMPILE_OPTIONS_CXX_DEBUG "")
set(libfdk_aac_FDK-AAC_fdk-aac_LIBS_DEBUG fdk-aac)
set(libfdk_aac_FDK-AAC_fdk-aac_SYSTEM_LIBS_DEBUG )
set(libfdk_aac_FDK-AAC_fdk-aac_FRAMEWORK_DIRS_DEBUG )
set(libfdk_aac_FDK-AAC_fdk-aac_FRAMEWORKS_DEBUG )
set(libfdk_aac_FDK-AAC_fdk-aac_DEPENDENCIES_DEBUG )
set(libfdk_aac_FDK-AAC_fdk-aac_SHARED_LINK_FLAGS_DEBUG )
set(libfdk_aac_FDK-AAC_fdk-aac_EXE_LINK_FLAGS_DEBUG )
set(libfdk_aac_FDK-AAC_fdk-aac_NO_SONAME_MODE_DEBUG FALSE)

# COMPOUND VARIABLES
set(libfdk_aac_FDK-AAC_fdk-aac_LINKER_FLAGS_DEBUG
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${libfdk_aac_FDK-AAC_fdk-aac_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${libfdk_aac_FDK-AAC_fdk-aac_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${libfdk_aac_FDK-AAC_fdk-aac_EXE_LINK_FLAGS_DEBUG}>
)
set(libfdk_aac_FDK-AAC_fdk-aac_COMPILE_OPTIONS_DEBUG
    "$<$<COMPILE_LANGUAGE:CXX>:${libfdk_aac_FDK-AAC_fdk-aac_COMPILE_OPTIONS_CXX_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:C>:${libfdk_aac_FDK-AAC_fdk-aac_COMPILE_OPTIONS_C_DEBUG}>")