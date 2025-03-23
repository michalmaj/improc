########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

list(APPEND openexr_COMPONENT_NAMES OpenEXR::OpenEXRConfig OpenEXR::IexConfig OpenEXR::IlmThreadConfig OpenEXR::Iex OpenEXR::OpenEXRCore OpenEXR::IlmThread OpenEXR::OpenEXR OpenEXR::OpenEXRUtil)
list(REMOVE_DUPLICATES openexr_COMPONENT_NAMES)
if(DEFINED openexr_FIND_DEPENDENCY_NAMES)
  list(APPEND openexr_FIND_DEPENDENCY_NAMES Imath libdeflate ZLIB)
  list(REMOVE_DUPLICATES openexr_FIND_DEPENDENCY_NAMES)
else()
  set(openexr_FIND_DEPENDENCY_NAMES Imath libdeflate ZLIB)
endif()
set(Imath_FIND_MODE "NO_MODULE")
set(libdeflate_FIND_MODE "NO_MODULE")
set(ZLIB_FIND_MODE "NO_MODULE")

########### VARIABLES #######################################################################
#############################################################################################
set(openexr_PACKAGE_FOLDER_DEBUG "/Users/michalmaj/.conan2/p/b/opene025c9f70d1191/p")
set(openexr_BUILD_MODULES_PATHS_DEBUG )


set(openexr_INCLUDE_DIRS_DEBUG )
set(openexr_RES_DIRS_DEBUG )
set(openexr_DEFINITIONS_DEBUG )
set(openexr_SHARED_LINK_FLAGS_DEBUG )
set(openexr_EXE_LINK_FLAGS_DEBUG )
set(openexr_OBJECTS_DEBUG )
set(openexr_COMPILE_DEFINITIONS_DEBUG )
set(openexr_COMPILE_OPTIONS_C_DEBUG )
set(openexr_COMPILE_OPTIONS_CXX_DEBUG )
set(openexr_LIB_DIRS_DEBUG "${openexr_PACKAGE_FOLDER_DEBUG}/lib")
set(openexr_BIN_DIRS_DEBUG )
set(openexr_LIBRARY_TYPE_DEBUG STATIC)
set(openexr_IS_HOST_WINDOWS_DEBUG 0)
set(openexr_LIBS_DEBUG OpenEXRUtil-3_2_d OpenEXR-3_2_d IlmThread-3_2_d OpenEXRCore-3_2_d Iex-3_2_d)
set(openexr_SYSTEM_LIBS_DEBUG )
set(openexr_FRAMEWORK_DIRS_DEBUG )
set(openexr_FRAMEWORKS_DEBUG )
set(openexr_BUILD_DIRS_DEBUG )
set(openexr_NO_SONAME_MODE_DEBUG FALSE)


# COMPOUND VARIABLES
set(openexr_COMPILE_OPTIONS_DEBUG
    "$<$<COMPILE_LANGUAGE:CXX>:${openexr_COMPILE_OPTIONS_CXX_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:C>:${openexr_COMPILE_OPTIONS_C_DEBUG}>")
set(openexr_LINKER_FLAGS_DEBUG
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${openexr_SHARED_LINK_FLAGS_DEBUG}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${openexr_SHARED_LINK_FLAGS_DEBUG}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${openexr_EXE_LINK_FLAGS_DEBUG}>")


set(openexr_COMPONENTS_DEBUG OpenEXR::OpenEXRConfig OpenEXR::IexConfig OpenEXR::IlmThreadConfig OpenEXR::Iex OpenEXR::OpenEXRCore OpenEXR::IlmThread OpenEXR::OpenEXR OpenEXR::OpenEXRUtil)
########### COMPONENT OpenEXR::OpenEXRUtil VARIABLES ############################################

set(openexr_OpenEXR_OpenEXRUtil_INCLUDE_DIRS_DEBUG )
set(openexr_OpenEXR_OpenEXRUtil_LIB_DIRS_DEBUG "${openexr_PACKAGE_FOLDER_DEBUG}/lib")
set(openexr_OpenEXR_OpenEXRUtil_BIN_DIRS_DEBUG )
set(openexr_OpenEXR_OpenEXRUtil_LIBRARY_TYPE_DEBUG STATIC)
set(openexr_OpenEXR_OpenEXRUtil_IS_HOST_WINDOWS_DEBUG 0)
set(openexr_OpenEXR_OpenEXRUtil_RES_DIRS_DEBUG )
set(openexr_OpenEXR_OpenEXRUtil_DEFINITIONS_DEBUG )
set(openexr_OpenEXR_OpenEXRUtil_OBJECTS_DEBUG )
set(openexr_OpenEXR_OpenEXRUtil_COMPILE_DEFINITIONS_DEBUG )
set(openexr_OpenEXR_OpenEXRUtil_COMPILE_OPTIONS_C_DEBUG "")
set(openexr_OpenEXR_OpenEXRUtil_COMPILE_OPTIONS_CXX_DEBUG "")
set(openexr_OpenEXR_OpenEXRUtil_LIBS_DEBUG OpenEXRUtil-3_2_d)
set(openexr_OpenEXR_OpenEXRUtil_SYSTEM_LIBS_DEBUG )
set(openexr_OpenEXR_OpenEXRUtil_FRAMEWORK_DIRS_DEBUG )
set(openexr_OpenEXR_OpenEXRUtil_FRAMEWORKS_DEBUG )
set(openexr_OpenEXR_OpenEXRUtil_DEPENDENCIES_DEBUG OpenEXR::OpenEXR)
set(openexr_OpenEXR_OpenEXRUtil_SHARED_LINK_FLAGS_DEBUG )
set(openexr_OpenEXR_OpenEXRUtil_EXE_LINK_FLAGS_DEBUG )
set(openexr_OpenEXR_OpenEXRUtil_NO_SONAME_MODE_DEBUG FALSE)

# COMPOUND VARIABLES
set(openexr_OpenEXR_OpenEXRUtil_LINKER_FLAGS_DEBUG
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${openexr_OpenEXR_OpenEXRUtil_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${openexr_OpenEXR_OpenEXRUtil_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${openexr_OpenEXR_OpenEXRUtil_EXE_LINK_FLAGS_DEBUG}>
)
set(openexr_OpenEXR_OpenEXRUtil_COMPILE_OPTIONS_DEBUG
    "$<$<COMPILE_LANGUAGE:CXX>:${openexr_OpenEXR_OpenEXRUtil_COMPILE_OPTIONS_CXX_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:C>:${openexr_OpenEXR_OpenEXRUtil_COMPILE_OPTIONS_C_DEBUG}>")
########### COMPONENT OpenEXR::OpenEXR VARIABLES ############################################

set(openexr_OpenEXR_OpenEXR_INCLUDE_DIRS_DEBUG )
set(openexr_OpenEXR_OpenEXR_LIB_DIRS_DEBUG "${openexr_PACKAGE_FOLDER_DEBUG}/lib")
set(openexr_OpenEXR_OpenEXR_BIN_DIRS_DEBUG )
set(openexr_OpenEXR_OpenEXR_LIBRARY_TYPE_DEBUG STATIC)
set(openexr_OpenEXR_OpenEXR_IS_HOST_WINDOWS_DEBUG 0)
set(openexr_OpenEXR_OpenEXR_RES_DIRS_DEBUG )
set(openexr_OpenEXR_OpenEXR_DEFINITIONS_DEBUG )
set(openexr_OpenEXR_OpenEXR_OBJECTS_DEBUG )
set(openexr_OpenEXR_OpenEXR_COMPILE_DEFINITIONS_DEBUG )
set(openexr_OpenEXR_OpenEXR_COMPILE_OPTIONS_C_DEBUG "")
set(openexr_OpenEXR_OpenEXR_COMPILE_OPTIONS_CXX_DEBUG "")
set(openexr_OpenEXR_OpenEXR_LIBS_DEBUG OpenEXR-3_2_d)
set(openexr_OpenEXR_OpenEXR_SYSTEM_LIBS_DEBUG )
set(openexr_OpenEXR_OpenEXR_FRAMEWORK_DIRS_DEBUG )
set(openexr_OpenEXR_OpenEXR_FRAMEWORKS_DEBUG )
set(openexr_OpenEXR_OpenEXR_DEPENDENCIES_DEBUG OpenEXR::OpenEXRCore OpenEXR::IlmThread OpenEXR::Iex Imath::Imath)
set(openexr_OpenEXR_OpenEXR_SHARED_LINK_FLAGS_DEBUG )
set(openexr_OpenEXR_OpenEXR_EXE_LINK_FLAGS_DEBUG )
set(openexr_OpenEXR_OpenEXR_NO_SONAME_MODE_DEBUG FALSE)

# COMPOUND VARIABLES
set(openexr_OpenEXR_OpenEXR_LINKER_FLAGS_DEBUG
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${openexr_OpenEXR_OpenEXR_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${openexr_OpenEXR_OpenEXR_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${openexr_OpenEXR_OpenEXR_EXE_LINK_FLAGS_DEBUG}>
)
set(openexr_OpenEXR_OpenEXR_COMPILE_OPTIONS_DEBUG
    "$<$<COMPILE_LANGUAGE:CXX>:${openexr_OpenEXR_OpenEXR_COMPILE_OPTIONS_CXX_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:C>:${openexr_OpenEXR_OpenEXR_COMPILE_OPTIONS_C_DEBUG}>")
########### COMPONENT OpenEXR::IlmThread VARIABLES ############################################

set(openexr_OpenEXR_IlmThread_INCLUDE_DIRS_DEBUG )
set(openexr_OpenEXR_IlmThread_LIB_DIRS_DEBUG "${openexr_PACKAGE_FOLDER_DEBUG}/lib")
set(openexr_OpenEXR_IlmThread_BIN_DIRS_DEBUG )
set(openexr_OpenEXR_IlmThread_LIBRARY_TYPE_DEBUG STATIC)
set(openexr_OpenEXR_IlmThread_IS_HOST_WINDOWS_DEBUG 0)
set(openexr_OpenEXR_IlmThread_RES_DIRS_DEBUG )
set(openexr_OpenEXR_IlmThread_DEFINITIONS_DEBUG )
set(openexr_OpenEXR_IlmThread_OBJECTS_DEBUG )
set(openexr_OpenEXR_IlmThread_COMPILE_DEFINITIONS_DEBUG )
set(openexr_OpenEXR_IlmThread_COMPILE_OPTIONS_C_DEBUG "")
set(openexr_OpenEXR_IlmThread_COMPILE_OPTIONS_CXX_DEBUG "")
set(openexr_OpenEXR_IlmThread_LIBS_DEBUG IlmThread-3_2_d)
set(openexr_OpenEXR_IlmThread_SYSTEM_LIBS_DEBUG )
set(openexr_OpenEXR_IlmThread_FRAMEWORK_DIRS_DEBUG )
set(openexr_OpenEXR_IlmThread_FRAMEWORKS_DEBUG )
set(openexr_OpenEXR_IlmThread_DEPENDENCIES_DEBUG OpenEXR::IlmThreadConfig OpenEXR::Iex)
set(openexr_OpenEXR_IlmThread_SHARED_LINK_FLAGS_DEBUG )
set(openexr_OpenEXR_IlmThread_EXE_LINK_FLAGS_DEBUG )
set(openexr_OpenEXR_IlmThread_NO_SONAME_MODE_DEBUG FALSE)

# COMPOUND VARIABLES
set(openexr_OpenEXR_IlmThread_LINKER_FLAGS_DEBUG
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${openexr_OpenEXR_IlmThread_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${openexr_OpenEXR_IlmThread_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${openexr_OpenEXR_IlmThread_EXE_LINK_FLAGS_DEBUG}>
)
set(openexr_OpenEXR_IlmThread_COMPILE_OPTIONS_DEBUG
    "$<$<COMPILE_LANGUAGE:CXX>:${openexr_OpenEXR_IlmThread_COMPILE_OPTIONS_CXX_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:C>:${openexr_OpenEXR_IlmThread_COMPILE_OPTIONS_C_DEBUG}>")
########### COMPONENT OpenEXR::OpenEXRCore VARIABLES ############################################

set(openexr_OpenEXR_OpenEXRCore_INCLUDE_DIRS_DEBUG )
set(openexr_OpenEXR_OpenEXRCore_LIB_DIRS_DEBUG "${openexr_PACKAGE_FOLDER_DEBUG}/lib")
set(openexr_OpenEXR_OpenEXRCore_BIN_DIRS_DEBUG )
set(openexr_OpenEXR_OpenEXRCore_LIBRARY_TYPE_DEBUG STATIC)
set(openexr_OpenEXR_OpenEXRCore_IS_HOST_WINDOWS_DEBUG 0)
set(openexr_OpenEXR_OpenEXRCore_RES_DIRS_DEBUG )
set(openexr_OpenEXR_OpenEXRCore_DEFINITIONS_DEBUG )
set(openexr_OpenEXR_OpenEXRCore_OBJECTS_DEBUG )
set(openexr_OpenEXR_OpenEXRCore_COMPILE_DEFINITIONS_DEBUG )
set(openexr_OpenEXR_OpenEXRCore_COMPILE_OPTIONS_C_DEBUG "")
set(openexr_OpenEXR_OpenEXRCore_COMPILE_OPTIONS_CXX_DEBUG "")
set(openexr_OpenEXR_OpenEXRCore_LIBS_DEBUG OpenEXRCore-3_2_d)
set(openexr_OpenEXR_OpenEXRCore_SYSTEM_LIBS_DEBUG )
set(openexr_OpenEXR_OpenEXRCore_FRAMEWORK_DIRS_DEBUG )
set(openexr_OpenEXR_OpenEXRCore_FRAMEWORKS_DEBUG )
set(openexr_OpenEXR_OpenEXRCore_DEPENDENCIES_DEBUG OpenEXR::OpenEXRConfig ZLIB::ZLIB libdeflate::libdeflate_static)
set(openexr_OpenEXR_OpenEXRCore_SHARED_LINK_FLAGS_DEBUG )
set(openexr_OpenEXR_OpenEXRCore_EXE_LINK_FLAGS_DEBUG )
set(openexr_OpenEXR_OpenEXRCore_NO_SONAME_MODE_DEBUG FALSE)

# COMPOUND VARIABLES
set(openexr_OpenEXR_OpenEXRCore_LINKER_FLAGS_DEBUG
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${openexr_OpenEXR_OpenEXRCore_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${openexr_OpenEXR_OpenEXRCore_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${openexr_OpenEXR_OpenEXRCore_EXE_LINK_FLAGS_DEBUG}>
)
set(openexr_OpenEXR_OpenEXRCore_COMPILE_OPTIONS_DEBUG
    "$<$<COMPILE_LANGUAGE:CXX>:${openexr_OpenEXR_OpenEXRCore_COMPILE_OPTIONS_CXX_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:C>:${openexr_OpenEXR_OpenEXRCore_COMPILE_OPTIONS_C_DEBUG}>")
########### COMPONENT OpenEXR::Iex VARIABLES ############################################

set(openexr_OpenEXR_Iex_INCLUDE_DIRS_DEBUG )
set(openexr_OpenEXR_Iex_LIB_DIRS_DEBUG "${openexr_PACKAGE_FOLDER_DEBUG}/lib")
set(openexr_OpenEXR_Iex_BIN_DIRS_DEBUG )
set(openexr_OpenEXR_Iex_LIBRARY_TYPE_DEBUG STATIC)
set(openexr_OpenEXR_Iex_IS_HOST_WINDOWS_DEBUG 0)
set(openexr_OpenEXR_Iex_RES_DIRS_DEBUG )
set(openexr_OpenEXR_Iex_DEFINITIONS_DEBUG )
set(openexr_OpenEXR_Iex_OBJECTS_DEBUG )
set(openexr_OpenEXR_Iex_COMPILE_DEFINITIONS_DEBUG )
set(openexr_OpenEXR_Iex_COMPILE_OPTIONS_C_DEBUG "")
set(openexr_OpenEXR_Iex_COMPILE_OPTIONS_CXX_DEBUG "")
set(openexr_OpenEXR_Iex_LIBS_DEBUG Iex-3_2_d)
set(openexr_OpenEXR_Iex_SYSTEM_LIBS_DEBUG )
set(openexr_OpenEXR_Iex_FRAMEWORK_DIRS_DEBUG )
set(openexr_OpenEXR_Iex_FRAMEWORKS_DEBUG )
set(openexr_OpenEXR_Iex_DEPENDENCIES_DEBUG OpenEXR::IexConfig)
set(openexr_OpenEXR_Iex_SHARED_LINK_FLAGS_DEBUG )
set(openexr_OpenEXR_Iex_EXE_LINK_FLAGS_DEBUG )
set(openexr_OpenEXR_Iex_NO_SONAME_MODE_DEBUG FALSE)

# COMPOUND VARIABLES
set(openexr_OpenEXR_Iex_LINKER_FLAGS_DEBUG
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${openexr_OpenEXR_Iex_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${openexr_OpenEXR_Iex_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${openexr_OpenEXR_Iex_EXE_LINK_FLAGS_DEBUG}>
)
set(openexr_OpenEXR_Iex_COMPILE_OPTIONS_DEBUG
    "$<$<COMPILE_LANGUAGE:CXX>:${openexr_OpenEXR_Iex_COMPILE_OPTIONS_CXX_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:C>:${openexr_OpenEXR_Iex_COMPILE_OPTIONS_C_DEBUG}>")
########### COMPONENT OpenEXR::IlmThreadConfig VARIABLES ############################################

set(openexr_OpenEXR_IlmThreadConfig_INCLUDE_DIRS_DEBUG )
set(openexr_OpenEXR_IlmThreadConfig_LIB_DIRS_DEBUG "${openexr_PACKAGE_FOLDER_DEBUG}/lib")
set(openexr_OpenEXR_IlmThreadConfig_BIN_DIRS_DEBUG )
set(openexr_OpenEXR_IlmThreadConfig_LIBRARY_TYPE_DEBUG STATIC)
set(openexr_OpenEXR_IlmThreadConfig_IS_HOST_WINDOWS_DEBUG 0)
set(openexr_OpenEXR_IlmThreadConfig_RES_DIRS_DEBUG )
set(openexr_OpenEXR_IlmThreadConfig_DEFINITIONS_DEBUG )
set(openexr_OpenEXR_IlmThreadConfig_OBJECTS_DEBUG )
set(openexr_OpenEXR_IlmThreadConfig_COMPILE_DEFINITIONS_DEBUG )
set(openexr_OpenEXR_IlmThreadConfig_COMPILE_OPTIONS_C_DEBUG "")
set(openexr_OpenEXR_IlmThreadConfig_COMPILE_OPTIONS_CXX_DEBUG "")
set(openexr_OpenEXR_IlmThreadConfig_LIBS_DEBUG )
set(openexr_OpenEXR_IlmThreadConfig_SYSTEM_LIBS_DEBUG )
set(openexr_OpenEXR_IlmThreadConfig_FRAMEWORK_DIRS_DEBUG )
set(openexr_OpenEXR_IlmThreadConfig_FRAMEWORKS_DEBUG )
set(openexr_OpenEXR_IlmThreadConfig_DEPENDENCIES_DEBUG )
set(openexr_OpenEXR_IlmThreadConfig_SHARED_LINK_FLAGS_DEBUG )
set(openexr_OpenEXR_IlmThreadConfig_EXE_LINK_FLAGS_DEBUG )
set(openexr_OpenEXR_IlmThreadConfig_NO_SONAME_MODE_DEBUG FALSE)

# COMPOUND VARIABLES
set(openexr_OpenEXR_IlmThreadConfig_LINKER_FLAGS_DEBUG
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${openexr_OpenEXR_IlmThreadConfig_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${openexr_OpenEXR_IlmThreadConfig_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${openexr_OpenEXR_IlmThreadConfig_EXE_LINK_FLAGS_DEBUG}>
)
set(openexr_OpenEXR_IlmThreadConfig_COMPILE_OPTIONS_DEBUG
    "$<$<COMPILE_LANGUAGE:CXX>:${openexr_OpenEXR_IlmThreadConfig_COMPILE_OPTIONS_CXX_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:C>:${openexr_OpenEXR_IlmThreadConfig_COMPILE_OPTIONS_C_DEBUG}>")
########### COMPONENT OpenEXR::IexConfig VARIABLES ############################################

set(openexr_OpenEXR_IexConfig_INCLUDE_DIRS_DEBUG )
set(openexr_OpenEXR_IexConfig_LIB_DIRS_DEBUG "${openexr_PACKAGE_FOLDER_DEBUG}/lib")
set(openexr_OpenEXR_IexConfig_BIN_DIRS_DEBUG )
set(openexr_OpenEXR_IexConfig_LIBRARY_TYPE_DEBUG STATIC)
set(openexr_OpenEXR_IexConfig_IS_HOST_WINDOWS_DEBUG 0)
set(openexr_OpenEXR_IexConfig_RES_DIRS_DEBUG )
set(openexr_OpenEXR_IexConfig_DEFINITIONS_DEBUG )
set(openexr_OpenEXR_IexConfig_OBJECTS_DEBUG )
set(openexr_OpenEXR_IexConfig_COMPILE_DEFINITIONS_DEBUG )
set(openexr_OpenEXR_IexConfig_COMPILE_OPTIONS_C_DEBUG "")
set(openexr_OpenEXR_IexConfig_COMPILE_OPTIONS_CXX_DEBUG "")
set(openexr_OpenEXR_IexConfig_LIBS_DEBUG )
set(openexr_OpenEXR_IexConfig_SYSTEM_LIBS_DEBUG )
set(openexr_OpenEXR_IexConfig_FRAMEWORK_DIRS_DEBUG )
set(openexr_OpenEXR_IexConfig_FRAMEWORKS_DEBUG )
set(openexr_OpenEXR_IexConfig_DEPENDENCIES_DEBUG )
set(openexr_OpenEXR_IexConfig_SHARED_LINK_FLAGS_DEBUG )
set(openexr_OpenEXR_IexConfig_EXE_LINK_FLAGS_DEBUG )
set(openexr_OpenEXR_IexConfig_NO_SONAME_MODE_DEBUG FALSE)

# COMPOUND VARIABLES
set(openexr_OpenEXR_IexConfig_LINKER_FLAGS_DEBUG
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${openexr_OpenEXR_IexConfig_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${openexr_OpenEXR_IexConfig_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${openexr_OpenEXR_IexConfig_EXE_LINK_FLAGS_DEBUG}>
)
set(openexr_OpenEXR_IexConfig_COMPILE_OPTIONS_DEBUG
    "$<$<COMPILE_LANGUAGE:CXX>:${openexr_OpenEXR_IexConfig_COMPILE_OPTIONS_CXX_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:C>:${openexr_OpenEXR_IexConfig_COMPILE_OPTIONS_C_DEBUG}>")
########### COMPONENT OpenEXR::OpenEXRConfig VARIABLES ############################################

set(openexr_OpenEXR_OpenEXRConfig_INCLUDE_DIRS_DEBUG )
set(openexr_OpenEXR_OpenEXRConfig_LIB_DIRS_DEBUG "${openexr_PACKAGE_FOLDER_DEBUG}/lib")
set(openexr_OpenEXR_OpenEXRConfig_BIN_DIRS_DEBUG )
set(openexr_OpenEXR_OpenEXRConfig_LIBRARY_TYPE_DEBUG STATIC)
set(openexr_OpenEXR_OpenEXRConfig_IS_HOST_WINDOWS_DEBUG 0)
set(openexr_OpenEXR_OpenEXRConfig_RES_DIRS_DEBUG )
set(openexr_OpenEXR_OpenEXRConfig_DEFINITIONS_DEBUG )
set(openexr_OpenEXR_OpenEXRConfig_OBJECTS_DEBUG )
set(openexr_OpenEXR_OpenEXRConfig_COMPILE_DEFINITIONS_DEBUG )
set(openexr_OpenEXR_OpenEXRConfig_COMPILE_OPTIONS_C_DEBUG "")
set(openexr_OpenEXR_OpenEXRConfig_COMPILE_OPTIONS_CXX_DEBUG "")
set(openexr_OpenEXR_OpenEXRConfig_LIBS_DEBUG )
set(openexr_OpenEXR_OpenEXRConfig_SYSTEM_LIBS_DEBUG )
set(openexr_OpenEXR_OpenEXRConfig_FRAMEWORK_DIRS_DEBUG )
set(openexr_OpenEXR_OpenEXRConfig_FRAMEWORKS_DEBUG )
set(openexr_OpenEXR_OpenEXRConfig_DEPENDENCIES_DEBUG )
set(openexr_OpenEXR_OpenEXRConfig_SHARED_LINK_FLAGS_DEBUG )
set(openexr_OpenEXR_OpenEXRConfig_EXE_LINK_FLAGS_DEBUG )
set(openexr_OpenEXR_OpenEXRConfig_NO_SONAME_MODE_DEBUG FALSE)

# COMPOUND VARIABLES
set(openexr_OpenEXR_OpenEXRConfig_LINKER_FLAGS_DEBUG
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${openexr_OpenEXR_OpenEXRConfig_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${openexr_OpenEXR_OpenEXRConfig_SHARED_LINK_FLAGS_DEBUG}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${openexr_OpenEXR_OpenEXRConfig_EXE_LINK_FLAGS_DEBUG}>
)
set(openexr_OpenEXR_OpenEXRConfig_COMPILE_OPTIONS_DEBUG
    "$<$<COMPILE_LANGUAGE:CXX>:${openexr_OpenEXR_OpenEXRConfig_COMPILE_OPTIONS_CXX_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:C>:${openexr_OpenEXR_OpenEXRConfig_COMPILE_OPTIONS_C_DEBUG}>")