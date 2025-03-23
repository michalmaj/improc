########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

set(quirc_COMPONENT_NAMES "")
if(DEFINED quirc_FIND_DEPENDENCY_NAMES)
  list(APPEND quirc_FIND_DEPENDENCY_NAMES )
  list(REMOVE_DUPLICATES quirc_FIND_DEPENDENCY_NAMES)
else()
  set(quirc_FIND_DEPENDENCY_NAMES )
endif()

########### VARIABLES #######################################################################
#############################################################################################
set(quirc_PACKAGE_FOLDER_DEBUG "/Users/michalmaj/.conan2/p/b/quirc275dbee08385e/p")
set(quirc_BUILD_MODULES_PATHS_DEBUG )


set(quirc_INCLUDE_DIRS_DEBUG )
set(quirc_RES_DIRS_DEBUG )
set(quirc_DEFINITIONS_DEBUG )
set(quirc_SHARED_LINK_FLAGS_DEBUG )
set(quirc_EXE_LINK_FLAGS_DEBUG )
set(quirc_OBJECTS_DEBUG )
set(quirc_COMPILE_DEFINITIONS_DEBUG )
set(quirc_COMPILE_OPTIONS_C_DEBUG )
set(quirc_COMPILE_OPTIONS_CXX_DEBUG )
set(quirc_LIB_DIRS_DEBUG "${quirc_PACKAGE_FOLDER_DEBUG}/lib")
set(quirc_BIN_DIRS_DEBUG )
set(quirc_LIBRARY_TYPE_DEBUG STATIC)
set(quirc_IS_HOST_WINDOWS_DEBUG 0)
set(quirc_LIBS_DEBUG quirc)
set(quirc_SYSTEM_LIBS_DEBUG )
set(quirc_FRAMEWORK_DIRS_DEBUG )
set(quirc_FRAMEWORKS_DEBUG )
set(quirc_BUILD_DIRS_DEBUG )
set(quirc_NO_SONAME_MODE_DEBUG FALSE)


# COMPOUND VARIABLES
set(quirc_COMPILE_OPTIONS_DEBUG
    "$<$<COMPILE_LANGUAGE:CXX>:${quirc_COMPILE_OPTIONS_CXX_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:C>:${quirc_COMPILE_OPTIONS_C_DEBUG}>")
set(quirc_LINKER_FLAGS_DEBUG
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${quirc_SHARED_LINK_FLAGS_DEBUG}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${quirc_SHARED_LINK_FLAGS_DEBUG}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${quirc_EXE_LINK_FLAGS_DEBUG}>")


set(quirc_COMPONENTS_DEBUG )