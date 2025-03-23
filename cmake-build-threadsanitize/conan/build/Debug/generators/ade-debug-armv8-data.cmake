########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

set(ade_COMPONENT_NAMES "")
if(DEFINED ade_FIND_DEPENDENCY_NAMES)
  list(APPEND ade_FIND_DEPENDENCY_NAMES )
  list(REMOVE_DUPLICATES ade_FIND_DEPENDENCY_NAMES)
else()
  set(ade_FIND_DEPENDENCY_NAMES )
endif()

########### VARIABLES #######################################################################
#############################################################################################
set(ade_PACKAGE_FOLDER_DEBUG "/Users/michalmaj/.conan2/p/b/aded89f0a8d410ce/p")
set(ade_BUILD_MODULES_PATHS_DEBUG )


set(ade_INCLUDE_DIRS_DEBUG )
set(ade_RES_DIRS_DEBUG )
set(ade_DEFINITIONS_DEBUG )
set(ade_SHARED_LINK_FLAGS_DEBUG )
set(ade_EXE_LINK_FLAGS_DEBUG )
set(ade_OBJECTS_DEBUG )
set(ade_COMPILE_DEFINITIONS_DEBUG )
set(ade_COMPILE_OPTIONS_C_DEBUG )
set(ade_COMPILE_OPTIONS_CXX_DEBUG )
set(ade_LIB_DIRS_DEBUG "${ade_PACKAGE_FOLDER_DEBUG}/lib")
set(ade_BIN_DIRS_DEBUG )
set(ade_LIBRARY_TYPE_DEBUG STATIC)
set(ade_IS_HOST_WINDOWS_DEBUG 0)
set(ade_LIBS_DEBUG ade)
set(ade_SYSTEM_LIBS_DEBUG )
set(ade_FRAMEWORK_DIRS_DEBUG )
set(ade_FRAMEWORKS_DEBUG )
set(ade_BUILD_DIRS_DEBUG )
set(ade_NO_SONAME_MODE_DEBUG FALSE)


# COMPOUND VARIABLES
set(ade_COMPILE_OPTIONS_DEBUG
    "$<$<COMPILE_LANGUAGE:CXX>:${ade_COMPILE_OPTIONS_CXX_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:C>:${ade_COMPILE_OPTIONS_C_DEBUG}>")
set(ade_LINKER_FLAGS_DEBUG
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${ade_SHARED_LINK_FLAGS_DEBUG}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${ade_SHARED_LINK_FLAGS_DEBUG}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${ade_EXE_LINK_FLAGS_DEBUG}>")


set(ade_COMPONENTS_DEBUG )