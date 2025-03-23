########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

set(openjpeg_COMPONENT_NAMES "")
if(DEFINED openjpeg_FIND_DEPENDENCY_NAMES)
  list(APPEND openjpeg_FIND_DEPENDENCY_NAMES )
  list(REMOVE_DUPLICATES openjpeg_FIND_DEPENDENCY_NAMES)
else()
  set(openjpeg_FIND_DEPENDENCY_NAMES )
endif()

########### VARIABLES #######################################################################
#############################################################################################
set(openjpeg_PACKAGE_FOLDER_DEBUG "/Users/michalmaj/.conan2/p/b/openj590fc4ad4b2c0/p")
set(openjpeg_BUILD_MODULES_PATHS_DEBUG "${openjpeg_PACKAGE_FOLDER_DEBUG}/lib/cmake/conan-official-openjpeg-variables.cmake")


set(openjpeg_INCLUDE_DIRS_DEBUG )
set(openjpeg_RES_DIRS_DEBUG )
set(openjpeg_DEFINITIONS_DEBUG )
set(openjpeg_SHARED_LINK_FLAGS_DEBUG )
set(openjpeg_EXE_LINK_FLAGS_DEBUG )
set(openjpeg_OBJECTS_DEBUG )
set(openjpeg_COMPILE_DEFINITIONS_DEBUG )
set(openjpeg_COMPILE_OPTIONS_C_DEBUG )
set(openjpeg_COMPILE_OPTIONS_CXX_DEBUG )
set(openjpeg_LIB_DIRS_DEBUG "${openjpeg_PACKAGE_FOLDER_DEBUG}/lib")
set(openjpeg_BIN_DIRS_DEBUG )
set(openjpeg_LIBRARY_TYPE_DEBUG STATIC)
set(openjpeg_IS_HOST_WINDOWS_DEBUG 0)
set(openjpeg_LIBS_DEBUG openjp2)
set(openjpeg_SYSTEM_LIBS_DEBUG )
set(openjpeg_FRAMEWORK_DIRS_DEBUG )
set(openjpeg_FRAMEWORKS_DEBUG )
set(openjpeg_BUILD_DIRS_DEBUG "${openjpeg_PACKAGE_FOLDER_DEBUG}/lib/cmake")
set(openjpeg_NO_SONAME_MODE_DEBUG FALSE)


# COMPOUND VARIABLES
set(openjpeg_COMPILE_OPTIONS_DEBUG
    "$<$<COMPILE_LANGUAGE:CXX>:${openjpeg_COMPILE_OPTIONS_CXX_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:C>:${openjpeg_COMPILE_OPTIONS_C_DEBUG}>")
set(openjpeg_LINKER_FLAGS_DEBUG
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${openjpeg_SHARED_LINK_FLAGS_DEBUG}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${openjpeg_SHARED_LINK_FLAGS_DEBUG}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${openjpeg_EXE_LINK_FLAGS_DEBUG}>")


set(openjpeg_COMPONENTS_DEBUG )