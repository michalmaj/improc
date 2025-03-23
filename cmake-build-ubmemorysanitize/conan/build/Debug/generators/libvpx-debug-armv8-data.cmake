########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

set(libvpx_COMPONENT_NAMES "")
if(DEFINED libvpx_FIND_DEPENDENCY_NAMES)
  list(APPEND libvpx_FIND_DEPENDENCY_NAMES )
  list(REMOVE_DUPLICATES libvpx_FIND_DEPENDENCY_NAMES)
else()
  set(libvpx_FIND_DEPENDENCY_NAMES )
endif()

########### VARIABLES #######################################################################
#############################################################################################
set(libvpx_PACKAGE_FOLDER_DEBUG "/Users/michalmaj/.conan2/p/b/libvpafd42c3a4fae7/p")
set(libvpx_BUILD_MODULES_PATHS_DEBUG )


set(libvpx_INCLUDE_DIRS_DEBUG )
set(libvpx_RES_DIRS_DEBUG )
set(libvpx_DEFINITIONS_DEBUG )
set(libvpx_SHARED_LINK_FLAGS_DEBUG )
set(libvpx_EXE_LINK_FLAGS_DEBUG )
set(libvpx_OBJECTS_DEBUG )
set(libvpx_COMPILE_DEFINITIONS_DEBUG )
set(libvpx_COMPILE_OPTIONS_C_DEBUG )
set(libvpx_COMPILE_OPTIONS_CXX_DEBUG )
set(libvpx_LIB_DIRS_DEBUG "${libvpx_PACKAGE_FOLDER_DEBUG}/lib")
set(libvpx_BIN_DIRS_DEBUG )
set(libvpx_LIBRARY_TYPE_DEBUG STATIC)
set(libvpx_IS_HOST_WINDOWS_DEBUG 0)
set(libvpx_LIBS_DEBUG vpx)
set(libvpx_SYSTEM_LIBS_DEBUG c++)
set(libvpx_FRAMEWORK_DIRS_DEBUG )
set(libvpx_FRAMEWORKS_DEBUG )
set(libvpx_BUILD_DIRS_DEBUG )
set(libvpx_NO_SONAME_MODE_DEBUG FALSE)


# COMPOUND VARIABLES
set(libvpx_COMPILE_OPTIONS_DEBUG
    "$<$<COMPILE_LANGUAGE:CXX>:${libvpx_COMPILE_OPTIONS_CXX_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:C>:${libvpx_COMPILE_OPTIONS_C_DEBUG}>")
set(libvpx_LINKER_FLAGS_DEBUG
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${libvpx_SHARED_LINK_FLAGS_DEBUG}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${libvpx_SHARED_LINK_FLAGS_DEBUG}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${libvpx_EXE_LINK_FLAGS_DEBUG}>")


set(libvpx_COMPONENTS_DEBUG )