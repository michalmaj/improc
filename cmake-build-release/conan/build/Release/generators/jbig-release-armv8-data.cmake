########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

set(jbig_COMPONENT_NAMES "")
if(DEFINED jbig_FIND_DEPENDENCY_NAMES)
  list(APPEND jbig_FIND_DEPENDENCY_NAMES )
  list(REMOVE_DUPLICATES jbig_FIND_DEPENDENCY_NAMES)
else()
  set(jbig_FIND_DEPENDENCY_NAMES )
endif()

########### VARIABLES #######################################################################
#############################################################################################
set(jbig_PACKAGE_FOLDER_RELEASE "/Users/michalmaj/.conan2/p/b/jbig9b08dbdbc0f2e/p")
set(jbig_BUILD_MODULES_PATHS_RELEASE )


set(jbig_INCLUDE_DIRS_RELEASE )
set(jbig_RES_DIRS_RELEASE )
set(jbig_DEFINITIONS_RELEASE )
set(jbig_SHARED_LINK_FLAGS_RELEASE )
set(jbig_EXE_LINK_FLAGS_RELEASE )
set(jbig_OBJECTS_RELEASE )
set(jbig_COMPILE_DEFINITIONS_RELEASE )
set(jbig_COMPILE_OPTIONS_C_RELEASE )
set(jbig_COMPILE_OPTIONS_CXX_RELEASE )
set(jbig_LIB_DIRS_RELEASE "${jbig_PACKAGE_FOLDER_RELEASE}/lib")
set(jbig_BIN_DIRS_RELEASE )
set(jbig_LIBRARY_TYPE_RELEASE STATIC)
set(jbig_IS_HOST_WINDOWS_RELEASE 0)
set(jbig_LIBS_RELEASE jbig)
set(jbig_SYSTEM_LIBS_RELEASE )
set(jbig_FRAMEWORK_DIRS_RELEASE )
set(jbig_FRAMEWORKS_RELEASE )
set(jbig_BUILD_DIRS_RELEASE )
set(jbig_NO_SONAME_MODE_RELEASE FALSE)


# COMPOUND VARIABLES
set(jbig_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${jbig_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${jbig_COMPILE_OPTIONS_C_RELEASE}>")
set(jbig_LINKER_FLAGS_RELEASE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${jbig_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${jbig_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${jbig_EXE_LINK_FLAGS_RELEASE}>")


set(jbig_COMPONENTS_RELEASE )