########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

list(APPEND ogg_COMPONENT_NAMES Ogg::ogg)
list(REMOVE_DUPLICATES ogg_COMPONENT_NAMES)
if(DEFINED ogg_FIND_DEPENDENCY_NAMES)
  list(APPEND ogg_FIND_DEPENDENCY_NAMES )
  list(REMOVE_DUPLICATES ogg_FIND_DEPENDENCY_NAMES)
else()
  set(ogg_FIND_DEPENDENCY_NAMES )
endif()

########### VARIABLES #######################################################################
#############################################################################################
set(ogg_PACKAGE_FOLDER_RELEASE "/Users/michalmaj/.conan2/p/b/ogga9eb13d871b96/p")
set(ogg_BUILD_MODULES_PATHS_RELEASE )


set(ogg_INCLUDE_DIRS_RELEASE )
set(ogg_RES_DIRS_RELEASE )
set(ogg_DEFINITIONS_RELEASE )
set(ogg_SHARED_LINK_FLAGS_RELEASE )
set(ogg_EXE_LINK_FLAGS_RELEASE )
set(ogg_OBJECTS_RELEASE )
set(ogg_COMPILE_DEFINITIONS_RELEASE )
set(ogg_COMPILE_OPTIONS_C_RELEASE )
set(ogg_COMPILE_OPTIONS_CXX_RELEASE )
set(ogg_LIB_DIRS_RELEASE "${ogg_PACKAGE_FOLDER_RELEASE}/lib")
set(ogg_BIN_DIRS_RELEASE )
set(ogg_LIBRARY_TYPE_RELEASE STATIC)
set(ogg_IS_HOST_WINDOWS_RELEASE 0)
set(ogg_LIBS_RELEASE ogg)
set(ogg_SYSTEM_LIBS_RELEASE )
set(ogg_FRAMEWORK_DIRS_RELEASE )
set(ogg_FRAMEWORKS_RELEASE )
set(ogg_BUILD_DIRS_RELEASE )
set(ogg_NO_SONAME_MODE_RELEASE FALSE)


# COMPOUND VARIABLES
set(ogg_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${ogg_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${ogg_COMPILE_OPTIONS_C_RELEASE}>")
set(ogg_LINKER_FLAGS_RELEASE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${ogg_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${ogg_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${ogg_EXE_LINK_FLAGS_RELEASE}>")


set(ogg_COMPONENTS_RELEASE Ogg::ogg)
########### COMPONENT Ogg::ogg VARIABLES ############################################

set(ogg_Ogg_ogg_INCLUDE_DIRS_RELEASE )
set(ogg_Ogg_ogg_LIB_DIRS_RELEASE "${ogg_PACKAGE_FOLDER_RELEASE}/lib")
set(ogg_Ogg_ogg_BIN_DIRS_RELEASE )
set(ogg_Ogg_ogg_LIBRARY_TYPE_RELEASE STATIC)
set(ogg_Ogg_ogg_IS_HOST_WINDOWS_RELEASE 0)
set(ogg_Ogg_ogg_RES_DIRS_RELEASE )
set(ogg_Ogg_ogg_DEFINITIONS_RELEASE )
set(ogg_Ogg_ogg_OBJECTS_RELEASE )
set(ogg_Ogg_ogg_COMPILE_DEFINITIONS_RELEASE )
set(ogg_Ogg_ogg_COMPILE_OPTIONS_C_RELEASE "")
set(ogg_Ogg_ogg_COMPILE_OPTIONS_CXX_RELEASE "")
set(ogg_Ogg_ogg_LIBS_RELEASE ogg)
set(ogg_Ogg_ogg_SYSTEM_LIBS_RELEASE )
set(ogg_Ogg_ogg_FRAMEWORK_DIRS_RELEASE )
set(ogg_Ogg_ogg_FRAMEWORKS_RELEASE )
set(ogg_Ogg_ogg_DEPENDENCIES_RELEASE )
set(ogg_Ogg_ogg_SHARED_LINK_FLAGS_RELEASE )
set(ogg_Ogg_ogg_EXE_LINK_FLAGS_RELEASE )
set(ogg_Ogg_ogg_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(ogg_Ogg_ogg_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${ogg_Ogg_ogg_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${ogg_Ogg_ogg_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${ogg_Ogg_ogg_EXE_LINK_FLAGS_RELEASE}>
)
set(ogg_Ogg_ogg_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${ogg_Ogg_ogg_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${ogg_Ogg_ogg_COMPILE_OPTIONS_C_RELEASE}>")