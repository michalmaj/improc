########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

set(openh264_COMPONENT_NAMES "")
if(DEFINED openh264_FIND_DEPENDENCY_NAMES)
  list(APPEND openh264_FIND_DEPENDENCY_NAMES )
  list(REMOVE_DUPLICATES openh264_FIND_DEPENDENCY_NAMES)
else()
  set(openh264_FIND_DEPENDENCY_NAMES )
endif()

########### VARIABLES #######################################################################
#############################################################################################
set(openh264_PACKAGE_FOLDER_DEBUG "/Users/michalmaj/.conan2/p/b/openh1378742f09d05/p")
set(openh264_BUILD_MODULES_PATHS_DEBUG )


set(openh264_INCLUDE_DIRS_DEBUG )
set(openh264_RES_DIRS_DEBUG )
set(openh264_DEFINITIONS_DEBUG )
set(openh264_SHARED_LINK_FLAGS_DEBUG )
set(openh264_EXE_LINK_FLAGS_DEBUG )
set(openh264_OBJECTS_DEBUG )
set(openh264_COMPILE_DEFINITIONS_DEBUG )
set(openh264_COMPILE_OPTIONS_C_DEBUG )
set(openh264_COMPILE_OPTIONS_CXX_DEBUG )
set(openh264_LIB_DIRS_DEBUG "${openh264_PACKAGE_FOLDER_DEBUG}/lib")
set(openh264_BIN_DIRS_DEBUG )
set(openh264_LIBRARY_TYPE_DEBUG STATIC)
set(openh264_IS_HOST_WINDOWS_DEBUG 0)
set(openh264_LIBS_DEBUG openh264)
set(openh264_SYSTEM_LIBS_DEBUG c++)
set(openh264_FRAMEWORK_DIRS_DEBUG )
set(openh264_FRAMEWORKS_DEBUG )
set(openh264_BUILD_DIRS_DEBUG )
set(openh264_NO_SONAME_MODE_DEBUG FALSE)


# COMPOUND VARIABLES
set(openh264_COMPILE_OPTIONS_DEBUG
    "$<$<COMPILE_LANGUAGE:CXX>:${openh264_COMPILE_OPTIONS_CXX_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:C>:${openh264_COMPILE_OPTIONS_C_DEBUG}>")
set(openh264_LINKER_FLAGS_DEBUG
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${openh264_SHARED_LINK_FLAGS_DEBUG}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${openh264_SHARED_LINK_FLAGS_DEBUG}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${openh264_EXE_LINK_FLAGS_DEBUG}>")


set(openh264_COMPONENTS_DEBUG )