# - Find elfutils
# Find elfutils libraries
#
# This module defines the following variables:#
#   ELFUTILS_FOUND        : true if everything is found
#   ELFUTILS_INCLUDE_DIR  : where to find dw.h, etc.
#   ELFUTILS_LIBRARIES    : the library to link against.

FIND_PATH(ELFUTILS_INCLUDE_DIR elfutils/version.h
    $ENV{Elfutils_ROOT_DIR})


# DWARF library

FIND_LIBRARY(DWARF_LIBRARY 
  NAMES dw 
  PATHS /usr/lib
        /usr/$ENV{MACHTYPE}/lib)

# Finish up

SET(ELFUTILS_LIBRARIES DWARF_LIBRARY)

MARK_AS_ADVANCED(ELFUTILS_INCLUDE_DIR ELFUTILS_LIBRARIES)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Elfutils DEFAULT_MSG ELFUTILS_INCLUDE_DIR ELFUTILS_LIBRARIES)
