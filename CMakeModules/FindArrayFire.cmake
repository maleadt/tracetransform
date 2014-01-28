# - Find ArrayFire
# This module finds the native ArrayFire includes and library
#
# Defined variables:
#   ArrayFire_FOUND       - true if ArrayFire found.
#   ArrayFire_INCLUDE_DIR - where to find arrayfire.h, etc.
#   ArrayFire_LIBRARIES   - list of libraries.
#

# Main library

FIND_PATH(ArrayFire_INCLUDE_DIR "arrayfire.h"
          PATH_SUFFIXES "include"
          PATHS $ENV{AF_PATH}
                /usr
                /usr/local)

FIND_LIBRARY(ArrayFire_LIBRARY
             NAMES "afcu"
             PATH_SUFFIXES "lib64"
             PATHS $ENV{AF_PATH}
                   /usr
                   usr/local)

if (ArrayFire_LIBRARY)
  list(APPEND ArrayFire_LIBRARIES ${ArrayFire_LIBRARY})
endif (ArrayFire_LIBRARY)


# GFX library

FIND_LIBRARY(ArrayFire_GFX_LIBRARY
             NAMES "afGFX"
             PATH_SUFFIXES "lib64"
             PATHS /usr/ usr/local /opt $ENV{AF_PATH})

if (ArrayFire_GFX_LIBRARY)
  list(APPEND ArrayFire_LIBRARIES ${ArrayFire_GFX_LIBRARY})
endif (ArrayFire_GFX_LIBRARY)


# Finish up

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS("ArrayFire" DEFAULT_MSG ArrayFire_INCLUDE_DIR ArrayFire_LIBRARIES)

MARK_AS_ADVANCED(ArrayFire_INCLUDE_DIR ArrayFire_LIBRARIES)
