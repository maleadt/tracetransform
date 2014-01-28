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
          PATHS ${ArrayFire_ROOT_DIR}
                $ENV{AF_PATH}
                /usr
                /usr/local
                /opt/arrayfire)

get_filename_component(ArrayFire_ACTUAL_ROOT_DIR ${ArrayFire_INCLUDE_DIR} DIRECTORY)

# 1.9
FIND_LIBRARY(ArrayFire_LIBRARY19
             NAMES "af"
             PATH_SUFFIXES "lib64"
             PATHS ${ArrayFire_ACTUAL_ROOT_DIR}
                   /usr
                   /usr/local
                   /opt/arrayfire)

# 2.0
FIND_LIBRARY(ArrayFire_LIBRARY20
             NAMES "afcu"
             PATH_SUFFIXES "lib64"
             PATHS ${ArrayFire_ACTUAL_ROOT_DIR}
                   /usr
                   /usr/local
                   /opt/arrayfire)

if (ArrayFire_LIBRARY19)
  list(APPEND ArrayFire_LIBRARIES ${ArrayFire_LIBRARY19})
elseif (ArrayFire_LIBRARY20)
  list(APPEND ArrayFire_LIBRARIES ${ArrayFire_LIBRARY20})
endif()


# GFX library

FIND_LIBRARY(ArrayFire_GFX_LIBRARY
             NAMES "afGFX"
             PATH_SUFFIXES "lib64"
             PATHS ${ArrayFire_ACTUAL_ROOT_DIR}
                   /usr
                   /usr/local
                   /opt/arrayfire)

if (ArrayFire_GFX_LIBRARY)
  list(APPEND ArrayFire_LIBRARIES ${ArrayFire_GFX_LIBRARY})
endif (ArrayFire_GFX_LIBRARY)


# Finish up

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS("ArrayFire" DEFAULT_MSG ArrayFire_INCLUDE_DIR ArrayFire_LIBRARIES)

MARK_AS_ADVANCED(ArrayFire_INCLUDE_DIR ArrayFire_LIBRARIES)
