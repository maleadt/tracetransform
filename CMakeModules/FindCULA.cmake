# - Find CULA
# This module finds the native CULA includes and library
#
# Defined variables:
#   CULA_FOUND       : true if CULA found.
#   CULA_INCLUDE_DIR : where to find cula.h, etc.
#   CULA_LIBRARIES   : list of libraries.

FIND_PATH(CULA_INCLUDE_DIR "cula.h"
          PATH_SUFFIXES "include"
          PATHS /usr
                /usr/local
                /opt/cula)

 
# R12

FIND_LIBRARY(CULA_LIBRARY
             NAMES "cula"
             PATH_SUFFIXES "lib64"
             PATHS /usr
                   /usr/local
                   /opt/cula)

if (CULA_LIBRARY)
  list(APPEND CULA_LIBRARIES ${CULA_LIBRARY})
endif (CULA_LIBRARY)


# R13

FIND_LIBRARY(CULA_LAPACK_LIBRARY
             NAMES "cula_lapack"
             PATH_SUFFIXES "lib64"
             PATHS /usr
                   /usr/local
                   /opt/cula)

FIND_LIBRARY(CULA_CORE_LIBRARY
             NAMES "cula_core"
             PATH_SUFFIXES "lib64"
             PATHS /usr
                   /usr/local
                   /opt/cula)

if (CULA_LAPACK_LIBRARY)
  list(APPEND CULA_LIBRARIES ${CULA_LAPACK_LIBRARY})
endif (CULA_LAPACK_LIBRARY)
if (CULA_CORE_LIBRARY)
  list(APPEND CULA_LIBRARIES ${CULA_CORE_LIBRARY})
endif (CULA_CORE_LIBRARY)


# Finish up

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CULA DEFAULT_MSG CULA_INCLUDE_DIR CULA_LIBRARIES)

MARK_AS_ADVANCED(CULA_INCLUDE_DIR CULA_LIBRARIES)
