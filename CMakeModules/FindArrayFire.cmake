# - Find ArrayFire
# Find the native ArrayFire includes and library
#
#   ArrayFire_FOUND       - True if ArrayFire found.
#   ArrayFire_INCLUDE_DIR - where to find arrayfire.h, etc.
#   ArrayFire_LIBRARIES   - List of libraries when using ArrayFire.
#

IF( ArrayFire_INCLUDE_DIR )
    # Already in cache, be silent
    SET( ArrayFire_FIND_QUIETLY TRUE )
ENDIF( ArrayFire_INCLUDE_DIR )

FIND_PATH( ArrayFire_INCLUDE_DIR "arrayfire.h"
           PATH_SUFFIXES "arrayfire/include"
           PATHS /usr /usr/local /opt )

MESSAGE("ArrayFire_INCLUDE_DIR = ${ArrayFire_INCLUDE_DIR}")


FIND_LIBRARY( ArrayFire_LIBRARY
              NAMES "af"
              PATH_SUFFIXES "arrayfire/lib64"
              PATHS /usr/ usr/local /opt )

FIND_LIBRARY( ArrayFire_GFX_LIBRARY
              NAMES "afGFX"
              PATH_SUFFIXES "arrayfire/lib64"
              PATHS /usr/ usr/local /opt )

if (ArrayFire_LIBRARY)
  list(APPEND ArrayFire_LIBRARIES ${ArrayFire_LIBRARY})
endif (ArrayFire_LIBRARY)

if (ArrayFire_GFX_LIBRARY)
  list(APPEND ArrayFire_LIBRARIES ${ArrayFire_GFX_LIBRARY})
endif (ArrayFire_GFX_LIBRARY)

MESSAGE("ArrayFire_LIBRARIES = ${ArrayFire_LIBRARIES}")

# handle the QUIETLY and REQUIRED arguments and set ArrayFire_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE( "FindPackageHandleStandardArgs" )
FIND_PACKAGE_HANDLE_STANDARD_ARGS( "ArrayFire" DEFAULT_MSG ArrayFire_INCLUDE_DIR ArrayFire_LIBRARIES )

MARK_AS_ADVANCED( ArrayFire_INCLUDE_DIR ArrayFire_LIBRARIES )
