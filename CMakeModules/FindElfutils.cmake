FIND_PATH(Elfutils_ROOT_DIR elfutils/libdw.h
	$ENV{Elfutils_ROOT_DIR})

SET(Elfutils_INCLUDE_DIR ${Elfutils_ROOT_DIR})

FIND_LIBRARY(DW_LIBRARY 
  NAMES dw 
  PATHS /usr/lib
        /usr/$ENV{MACHTYPE}/lib)

IF (Elfutils_INCLUDE_DIR)
        SET(Elfutils_FOUND TRUE)
ENDIF (Elfutils_INCLUDE_DIR)
