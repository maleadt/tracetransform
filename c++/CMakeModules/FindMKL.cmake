# - Find Intel MKL
# This module finds Intel MKL libraries by using the mkl_link_tool.
#
# Accepted options:
#   MKL_LINKTOOL_OPTS : additional arguments to pass to mkl_link_tool
#
# Defined variables
#   MKL_FOUND   : true if everything is found
#   MKL_CFLAGS  : compilation flags to add
#   MKL_LDFLAGS : linker flags to add
#   MKL_ENV     : runtime environment flags to use

# TODO: don't regex -lm away

FIND_PROGRAM(MKL_LINKTOOL mkl_link_tool
    PATHS $ENV{MKLROOT}/tools)

IF (MKL_LINKTOOL)
    EXECUTE_PROCESS(COMMAND ${MKL_LINKTOOL} ${MKL_LINKTOOL_OPTS} -libs
        OUTPUT_VARIABLE MKL_LDFLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_VARIABLE MKL_ERROR
        RESULT_VARIABLE RETVAR)
    IF (NOT "${RETVAR}" STREQUAL "0")
      message(FATAL_ERROR "MKL link tool failure while detecting library flags\n${MKL_ERROR}")
    ENDIF()
    STRING(REGEX REPLACE "-lm$" "" MKL_LDFLAGS ${MKL_LDFLAGS})
    STRING(REPLACE "$(MKLROOT\)" $ENV{MKLROOT} MKL_LDFLAGS ${MKL_LDFLAGS})

    EXECUTE_PROCESS(COMMAND ${MKL_LINKTOOL} ${MKL_LINKTOOL_OPTS} -opts
        OUTPUT_VARIABLE MKL_CFLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_VARIABLE MKL_ERROR
        RESULT_VARIABLE RETVAR)
    IF (NOT "${RETVAR}" STREQUAL "0")
      message(FATAL_ERROR "MKL link tool failure while detecting compilation flags\n${MKL_ERROR}")
    ENDIF()
    STRING(REGEX REPLACE "-lm$" "" MKL_CFLAGS ${MKL_CFLAGS})
    STRING(REPLACE "$(MKLROOT\)" $ENV{MKLROOT} MKL_CFLAGS ${MKL_CFLAGS})

    EXECUTE_PROCESS(COMMAND ${MKL_LINKTOOL} ${MKL_LINKTOOL_OPTS} -env
        OUTPUT_VARIABLE MKL_ENV
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_VARIABLE MKL_ERROR
        RESULT_VARIABLE RETVAR)
    IF (NOT "${RETVAR}" STREQUAL "0")
      message(FATAL_ERROR "MKL link tool failure while detecting environment flags\n${MKL_ERROR}")
    ENDIF()
    STRING(REPLACE "$(MKLROOT\)" $ENV{MKLROOT} MKL_ENV ${MKL_ENV})
ENDIF (MKL_LINKTOOL)


# Finish up

MARK_AS_ADVANCED(MKL_CFLAGS MKL_LDFLAGS MKL_ENV)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MKL DEFAULT_MSG MKL_LDFLAGS MKL_CFLAGS MKL_ENV)
