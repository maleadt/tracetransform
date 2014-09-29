CC=gcc-4.4
CFLAGS=-std=c99 -fPIC -D_GNU_SOURCE
SRCS=$(wildcard *.c)
OBJS=$(SRCS:.c=.mexa64)

.PHONY: all
all: $(OBJS)

.PHONY: clean
clean:
	rm $(OBJS)

%.mexa64: %.c
	matlab-mex CC="${CC}" CFLAGS="${CFLAGS}" $<
