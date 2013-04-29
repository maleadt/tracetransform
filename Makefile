SRCS=$(wildcard *.c)
OBJS=$(SRCS:.c=.mexa64)
all: $(OBJS)
clean:
	rm $(OBJS)
%.mexa64: %.c
	matlab-mex $<
