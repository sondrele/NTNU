INCLUDEPATH=\
    -Iinc
CC=g++
CC=clang++
CFLAGS = $(INCLUDEPATH) -Wall -g -std=c++11
#CIMGFLAGS = $(CFLAGS) -L/usr/X11R6/lib -lm -lpthread -lX11
LDFLAGS=
SRCS = lib/scene_io.cpp raytracer.cpp
OBJS = $(SRCS:.cpp=.o)

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

all: main

main: $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -rf **/*.o