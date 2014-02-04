INCLUDEPATH=\
	-I../matrix/inc\
    -Iinc
CC=clang++
CC=g++
CFLAGS = $(INCLUDEPATH) -Wall -g -std=c++11
CIMGFLAGS = $(CFLAGS) -L/usr/X11R6/lib -lm -lpthread -lX11
LDFLAGS=
SRCS = lib/scene_io.cpp raytracer.cpp
OBJS = $(SRCS:.cpp=.o)

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

all: raytracer

raytracer: $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS)

IMG=main.cpp lib/rayimage.cpp src/raybuffer.cpp
IMGOBJS = $(IMG:.cpp=.o)
main: $(IMGOBJS)
	$(CC) -o $@ $^ $(CIMGFLAGS)

clean:
	rm -f *.o
	rm -rf **/*.o