INCLUDEPATH=\
    -I../matrix/inc\
    -Iinc
CC=clang++
CC=g++
CFLAGS = $(INCLUDEPATH) -Wall -g -std=c++11
CIMGFLAGS = $(CFLAGS) -L/usr/X11R6/lib -lm -lpthread -lX11
LDFLAGS=
LIBS = \
    lib/scene_io.cpp\
    lib/rayimage.cpp\
    ../matrix/src/Matrix.cpp\
    ../matrix/src/Vect.cpp
SRCS = src/ray.cpp\
    src/intersection.cpp\
    src/camera.cpp\
    src/raybuffer.cpp\
    src/rayscene.cpp\
    src/rayscene_factory.cpp\
    src/rayscene_shapes.cpp\
    src/whitted.cpp\
    src/raytracer.cpp
OBJS = $(SRCS:.cpp=.o) $(LIBS:.cpp=.o) 

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

all: raytracer main

raytracer: raytracer.o $(OBJS)
	$(CC) -o $@ $^ $(CIMGFLAGS)

IMG=main.cpp lib/rayimage.cpp src/raybuffer.cpp
IMGOBJS = $(IMG:.cpp=.o)

test1: all
	./raytracer test1 150 150 2

clean:
	rm -f *.o
	rm -rf **/*.o