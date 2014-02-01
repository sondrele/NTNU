#Set this to @ to keep the makefile quiet
ifndef SILENCE
	SILENCE = @
endif

#--- Inputs ----#
COMPONENT_NAME = all
CPPUTEST_HOME = ../cpputest

CPPUTEST_USE_EXTENSIONS = Y
CPP_PLATFORM = GCC
CPPFLAGS += -include $(CPPUTEST_HOME)/include/CppUTest/MemoryLeakDetectorNewMacros.h
CPPUTEST_CPPFLAGS = -x c++ -g
CFLAGS += -std=c++11 -include $(CPPUTEST_HOME)/include/CppUTest/MemoryLeakDetectorMallocMacros.h

INCLUDE_DIRS=\
  $(CPPUTEST_HOME)/include\
  ../inc

SRC_DIRS=\
  ../src

TEST_SRC_DIRS=\
  ./src\
  ./

#CC = g++
#CC = clang++
#CFLAGS = $(INC_DIRS) -Wall -g -std=c++11

include $(CPPUTEST_HOME)/build/MakefileWorker.mk