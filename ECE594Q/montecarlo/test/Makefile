#Set this to @ to keep the makefile quiet
ifndef SILENCE
    SILENCE = @
endif

#--- Inputs ----#
COMPONENT_NAME = all
CPPUTEST_HOME = ../../cpputest

CPPUTEST_USE_EXTENSIONS = Y
CPP_PLATFORM = GCC
# CXX = clang
CPPFLAGS += -include $(CPPUTEST_HOME)/include/CppUTest/MemoryLeakDetectorNewMacros.h
CPPUTEST_CPPFLAGS = -x c++ -g
CFLAGS += -std=c++11 -include $(CPPUTEST_HOME)/include/CppUTest/MemoryLeakDetectorMallocMacros.h
LD_LIBRARIES = -lm -L$(CPPUTEST_HOME)/lib -lCppUTest -lCppUTestExt 
#-L/lib/x86_64-linux-gnu/libm.so.6 

INCLUDE_DIRS=\
  $(CPPUTEST_HOME)/include\
  ./lib\
  ../inc\
  ../../matrix/inc

SRC_DIRS=\
  ../src\
  ../../matrix/src

TEST_SRC_DIRS=\
  ./src\
  ./

include $(CPPUTEST_HOME)/build/MakefileWorker.mk