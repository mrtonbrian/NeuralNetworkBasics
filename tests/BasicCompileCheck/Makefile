# use g++ with C++11 support
CXX=g++
CXXFLAGS?=-Wall -Wextra -Wno-unused-parameter -Wno-reorder -Werror=return-type -pedantic -Og -std=c++17
TESTFILES=main

all: $(TESTFILES)

main: main.cpp
	$(CXX) $(CXXFLAGS) -o main main.cpp

clean:
	$(RM) $(TESTFILES)  *.o 