
CXX = g++
HEADERS = dwread.h
OBJECTS = dwread.o
CXXFLAGS = -std=c++17
LDFLAGS = -lboost_filesystem -lboost_system

%.o: %.cpp $(HEADERS)
	g++ $(CXXFLAGS) -c $< -o $@

dwread: $(OBJECTS)
	g++ $(CXXFLAGS) $(OBJECTS) $(LDFLAGS) -o $@

debug: CXXFLAGS += -g
debug: dwread

release: CXXFLAGS += -O3
release: dwread

.PHONY: clean
clean:
	rm -f %.o dwread
