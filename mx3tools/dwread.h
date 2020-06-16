#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <tuple>
#include <cstring>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

int nLines(std::string, int);
void parseBuffer(std::string, int, std::vector<double>&, std::vector<double>&);
std::vector<double>* parseFile(std::string, int, int);
void mxmyToPhi(std::vector<double>&, std::vector<double>&, std::vector<double>&);
void dPhi(std::vector<double>&);
std::tuple<int, int> nBloch(std::vector<double>&);
std::tuple<int, int> readDWFile(std::string);
