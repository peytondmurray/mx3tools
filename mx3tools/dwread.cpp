#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <boost/format.hpp>
// #include <numpy/arrayobject.h>


int nLines(std::string path, int nheader) {
    std::string line;
    int linecount = 0;
    std::ifstream f(path);
    if (!f) {
        std::cout << "Unable to open file.\n";
        exit(1);
    }
    while (std::getline(f, line)) {
        linecount += 1;
    }
    f.close();
    return linecount - nheader;
}

void parseBuffer(std::string buffer,
                 int i,
                 std::vector<double> &x,
                 std::vector<double> &y,
                 std::vector<double> &z,
                 std::vector<double> &mx,
                 std::vector<double> &my,
                 std::vector<double> &mz) {

    int pos = 0;

    // Strip trailing newline
    buffer = buffer.substr(0, buffer.find("\n"));

    pos = buffer.find(",");
    x[i] = std::stod(buffer.substr(0, pos));
    buffer = buffer.substr(pos+1, std::string::npos);

    pos = buffer.find(",");
    y[i] = std::stod(buffer.substr(0, pos));
    buffer = buffer.substr(pos+1, std::string::npos);

    pos = buffer.find(",");
    z[i] = std::stod(buffer.substr(0, pos));
    buffer = buffer.substr(pos+1, std::string::npos);

    pos = buffer.find(",");
    mx[i] = std::stod(buffer.substr(0, pos));
    buffer = buffer.substr(pos+1, std::string::npos);

    pos = buffer.find(",");
    my[i] = std::stod(buffer.substr(0, pos));
    buffer = buffer.substr(pos+1, std::string::npos);

    pos = buffer.find(",");
    mz[i] = std::stod(buffer.substr(0, pos));

    return;
}

// int readDWFile(std::string path) {
int main() {

    int nheader = 4;
    std::string path = "/home/pdmurray/Desktop/Workspace/dmidw/barkhausen/D_0.0e-3/2019-05-26/barkhausen_0.out/domainwall000000.csv";
    int size = nLines(path, nheader);
    std::ifstream f(path);
    std::string buffer;
    int i = 0;
    auto x = std::vector<double>(size);
    auto y = std::vector<double>(size);
    auto z = std::vector<double>(size);
    auto mx = std::vector<double>(size);
    auto my = std::vector<double>(size);
    auto mz = std::vector<double>(size);

    for (int i=0; i<nheader; i++) {
        std::getline(f, buffer);
    }
    while (std::getline(f, buffer)) {
        parseBuffer(buffer, i, x, y, z, mx, my, mz);
        i++;
    }

    std::string s;
    for (int i=0; i<x.size(); i++) {
        // std::sprintf(?)
        std::cout << x[i] << " "<< y[i] << " " << z[i] << std::endl;
    }

}
