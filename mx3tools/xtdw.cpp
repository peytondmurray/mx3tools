#include <xtdw.h>
#include <vector>
#define PI 3.14159265358979323846264338327950288419

// namespace fs = std::filesystem;
namespace bfs = boost::filesystem;

int nLines(std::string path, int nheader) {
    std::string line;
    int linecount = 0;
    std::ifstream f(path);
    while (std::getline(f, line)) {
        linecount += 1;
    }
    f.close();
    return linecount - nheader;
}

xt::xarray<float>* parseFile(std::string path, int nheader) {
    int size = nLines(path, nheader);
    auto phi = new xt::xtensor<float, 2>({size});

    // xt::xtensor_fixed<std::string, xt::xshape<6>> vec();
    std::vector<std::string> vec;
    std::string buffer;
    std::ifstream f(path);
    for (int j=0; j<nheader; j++) std::getline(f, buffer);

    int i = 0;
    while (std::getline(f, buffer)) {
        boost::algorithm::split(vec, buffer, boost::is_any_of(","));
        (*phi)[i] = atan2(std::stof(vec[4]), std::stof(vec[3]));
        i++;
    }
}