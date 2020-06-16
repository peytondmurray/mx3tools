#include "dwread.h"
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

std::vector<double>* parseFile(std::string path, int nheader, int size) {
    auto phi = new std::vector<double>(size);
    std::vector<std::string> vec;
    std::ifstream f(path);
    std::string buffer;
    int i = 0;

    for (int j=0; j<nheader; j++) {
        std::getline(f, buffer);
    }
    while (std::getline(f, buffer)) {
        boost::algorithm::split(vec, buffer, boost::is_any_of(","));
        (*phi)[i] = atan2(std::stod(vec[4]), std::stod(vec[3]));
        i++;
    }

    dPhi(*phi);
    f.close();
    return phi;
}

void dPhi(std::vector<double> &phi) {
    double dPhi;
    for (int i=phi.size()-1; i>0; i--) {
        dPhi = phi[i] - phi[i-1];
        if (dPhi < -1*PI)
        {
            phi[i] = 2*PI + dPhi;
        } else if (dPhi > PI) {
            phi[i] = 2*PI - dPhi;
        }
        else phi[i] = dPhi;
    }
    phi[0] = 0;
    return;
}

std::tuple<int, int> nBloch(std::vector<double> &dphi) {

    double s = 0;
    int np = 0;
    int nm = 0;

    for (int i=0; i<int(dphi.size()); i++) {
        s += dphi[i];
        if (s > PI) {
            s -= PI;
            np += 1;
        } else if (s < -PI) {
            s += PI;
            nm += 1;
        }
    }
    return {np, nm};
}

std::tuple<std::vector<int>*, std::vector<int>*> avgBlochSimData(std::string path) {

    std::vector<std::string> filenames;

    for (const auto & item : bfs::directory_iterator(path)) {
        if (bfs::path(item.path()).extension() == ".csv") filenames.push_back(item.path().string());
    }
    std::sort(filenames.begin(), filenames.end());
    auto np = new std::vector<int>(filenames.size());
    auto nm = new std::vector<int>(filenames.size());

    for (int i=0; i<int(filenames.size()); i++) {
        std::cout << "\t" << filenames[i] << "\r";
        auto [_np, _nm] = readDWFile(filenames[i]);
        (*np)[i] = _np;
        (*nm)[i] = _nm;
    }

    return {np, nm};
}

std::tuple<int, int> readDWFile(std::string path) {
    int nheader = 4;
    int size = nLines(path, nheader);
    auto phi = parseFile(path, nheader, size);
    auto [np, nm] = nBloch(*phi);
    delete phi;
    return {np, nm};
}


void deallocate(std::vector<std::vector<int>*> a) {
    for (int i=0; i<int(a.size()); i++) delete a[i];
}

int main(int argc, char **argv) {

    if ((argc == 5) && (std::strcmp(argv[1], "-i") == 0) && (std::strcmp(argv[3], "-o") == 0)) {
        std::string path(argv[2]);
        std::ofstream outfile;
        bfs::path fpath;
        int np_avg = 0;
        int nm_avg = 0;

        outfile.open(argv[4]);
        outfile << "Simulation,Bloch (+),Bloch(-)\n";
        for (const auto & item : bfs::directory_iterator(path)) {
            if (bfs::path(item.path()).extension() == ".out") {
                std::cout << item.path() << std::endl;
                auto [_np, _nm] = avgBlochSimData(item.path().string());

                for (int i=0; i<int((*_np).size()); i++) {
                    outfile << bfs::path(item.path()).filename() << "," << (*_np)[i] << "," << (*_nm)[i] << std::endl;
                }
                delete _np;
                delete _nm;
            }
        }

        outfile.close();
        std::cout << std::endl << "Finished." << std::endl;
    } else {
        std::cout << "Usage:\n\tdwread -o <data_directory> -i <output filename>\n";
    }
    return 0;
}
