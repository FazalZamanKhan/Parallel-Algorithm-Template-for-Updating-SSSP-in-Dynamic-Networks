#include <metis.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream> 

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input_graph_file> <num_partitions> <output_part_file>\n";
        return 1;
    }

    std::string filename = argv[1];
    int num_parts = std::stoi(argv[2]);
    std::string output_part_file = argv[3];

    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error opening file: " << filename << "\n";
        return 1;
    }

    idx_t nVertices, nEdges, fmt = 0;
    infile >> nVertices >> nEdges >> fmt;

    bool isWeighted = (fmt == 1 || fmt == 10 || fmt == 11);

    std::vector<idx_t> xadj(nVertices + 1);
    std::vector<idx_t> adjncy;
    std::vector<idx_t> adjwgt;

    std::string line;
    std::getline(infile, line); // consume the remaining header line

    idx_t edge_counter = 0;
    for (idx_t i = 0; i < nVertices; ++i) {
        std::getline(infile, line);
        std::istringstream iss(line);
        idx_t neighbor;
        idx_t weight;

        xadj[i] = adjncy.size();

        while (iss >> neighbor) {
            adjncy.push_back(neighbor - 1); // Convert to 0-based
            if (isWeighted && iss >> weight)
                adjwgt.push_back(weight);
        }
    }
    xadj[nVertices] = adjncy.size();

    std::vector<idx_t> part(nVertices);
    idx_t objval;

    int result = METIS_PartGraphKway(
        &nVertices,
        nullptr,            // ncon
        xadj.data(),
        adjncy.data(),
        nullptr,            // vwgt
        nullptr,            // vsize
        isWeighted ? adjwgt.data() : nullptr,
        &num_parts,
        nullptr,            // tpwgts
        nullptr,            // ubvec
        nullptr,            // options
        &objval,
        part.data()
    );

    if (result != METIS_OK) {
        std::cerr << "METIS partitioning failed.\n";
        return 1;
    }

    std::ofstream outfile(output_part_file);
    if (!outfile) {
        std::cerr << "Error opening output file.\n";
        return 1;
    }

    for (auto p : part) {
        outfile << p << "\n";
    }

    std::cout << "Partitioning complete. Cut edges: " << objval << "\n";
    return 0;
}
