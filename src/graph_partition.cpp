#include <iostream>
#include <vector>
#include <fstream>
#include <mpi.h>
#include <metis.h>
#include <cassert>

struct Graph {
    int n; // number of vertices
    int m; // number of edges
    std::vector<int> xadj;
    std::vector<int> adjncy;
    std::vector<int> weights;
};

// Read a METIS .graph file
bool readGraph(const char* filename, Graph& graph) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    fscanf(file, "%d %d", &graph.n, &graph.m);
    graph.xadj.resize(graph.n + 1);
    graph.adjncy.resize(graph.m);
    graph.weights.resize(graph.m);

    for (int i = 0; i < graph.n + 1; ++i)
        fscanf(file, "%d", &graph.xadj[i]);

    for (int i = 0; i < graph.m; ++i)
        fscanf(file, "%d", &graph.adjncy[i]);

    for (int i = 0; i < graph.m; ++i)
        fscanf(file, "%d", &graph.weights[i]);

    fclose(file);

    // Correct for 1-based indexing
    for (int& x : graph.xadj) x -= 1;
    for (int& x : graph.adjncy) x -= 1;

    return true;
}

// Partition the graph using METIS
void partitionGraph(const Graph& graph, std::vector<int>& part, int nparts) {
    idx_t n = graph.n;
    idx_t ncon = 1;
    idx_t edgecut;

    std::vector<idx_t> xadj(graph.xadj.begin(), graph.xadj.end());
    std::vector<idx_t> adjncy(graph.adjncy.begin(), graph.adjncy.end());
    std::vector<idx_t> weights(graph.weights.begin(), graph.weights.end());
    part.resize(graph.n);

    std::vector<idx_t> partIdx(graph.n);

    int result = METIS_PartGraphKway(
        &n,
        &ncon,
        xadj.data(),
        adjncy.data(),
        NULL,               // vertex weights
        NULL,               // ve
        // rtex sizes
        weights.data(),     // edge weights
        (idx_t*)&nparts,    // number of partitions
        NULL,               // tpwgts
        NULL,               // ubvec
        NULL,               // options
        &edgecut,
        partIdx.data()
    );

    if (result != METIS_OK) {
        std::cerr << "METIS partitioning failed!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < graph.n; ++i)
        part[i] = partIdx[i];
}

// Write partitioned data to output file
void writePartitionedGraph(const Graph& graph, const std::vector<int>& part, const char* outFilename) {
    std::ofstream out(outFilename);
    if (!out.is_open()) {
        std::cerr << "Failed to open output file." << std::endl;
        return;
    }

    out << "# NodeID PartitionID" << std::endl;
    for (int i = 0; i < graph.n; ++i) {
        out << i << " " << part[i] << std::endl;
    }

    out.close();
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Graph graph;
    std::vector<int> part;

    if (rank == 0) {
        // Step 1: Read graph
        if (!readGraph("metis.graph", graph)) {
            MPI_Finalize();
            return -1;
        }

        // Step 2: Partition graph using METIS
        partitionGraph(graph, part, size);

        // Step 3: Write partitioning results to file
        writePartitionedGraph(graph, part, "partitioned.graph");

        std::cout << "Graph partitioned into " << size << " parts. Output written to 'partitioned.graph'.\n";
    }

    MPI_Finalize();
    return 0;
}
