#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <limits>
#include <queue>
#include <algorithm>
#include <set>

const int INF = std::numeric_limits<int>::max();

struct Edge {
    int to;
    int weight;
};

void read_metis_graph(const std::string& filename, std::vector<std::vector<Edge>>& adj_list, int& num_vertices) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error opening file: " << filename << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::string line;
    while (std::getline(infile, line)) {
        if (line[0] != '%') break;
    }

    std::istringstream header(line);
    int num_edges, fmt = 0;
    header >> num_vertices >> num_edges >> fmt;

    bool isWeighted = (fmt == 1 || fmt == 10 || fmt == 11);
    adj_list.resize(num_vertices);

    for (int i = 0; i < num_vertices; ++i) {
        if (!std::getline(infile, line)) {
            std::cerr << "Error reading line " << i + 1 << " from graph file.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::istringstream iss(line);
        int neighbor, weight;
        while (iss >> neighbor) {
            if (isWeighted) {
                if (!(iss >> weight)) {
                    std::cerr << "Error reading weight.\n";
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            } else {
                weight = 1;
            }
            adj_list[i].push_back({neighbor - 1, weight});
        }
    }
}

void distribute_graph(const std::vector<std::vector<Edge>>& adj_list, std::vector<std::vector<Edge>>& local_adj_list, int rank, int size, int& start_idx, int& end_idx) {
    int num_vertices = adj_list.size();
    int vertices_per_proc = num_vertices / size;
    int remainder = num_vertices % size;

    start_idx = rank * vertices_per_proc + std::min(rank, remainder);
    end_idx = start_idx + vertices_per_proc + (rank < remainder ? 1 : 0);

    local_adj_list.assign(adj_list.begin() + start_idx, adj_list.begin() + end_idx);
}

void apply_changes(std::vector<std::vector<Edge>>& adj_list, const std::string& changes_file) {
    std::ifstream infile(changes_file);
    if (!infile) {
        std::cerr << "Error opening changes file.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        char type;
        int u, v, w;
        if (!(iss >> type >> u >> v >> w)) continue;
        if (type == 'I') {
            adj_list[u].push_back({v, w});
        } else if (type == 'D') {
            auto& edges = adj_list[u];
            edges.erase(std::remove_if(edges.begin(), edges.end(), [&](const Edge& e) {
                return e.to == v && e.weight == w;
            }), edges.end());
        }
    }
}

void dijkstra(const std::vector<std::vector<Edge>>& local_adj_list, int global_start_vertex, int start_idx, int end_idx, std::vector<int>& local_distances, int num_vertices, int rank, int size) {
    int local_n = local_adj_list.size();
    local_distances.assign(local_n, INF);

    std::vector<int> global_distances(num_vertices, INF);
    std::vector<bool> visited(num_vertices, false);

    if (global_start_vertex >= start_idx && global_start_vertex < end_idx) {
        local_distances[global_start_vertex - start_idx] = 0;
    }

    for (int i = 0; i < num_vertices; ++i) {
        int local_min = INF, local_min_idx = -1;
        for (int j = 0; j < local_n; ++j) {
            int global_idx = j + start_idx;
            if (!visited[global_idx] && local_distances[j] < local_min) {
                local_min = local_distances[j];
                local_min_idx = global_idx;
            }
        }

        struct {
            int dist;
            int idx;
        } local_data = {local_min, local_min_idx}, global_data;

        MPI_Allreduce(&local_data, &global_data, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

        if (global_data.idx == -1) break;
        visited[global_data.idx] = true;

        int u = global_data.idx;
        int u_dist = global_data.dist;

        if (u >= start_idx && u < end_idx) {
            int local_u = u - start_idx;
            for (const auto& edge : local_adj_list[local_u]) {
                int v = edge.to;
                int new_dist = u_dist + edge.weight;
                if (v >= start_idx && v < end_idx) {
                    int local_v = v - start_idx;
                    if (local_distances[local_v] > new_dist) {
                        local_distances[local_v] = new_dist;
                    }
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = MPI_Wtime();  // Start time measurement

    if (argc < 4) {
        if (rank == 0) std::cerr << "Usage: ./mpi_sssp <metis.graph> <changes.txt> <source_vertex>\n";
        MPI_Finalize();
        return 1;
    }

    std::string graph_file = argv[1];
    std::string changes_file = argv[2];
    int source_vertex = std::stoi(argv[3]);

    std::vector<std::vector<Edge>> adj_list;
    int num_vertices;

    if (rank == 0) {
        read_metis_graph(graph_file, adj_list, num_vertices);
        apply_changes(adj_list, changes_file);
    }

    MPI_Bcast(&num_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) adj_list.resize(num_vertices);

    for (int i = 0; i < num_vertices; ++i) {
        int edge_count;
        if (rank == 0) edge_count = adj_list[i].size();
        MPI_Bcast(&edge_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) adj_list[i].resize(edge_count);
        for (int j = 0; j < edge_count; ++j) {
            int to, weight;
            if (rank == 0) {
                to = adj_list[i][j].to;
                weight = adj_list[i][j].weight;
            }
            MPI_Bcast(&to, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&weight, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (rank != 0) {
                adj_list[i][j] = {to, weight};
            }
        }
    }

    std::vector<std::vector<Edge>> local_adj_list;
    int start_idx, end_idx;
    distribute_graph(adj_list, local_adj_list, rank, size, start_idx, end_idx);

    std::vector<int> local_distances;
    dijkstra(local_adj_list, source_vertex, start_idx, end_idx, local_distances, num_vertices, rank, size);

    std::vector<int> all_distances;
    if (rank == 0) all_distances.resize(num_vertices);

    std::vector<int> recvcounts(size), displs(size);
    int base = num_vertices / size, rem = num_vertices % size;
    for (int i = 0; i < size; ++i) {
        recvcounts[i] = base + (i < rem ? 1 : 0);
        displs[i] = i * base + std::min(i, rem);
    }

    MPI_Gatherv(local_distances.data(), local_distances.size(), MPI_INT,
                all_distances.data(), recvcounts.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();  // End time measurement
    double elapsed_time = end_time - start_time;

    if (rank == 0) {
        // Output the results to output_mpi.txt
        std::ofstream outfile("output_mpi.txt");
        if (outfile.is_open()) {
            outfile << "Execution Time: " << elapsed_time << " seconds\n";
            outfile << "Final shortest distances from vertex " << source_vertex << ":\n";
            for (int i = 0; i < num_vertices; ++i) {
                if (all_distances[i] == INF) outfile << "Vertex " << i << ": INF\n";
                else outfile << "Vertex " << i << ": " << all_distances[i] << "\n";
            }
            outfile.close();
        } else {
            std::cerr << "Error opening output file for writing.\n";
        }
    }

    MPI_Finalize();
    return 0;
}




//COMMANDS TO RUN THE CODE :


//mpicxx -o mpi_sssp mpi_sssp.cpp
// mpirun -np 2 ./mpi_sssp metis.graph changes.txt 0
