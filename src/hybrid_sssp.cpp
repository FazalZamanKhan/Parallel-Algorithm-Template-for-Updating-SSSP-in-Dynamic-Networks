#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <climits>
#include <algorithm>

const int INF = INT_MAX;

struct Edge {
    int to;
    int weight;
};

void read_metis_graph(const std::string& filename, std::vector<std::vector<Edge>>& adj_list, int& num_vertices) {
    std::ifstream infile(filename);
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
        if (!std::getline(infile, line)) break;
        std::istringstream iss(line);
        int neighbor, weight = 1; // Default weight
        while (iss >> neighbor) {
            if (isWeighted) iss >> weight;
            adj_list[i].push_back({neighbor - 1, weight});
        }
    }
}

void distribute_graph(const std::vector<std::vector<Edge>>& adj_list, 
                     std::vector<std::vector<Edge>>& local_adj_list, 
                     int rank, int size, int& start_idx, int& end_idx) {
    int num_vertices = adj_list.size();
    int per_proc = num_vertices / size;
    int rem = num_vertices % size;
    start_idx = rank * per_proc + std::min(rank, rem);
    end_idx = start_idx + per_proc + (rank < rem ? 1 : 0);
    local_adj_list.assign(adj_list.begin() + start_idx, adj_list.begin() + end_idx);
}

void parallel_dijkstra(const std::vector<std::vector<Edge>>& local_adj_list, 
                      int start_vertex, int start_idx, int end_idx, 
                      std::vector<int>& local_distances, int num_vertices, 
                      int rank, int size) {
    int local_n = local_adj_list.size();
    local_distances.assign(local_n, INF);
    
    // Initialize distances
    if (start_vertex >= start_idx && start_vertex < end_idx) {
        local_distances[start_vertex - start_idx] = 0;
    }

    std::vector<bool> visited(num_vertices, false);
    int global_min_idx = start_vertex;
    int global_min_dist = 0;

    for (int i = 0; i < num_vertices; ++i) {
        // Find local minimum
        int local_min = INF;
        int local_min_idx = -1;
        
        #pragma omp parallel for reduction(min: local_min)
        for (int j = 0; j < local_n; ++j) {
            int global_j = j + start_idx;
            if (!visited[global_j] && local_distances[j] < local_min) {
                local_min = local_distances[j];
                local_min_idx = global_j;
            }
        }

        // Find global minimum
        struct {
            int dist;
            int idx;
        } local_data = {local_min, local_min_idx}, global_data;

        MPI_Allreduce(&local_data, &global_data, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
        global_min_dist = global_data.dist;
        global_min_idx = global_data.idx;

        if (global_min_idx == -1 || global_min_dist == INF) break;

        visited[global_min_idx] = true;

        // Update neighbors
        if (global_min_idx >= start_idx && global_min_idx < end_idx) {
            int local_u = global_min_idx - start_idx;
            
            #pragma omp parallel for
            for (size_t k = 0; k < local_adj_list[local_u].size(); ++k) {
                const Edge& edge = local_adj_list[local_u][k];
                int v = edge.to;
                int new_dist = global_min_dist + edge.weight;
                
                if (v >= 0 && v < num_vertices && new_dist < INF) {
                    if (v >= start_idx && v < end_idx) {
                        int local_v = v - start_idx;
                        if (local_distances[local_v] > new_dist) {
                            #pragma omp critical
                            {
                                if (local_distances[local_v] > new_dist) {
                                    local_distances[local_v] = new_dist;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) std::cerr << "Usage: ./hybrid_sssp <graph_file> <source_vertex>\n";
        MPI_Finalize();
        return 1;
    }

    std::string graph_file = argv[1];
    int source_vertex = std::stoi(argv[2]);

    std::vector<std::vector<Edge>> adj_list;
    int num_vertices = 0;

    if (rank == 0) {
        read_metis_graph(graph_file, adj_list, num_vertices);
        if (source_vertex < 0 || source_vertex >= num_vertices) {
            std::cerr << "Invalid source vertex\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&num_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) adj_list.resize(num_vertices);

    // Broadcast the adjacency list
    for (int i = 0; i < num_vertices; ++i) {
        int edge_count = adj_list[i].size();
        MPI_Bcast(&edge_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) adj_list[i].resize(edge_count);
        
        if (edge_count > 0) {
            if (rank == 0) {
                MPI_Bcast(&adj_list[i][0], edge_count * sizeof(Edge), MPI_BYTE, 0, MPI_COMM_WORLD);
            } else {
                MPI_Bcast(&adj_list[i][0], edge_count * sizeof(Edge), MPI_BYTE, 0, MPI_COMM_WORLD);
            }
        }
    }

    std::vector<std::vector<Edge>> local_adj_list;
    int start_idx, end_idx;
    distribute_graph(adj_list, local_adj_list, rank, size, start_idx, end_idx);

    std::vector<int> local_distances;
    double t_start = MPI_Wtime();
    parallel_dijkstra(local_adj_list, source_vertex, start_idx, end_idx, local_distances, num_vertices, rank, size);
    double t_end = MPI_Wtime();

    // Gather results
    std::vector<int> all_distances;
    if (rank == 0) all_distances.resize(num_vertices, INF);

    std::vector<int> recvcounts(size), displs(size);
    int base = num_vertices / size;
    int rem = num_vertices % size;
    for (int i = 0; i < size; ++i) {
        recvcounts[i] = base + (i < rem ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i-1] + recvcounts[i-1];
    }

    MPI_Gatherv(local_distances.data(), local_distances.size(), MPI_INT,
                all_distances.data(), recvcounts.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::ofstream out("hybrid_output.txt");
        if (!out) {
            std::cerr << "Error opening output file\n";
        } else {
            out << "Execution Time: " << (t_end - t_start) << " seconds\n";
            for (int i = 0; i < num_vertices; ++i) {
                if (all_distances[i] == INF)
                    out << "Vertex " << i << ": INF\n";
                else
                    out << "Vertex " << i << ": " << all_distances[i] << "\n";
            }
            out.close();
        }
    }

    MPI_Finalize();
    return 0;
}