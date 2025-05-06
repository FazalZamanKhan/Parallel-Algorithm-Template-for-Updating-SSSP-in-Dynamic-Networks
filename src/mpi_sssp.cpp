#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <limits>
#include <queue>
#include <algorithm>

const int INF = std::numeric_limits<int>::max();

struct Edge {
    int to;
    int weight;
};

void read_metis_graph(const std::string& filename, std::vector<std::vector<Edge>>& adj_list, int& num_vertices, int& num_edges) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error opening file: " << filename << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::string line;
    // Skip comments
    while (std::getline(infile, line)) {
        if (line[0] != '%') break;
    }

    std::istringstream header(line);
    int fmt = 0;
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
                    std::cerr << "Error reading weight for edge.\n";
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            } else {
                weight = 1;
            }
            adj_list[i].push_back({neighbor - 1, weight}); // Convert to 0-based index
        }
    }
}

void distribute_graph(const std::vector<std::vector<Edge>>& adj_list, std::vector<std::vector<Edge>>& local_adj_list, int rank, int size) {
    int num_vertices = adj_list.size();
    int vertices_per_proc = num_vertices / size;
    int remainder = num_vertices % size;

    int start = rank * vertices_per_proc + std::min(rank, remainder);
    int end = start + vertices_per_proc + (rank < remainder ? 1 : 0);

    local_adj_list.assign(adj_list.begin() + start, adj_list.begin() + end);
}

void dijkstra(const std::vector<std::vector<Edge>>& local_adj_list, int global_start_vertex, int rank, int size, std::vector<int>& local_distances) {
    int local_n = local_adj_list.size();
    local_distances.assign(local_n, INF);

    int num_vertices;
    MPI_Allreduce(&local_n, &num_vertices, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    int vertices_per_proc = num_vertices / size;
    int remainder = num_vertices % size;

    int start = rank * vertices_per_proc + std::min(rank, remainder);
    int end = start + vertices_per_proc + (rank < remainder ? 1 : 0);

    std::vector<bool> visited(local_n, false);
    std::vector<int> global_distances(num_vertices, INF);

    if (global_start_vertex >= start && global_start_vertex < end) {
        local_distances[global_start_vertex - start] = 0;
    }

    for (int i = 0; i < num_vertices; ++i) {
        int local_min = INF;
        int local_min_index = -1;

        for (int j = 0; j < local_n; ++j) {
            if (!visited[j] && local_distances[j] < local_min) {
                local_min = local_distances[j];
                local_min_index = j + start;
            }
        }

        struct {
            int value;
            int rank;
        } local_data = {local_min, (local_min_index == -1) ? -1 : rank}, global_data;

        MPI_Allreduce(&local_data, &global_data, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

        int u = global_data.rank;
        int u_global_index = -1;

        if (rank == global_data.rank && local_min_index != -1) {
            u_global_index = local_min_index;
            visited[local_min_index - start] = true;
        }

        MPI_Bcast(&u_global_index, 1, MPI_INT, global_data.rank, MPI_COMM_WORLD);

        if (u_global_index == -1) break;

        int u_distance;
        if (u_global_index >= start && u_global_index < end) {
            u_distance = local_distances[u_global_index - start];
        }
        MPI_Bcast(&u_distance, 1, MPI_INT, global_data.rank, MPI_COMM_WORLD);

        for (int j = 0; j < local_n; ++j) {
            for (const auto& edge : local_adj_list[j]) {
                if (edge.to == u_global_index) {
                    if (local_distances[j] > u_distance + edge.weight) {
                        local_distances[j] = u_distance + edge.weight;
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
        if (rank == 0)
            std::cerr << "Usage: mpirun -np <num_processes> ./mpi_sssp <graph_file> <source_vertex>\n";
        MPI_Finalize();
        return 1;
    }

    std::string graph_file = argv[1];
    int source_vertex = std::stoi(argv[2]);

    std::vector<std::vector<Edge>> adj_list;
    int num_vertices, num_edges;

    if (rank == 0) {
        read_metis_graph(graph_file, adj_list, num_vertices, num_edges);
    }

    MPI_Bcast(&num_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        adj_list.resize(num_vertices);
    }

    // Broadcast the adjacency list to all processes
    for (int i = 0; i < num_vertices; ++i) {
        int edge_count;
        if (rank == 0) {
            edge_count = adj_list[i].size();
        }
        MPI_Bcast(&edge_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            adj_list[i].resize(edge_count);
        }
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
    distribute_graph(adj_list, local_adj_list, rank, size);

    std::vector<int> local_distances;
    dijkstra(local_adj_list, source_vertex, rank, size, local_distances);

    // Gather results at root process
    std::vector<int> all_distances;
    if (rank == 0) {
        all_distances.resize(num_vertices);
    }

    int local_n = local_adj_list.size();
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);

    int vertices_per_proc = num_vertices / size;
    int remainder = num_vertices % size;

    for (int i = 0; i < size; ++i) {
        recvcounts[i] = vertices_per_proc + (i < remainder ? 1 : 0);
        displs[i] = i * vertices_per_proc + std::min(i, remainder);
    }

    MPI_Gatherv(local_distances.data(), local_n, MPI_INT,
                all_distances.data(), recvcounts.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Shortest distances from vertex " << source_vertex << ":\n";
        for (int i = 0; i < num_vertices; ++i) {
            if (all_distances[i] == INF) {
                std::cout << "Vertex " << i << ": INF\n";
            } else {
                std::cout << "Vertex " << i << ": " << all_distances[i] << "\n";
            }
        }
    }

    MPI_Finalize();
    return 0;
}
