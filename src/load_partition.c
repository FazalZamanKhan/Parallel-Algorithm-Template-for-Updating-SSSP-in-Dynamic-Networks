#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void load_partitioned_graph(const char *graph_file, const char *part_file, int rank, int size,
                            int **local_offsets, int **local_edges, int *local_n, int *local_m) {
    FILE *graph_fp = fopen(graph_file, "r");
    FILE *part_fp = fopen(part_file, "r");
    if (!graph_fp || !part_fp) {
        if (rank == 0) {
            fprintf(stderr, "Error: Unable to open graph or partition file.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int total_nodes, total_edges;
    if (fscanf(graph_fp, "%d %d\n", &total_nodes, &total_edges) != 2) {
        if (rank == 0) {
            fprintf(stderr, "Error: Invalid graph file format.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Read partition assignments
    int *partitions = malloc(total_nodes * sizeof(int));
    if (!partitions) {
        fprintf(stderr, "Error: Memory allocation failed for partitions.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < total_nodes; i++) {
        if (fscanf(part_fp, "%d", &partitions[i]) != 1) {
            if (rank == 0) {
                fprintf(stderr, "Error: Invalid partition file format.\n");
            }
            free(partitions);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    fclose(part_fp);

    // Count number of local nodes for this rank
    *local_n = 0;
    for (int i = 0; i < total_nodes; i++) {
        if (partitions[i] == rank) {
            (*local_n)++;
        }
    }

    *local_offsets = calloc((*local_n + 1), sizeof(int));
    *local_edges = malloc(total_edges * 2 * sizeof(int)); // Over-allocate
    if (!*local_offsets || !*local_edges) {
        fprintf(stderr, "Error: Memory allocation failed for local data.\n");
        free(partitions);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Skip the header line already read
    char line[10000]; // buffer for each adjacency list line
    int edge_count = 0, local_index = 0;
    (*local_offsets)[0] = 0;

    for (int i = 0; i < total_nodes; i++) {
        if (!fgets(line, sizeof(line), graph_fp)) {
            fprintf(stderr, "Error: Failed to read adjacency list.\n");
            free(partitions); free(*local_offsets); free(*local_edges);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (partitions[i] == rank) {
            char *token = strtok(line, " \t\n");
            while (token != NULL) {
                int neighbor = atoi(token);
                (*local_edges)[edge_count++] = neighbor;
                token = strtok(NULL, " \t\n");
            }
            (*local_offsets)[++local_index] = edge_count;
        }
    }

    *local_m = edge_count;
    free(partitions);
    fclose(graph_fp);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc < 3) {
      if (rank == 0)
          printf("Usage: mpirun -np <N> ./load_graph_mpi <graph_file> <part_file>\n");
      MPI_Finalize();
      return 1;
  }

  int *local_offsets = NULL, *local_edges = NULL, local_n = 0, local_m = 0;
  load_partitioned_graph(argv[1], argv[2], rank, size, &local_offsets, &local_edges, &local_n, &local_m);

  // 🛠️ THIS LINE IS WRONG IN YOUR OUTPUT
  // Replace this:
  // printf("Rank 0: Loaded %d local nodes and %d edges.\n", local_n, local_m);
  // With this:
  printf("Rank %d: Loaded %d local nodes and %d edges.\n", rank, local_n, local_m);

  free(local_offsets);
  free(local_edges);
  MPI_Finalize();
  return 0;
}
