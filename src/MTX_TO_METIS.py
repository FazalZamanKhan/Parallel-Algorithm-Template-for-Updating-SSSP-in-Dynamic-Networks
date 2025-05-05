from collections import defaultdict

def convert_mtx_to_metis(input_file, output_file):
    edge_list = defaultdict(set)
    
    with open(input_file, 'r') as f:
        # Skip the header
        line = f.readline()
        while line.startswith('%'):
            line = f.readline()

        # Read matrix dimensions
        rows, cols, num_edges = map(int, line.split())
        
        # Read the edges
        for line in f:
            u, v, _ = map(int, line.split())  # We ignore the third value (the weight)
            if u != v:
                edge_list[u].add(v)
                edge_list[v].add(u)  # For undirected graph
    
    # Create a mapping from original node ids to new ids (0-based)
    all_nodes = sorted(edge_list.keys())
    node_id_map = {old: new for new, old in enumerate(all_nodes)}

    n = len(all_nodes)
    with open(output_file, 'w') as f:
        f.write(f"{n} {sum(len(vs) for vs in edge_list.values()) // 2}\n")
        for node in all_nodes:
            neighbors = sorted(edge_list[node])
            line = ' '.join(str(node_id_map[v] + 1) for v in neighbors)  # METIS is 1-based indexing
            f.write(line + '\n')

if __name__ == "__main__":
    input_path = "../datasets/email-Eu-core-weighted.mtx"  # Your Matrix Market file
    output_path = "metis.graph"  # Desired METIS output file
    convert_mtx_to_metis(input_path, output_path)
    print("Conversion complete. Output written to", output_path)
