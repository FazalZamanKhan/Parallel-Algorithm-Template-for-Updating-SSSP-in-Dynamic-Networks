from collections import defaultdict

def convert_snap_to_metis(input_file, output_file):
    edge_list = defaultdict(set)

    with open(input_file, 'r') as f:
        for line in f:
            if line.startswith("#"): continue
            u, v = map(int, line.strip().split())
            if u != v:
                edge_list[u].add(v)
                edge_list[v].add(u)  # undirected

    all_nodes = sorted(edge_list.keys())
    node_id_map = {old: new for new, old in enumerate(all_nodes)}

    n = len(all_nodes)
    with open(output_file, 'w') as f:
        f.write(f"{n} {sum(len(vs) for vs in edge_list.values()) // 2}\n")
        for node in all_nodes:
            neighbors = sorted(edge_list[node])
            line = ' '.join(str(node_id_map[v] + 1) for v in neighbors)
            f.write(line + '\n')

if __name__ == "__main__":
    input_path = "../datasets/email-Eu-core.txt"
    output_path = "email-Eu-core.graph"
    convert_snap_to_metis(input_path, output_path)
    print("Conversion complete. Output written to", output_path)
