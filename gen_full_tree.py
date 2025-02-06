import matplotlib.pyplot as plt
import networkx as nx

ary_list = [1, 1, 3, 3]
accepted_nodes = [tuple([i]) for i in range(ary_list[0])]
start_idx = 0
for i in range(1, len(ary_list)):
    nodes_to_process = accepted_nodes[start_idx:]
    start_idx = len(accepted_nodes)
    for node in nodes_to_process:
        for j in range(ary_list[i]):
            accepted_nodes.append(node + tuple([j]))
print(accepted_nodes)


def plot_and_save_graph(accept_nodes, output_path):
    plt.figure(figsize=(30, 15))

    G = nx.DiGraph()
    for path in accept_nodes:
        for i in range(len(path)):
            if i == 0:
                parent = "root"
            else:
                parent = tuple(path[:i])
            child = tuple(path[: i + 1])
            G.add_edge(parent, child)

    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    nx.draw(G, pos, with_labels=False, node_size=1000, node_color="skyblue", font_size=30, width=2, edge_color="gray")
    plt.savefig(output_path)


# plot_and_save_graph(accepted_nodes, "data/full_tree.jpg")
