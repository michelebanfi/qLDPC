import matplotlib.pyplot as plt
import networkx as nx

def plotGraph(adj, path):

    G = nx.Graph()
    m = len(adj)
    n = len(adj[0])

    checks = list(range(m))
    var = list(range(m, m + n))

    G.add_nodes_from(checks, bipartite=0)
    G.add_nodes_from(var, bipartite=1)

    for i in range(m):
        for j in range(n):
            if adj[i][j] == 1:
                G.add_edge(i, m + j)

    pos = nx.bipartite_layout(G, var)

    plt.figure(figsize=(8, 6))
    
    nx.draw_networkx_nodes(G, pos, nodelist=var, node_shape='o', node_color='lightblue', node_size=600)
    nx.draw_networkx_nodes(G, pos, nodelist=checks, node_shape='s', node_color='lightgreen', node_size=600)

    nx.draw_networkx_edges(G, pos)
    labels = {i: f"C{i+1}" for i in checks}
    labels.update({m + j: f"V{j+1}" for j in range(n)})
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

    plt.axis('off')
    plt.savefig(f"media/{path}.png", bbox_inches='tight', dpi=200)
    plt.close()
    
def plotMatrix(adj, path):
    plt.imshow(adj, cmap='Greys', interpolation='nearest')
    # plt.xlabel('Variable Nodes')
    # plt.ylabel('Check Nodes')
    # plt.title('Parity-Check Matrix')
    # plt.colorbar(label='Connection')
    plt.savefig(f"media/{path}_matrix.png", bbox_inches='tight', dpi=200)
    plt.close()