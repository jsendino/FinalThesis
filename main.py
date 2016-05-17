#!/usr/bin/env python
import numpy as np
import networkx as nx

from constants import Constants
from cost import Cost
from consumers import Consumers

__author__ = 'jorge'


def main():
    seed = 18765
    np.random.seed(seed)

    # Generate barabasi-albert graph with 100 nodes
    # G = nx.barabasi_albert_graph(100, m=1, seed=seed)
    # G = nx.powerlaw_cluster_graph(100, 1, 0, seed)
    G = nx.random_powerlaw_tree(100, 3, seed)
    # Get list of nodes with degree connectivity in descending order
    node_degree_list = sorted(nx.degree_centrality(G).items(),
                              key=lambda x:x[1],
                              reverse=True)
    # Take as providers those nodes with degree connectivity greater that 10%
    provider_list = list(filter(lambda x: x[1] > 0.05, node_degree_list))
    providers = [pro[0] for pro in provider_list]
    # Create dictionary with each node taken as provider and nearest provider to them (that will be the providers which
    # can drag customers from the first)
    distances = {}
    # For each node in list, find distance with the others and then remove connection between them
    for i, pro1 in enumerate(providers):
        d = np.zeros(len(providers))
        for j, pro2 in enumerate(providers):
            # Do not compare a node with itself
            if j == i:
                d[j] = float('inf')
                continue
            d[j] = len(nx.shortest_path(G, pro1, pro2))
        # From those nodes at minimum distance, choice randomly between them
        nearest_neighbours = np.where(d == np.min(d))[0]
        #distances[i] = np.random.choice(nearest_neighbours)
        distances[i] = nearest_neighbours

    for i, pro1 in enumerate(providers):
        for pro2 in providers[i+1:]:
            if (pro1, pro2) in G.edges() or (pro2, pro1) in G.edges():
                # Remove edges between providers to isolate neighbors
                G.remove_edge(pro1, pro2)
            try:
                # Get path between providers (if exists any yet) as list of edges
                path = nx.shortest_path(G, pro1, pro2)
                edge_list = list(zip(path[:-1], path[1:]))
                # Remove all edges but the first (that edge that connects the house with its provider)
                for edge in edge_list[1:]:
                    G.remove_edge(edge[0], edge[1])
            except:
                continue



    #consumers = Consumers(Constants.num_blocks, [25] * Constants.num_blocks)
    # Same number of blocks as providers
    Constants.num_producers = Constants.num_blocks = len(providers)
    consumers = Consumers(len(providers),
                          [len(nx.node_connected_component(G, i)) for i in providers],
                          list(distances.items()))

    # Compute final price of the energy (will update the demand too)
    price = Cost.optimal_demand_response(consumers)

    # for i in range(b.size):
    #     b[i].plot_demand()

    Cost.plot_price(price)
    # #Cost.plot_price(house_price)
    # Battery.plot_battery(Cost.price)

    consumers.compute_total_measures()

    np.savetxt("expenditures.txt", consumers.total_expenditures)
    np.savetxt("demand.txt", consumers.total_demand)

if __name__ == "__main__":
    main()
