#!/usr/bin/env python
import numpy as np
import networkx as nx

from constants import Constants
from cost import Cost
from consumers import Consumers
from topology import Topology

__author__ = 'jorge'


def main():
    seed = 18765
    np.random.seed(seed)

    # Generate graph with 100 nodes (choose one of them)
    G = nx.barabasi_albert_graph(100, m=1, seed=seed)
    # G = nx.powerlaw_cluster_graph(100, 1, 0, seed)
    # G = nx.random_powerlaw_tree(100, 3, seed)
    topo = Topology(G)
    # Analyze graph and get providers
    providers = topo.get_providers()
    # For each block, get those providers from which it can get the energy
    # (those providers that are at minimum distance)
    available_providers = topo.get_available_providers()
    # Isolate providers so that houses attached to each of them can be easily computed
    topo.isolate_providers()

    #consumers = Consumers(Constants.num_blocks, [25] * Constants.num_blocks)
    # Same number of blocks as providers
    Constants.num_producers = Constants.num_blocks = len(providers)
    consumers = Consumers(len(providers),
                          [len(nx.node_connected_component(topo.G, i)) for i in providers],
                          list(available_providers.items()))

    # Compute final price of the energy (will update the demand too)
    price = Cost.optimal_demand_response(consumers)

    Cost.plot_price(price)
    # Cost.plot_price(house_price)
    # Battery.plot_battery(Cost.price)

    consumers.compute_total_measures()

    np.savetxt("expenditures.txt", consumers.total_expenditures)
    np.savetxt("demand.txt", consumers.total_demand)

if __name__ == "__main__":
    main()
