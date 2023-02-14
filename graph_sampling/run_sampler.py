from .curvature_aware_sampler import CurvatureSampler
from .graph_curvature import OllivierRicci, FormanRicci, BalancedFormanRicci
import networkx as nx
import argparse
import numpy as np

CURVATURES_DICT = {
    "ollivier":OllivierRicci,
    "forman":FormanRicci,
    "balanced_forman":BalancedFormanRicci,
    "balanced":BalancedFormanRicci
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_runs", default=1, type=int)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--curvature", default="ollivier")
    parser.add_argument("-f", "fractions", nargs="+", type=float, required=True)
    parser.add_argument("--pin", type=float, required=True)
    parser.add_argument("--pout", type=float, required=True)
    parser.add_argument("-s", "--sample_frac", required=True, type=float)

    args = parser.parse_args()

    curvature = CURVATURES_DICT[args.curvature]
    sizes = [int(args.nodes * f) for f in args.fractions]
    probs = np.eye(args.nodes) * (args.pin - args.pout) + args.pout

    for i in range(args.n):
        G = nx.stochastic_block_model(sizes, probs)
        sampler = CurvatureSampler(curvature=curvature, number_of_nodes=args.nodes)
        final_subgraph, sampled_nodes, sampled_edges = sampler.sample(G)
        


if __name__ == "__main__":
    main()