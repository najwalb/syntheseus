import os
from syntheseus import Molecule
from syntheseus.search.algorithms.best_first import retro_star
from syntheseus.search.graph.and_or import AndNode, OrNode, AndOrGraph
from syntheseus.search.node_evaluation.common import ReactionModelLogProbCost
from syntheseus.reaction_prediction.inference import RootAlignedModel, LocalRetroModel
from syntheseus.search.mol_inventory import SmilesListInventory
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator
from syntheseus.search.algorithms.breadth_first import (
    AndOr_BreadthFirstSearch
)
from syntheseus.search.analysis.route_extraction import (
    iter_routes_time_order,
)
from syntheseus.search.graph.and_or import AndNode
from syntheseus.search.visualization import visualize_andor

# Some constants for all algorithms
RXN_MODEL_CALL_LIMIT = 1
TIME_LIMIT_S = 3

if __name__ == "__main__":
    # Set up a reaction model with caching enabled. Number of reactions
    # to request from the model at each step of the search needs to be
    # provided at construction time.
    test_mol = Molecule("Cc1ccc(-c2ccc(C)cc2)cc1")
    #model = LocalRetroModel(use_cache=True, default_num_results=10)
    model = RootAlignedModel(use_cache=True, default_num_results=10, with_classifier_guidance=True)
    print(f'model: {model}')

    # Dummy inventory with just two purchasable molecules.
    inventory = SmilesListInventory(
        smiles_list=["Cc1ccc(B(O)O)cc1", "O=Cc1ccc(I)cc1"]
    )

    # 1: OrNode cost function.
    # We will follow the original paper and give molecules a
    # cost of 0 if they are purchasable, and a cost of infinity
    # otherwise. This class is provided as a default in retro_star.
    # If purchasable molecules have non-zero costs then a different
    # cost function could be used.
    or_node_cost_fn = retro_star.MolIsPurchasableCost()

    # 2: AndNode cost function
    # We will follow the original paper and define the cost of the
    # reaction as the -log(softmax) of the reaction model output,
    # thresholded at a minimum value. We use the built-in
    # `ReactionModelLogProbCost` class for this. This class simply
    # reads out the "probability" value from `reaction.metadata`,
    # which is provided by the PaRoutesModel.

    and_node_cost_fn = ReactionModelLogProbCost(normalize=False)

    # 3: search heuristic (value function)
    # Here we just use a constant value function which is always 0,
    # corresponding to the "retro*-0" algorithm (the most optimistic).
    retro_star_value_function = ConstantNodeEvaluator(0.0)
        
    search_algorithm = retro_star.RetroStarSearch(
        reaction_model=model,
        mol_inventory=inventory,
        or_node_cost_fn=or_node_cost_fn,
        and_node_cost_fn=and_node_cost_fn,
        value_function=retro_star_value_function,
        limit_reaction_model_calls=RXN_MODEL_CALL_LIMIT,
        time_limit_s=TIME_LIMIT_S,
    )

    # search_algorithm = AndOr_BreadthFirstSearch(
    #     reaction_model=model,
    #     mol_inventory=inventory,
    #     limit_iterations=100,  # max number of algorithm iterations
    #     limit_reaction_model_calls=100,  # max number of model calls
    #     time_limit_s=60.0  # max runtime in seconds
    # )
    search_algorithm.reset()
    output_graph, _ = search_algorithm.run_from_mol(test_mol)
    print(f"Explored {len(output_graph)} nodes")

    # # Extract the routes simply in the order they were found.
    routes = list(iter_routes_time_order(output_graph, max_routes=10))
    print(f'found {len(routes)} routes')

    for idx, route in enumerate(routes):
        num_reactions = len({n for n in route if isinstance(n, AndNode)})
        print(f"Route {idx + 1} consists of {num_reactions} reactions")

    for idx, route in enumerate(routes):
        visualize_andor(
            output_graph, filename=os.path.join('experiments', f"route_{idx + 1}.pdf"), nodes=route
        )