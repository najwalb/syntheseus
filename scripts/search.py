import os
from syntheseus import Molecule
from syntheseus.reaction_prediction.inference import RootAlignedModel, LocalRetroModel
from syntheseus.search.mol_inventory import SmilesListInventory
from syntheseus.search.algorithms.breadth_first import (
    AndOr_BreadthFirstSearch
)
from syntheseus.search.analysis.route_extraction import (
    iter_routes_time_order,
)
from syntheseus.search.graph.and_or import AndNode
from syntheseus.search.visualization import visualize_andor


if __name__ == "__main__":
    # Set up a reaction model with caching enabled. Number of reactions
    # to request from the model at each step of the search needs to be
    # provided at construction time.
    test_mol = Molecule("Cc1ccc(-c2ccc(C)cc2)cc1")
    model = LocalRetroModel(use_cache=True, default_num_results=10)
    print(f'model: {model}')

    # Dummy inventory with just two purchasable molecules.
    inventory = SmilesListInventory(
        smiles_list=["Cc1ccc(B(O)O)cc1", "O=Cc1ccc(I)cc1"]
    )

    search_algorithm = AndOr_BreadthFirstSearch(
        reaction_model=model,
        mol_inventory=inventory,
        limit_iterations=100,  # max number of algorithm iterations
        limit_reaction_model_calls=100,  # max number of model calls
        time_limit_s=60.0  # max runtime in seconds
    )

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