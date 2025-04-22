import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="onmt")
import hydra
import time
import pickle
import pandas as pd
from syntheseus import Molecule
from pathlib import Path
from syntheseus.search.algorithms.best_first import retro_star
from syntheseus.search.graph.and_or import AndNode, OrNode, AndOrGraph
from syntheseus.search.node_evaluation.common import ReactionModelLogProbCost
from syntheseus.reaction_prediction.inference import RootAlignedModel, LocalRetroModel
from syntheseus.search.mol_inventory import SmilesListInventory
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator, ValueNodeEvaluator
from syntheseus.search.algorithms.breadth_first import (
    AndOr_BreadthFirstSearch
)
from syntheseus.search.analysis.route_extraction import (
    iter_routes_time_order,
)
from syntheseus.search.graph.and_or import AndNode
from syntheseus.search.visualization import visualize_andor

import logging
# Configure the root logger
logging.basicConfig(
    level=logging.DEBUG - 1,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Some constants for all algorithms
RXN_MODEL_CALL_LIMIT = 100 # 100
TIME_LIMIT_S = 600 # 300
PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[1]

@hydra.main(config_path='./configs', config_name='config.yaml')
def main(config):
    # Set up a reaction model with caching enabled. Number of reactions
    # to request from the model at each step of the search needs to be
    # provided at construction time.
    smi = 'OCCN1CCC(c2ccc3c(c2)-n2nc(-c4ncnn4CC(F)(F)F)cc2CCO3)CC1'
    test_mol = Molecule(smi)
    # model = LocalRetroModel(use_cache=True, default_num_results=10)
    model = RootAlignedModel(use_cache=True, 
                             default_num_results=100,# 10
                             model_dir=os.path.join(PROJECT_ROOT,
                                                    'scripts',
                                                    'rsmiles_full_no_overlap_checkpoints'))

    # Dummy inventory with just two purchasable molecules.
    print(f'======= creating inventory')
    start_time = time.time()
    inventory_path = os.path.join(PROJECT_ROOT,
                                  'scripts',
                                  'data',
                                  'desp_data',
                                  'inventory.pkl')
    if os.path.exists(inventory_path):
        with open(inventory_path, 'rb') as f:
            inventory = pickle.load(f)
    else:
        print(f'======= loading bbs')
        start_time = time.time()
        bb_mol2idx = os.path.join(PROJECT_ROOT,
                                'scripts',
                                'data',
                                'desp_data',
                                'origin_dict.csv')
        df = pd.read_csv(bb_mol2idx, index_col=0)
        bbs = df['mol'].tolist()
        print(f'loaded bbs in {time.time() - start_time} seconds')
        inventory = SmilesListInventory(
            smiles_list=bbs,
            print_every=1000000
        )
        # save inventory in a pickle file
        with open(inventory_path, 'wb') as f:
            pickle.dump(inventory, f)

    print(f'created inventory in {time.time() - start_time} seconds')
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
    #retro_star_value_function = ConstantNodeEvaluator(0.0)
    retro_star_value_function = ValueNodeEvaluator(
                                    value_model_path=os.path.join(PROJECT_ROOT,
                                      'scripts',
                                      'data',
                                      'desp_data',
                                      'retro_value.pt')
                                )
        
    print(f'retro_star_value_function {retro_star_value_function}\n')
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
    start_time = time.time()
    print(f'======= running search')
    output_graph, _ = search_algorithm.run_from_mol(test_mol)
    print(f"Explored {len(output_graph)} nodes in {time.time() - start_time} seconds.")

    # save output graph 
    print(f'======= saving output graph')
    start_time = time.time()
    output_graph_dir = os.path.join(PROJECT_ROOT,
                                     'scripts',
                                     'experiments',
                                     f'retroStar_root_aligned_callsLimit{RXN_MODEL_CALL_LIMIT}_timeLimit{TIME_LIMIT_S}', # experiment subfolder
                                     'graphs')
    os.makedirs(output_graph_dir, exist_ok=True)
    output_graph_path = os.path.join(output_graph_dir,
                                     f'output_graph.pkl')
    with open(output_graph_path, 'wb') as f:
        pickle.dump(output_graph, f)
    print(f'======= saved output graph to {output_graph_path} in {time.time() - start_time} seconds')
    
    # # Extract the routes simply in the order they were found.
    print(f'======= extracting routes')
    start_time = time.time()
    routes = list(iter_routes_time_order(output_graph, max_routes=100))
    print(f'Extracted {len(routes)} routes in {time.time() - start_time} seconds')

    for idx, route in enumerate(routes):
        num_reactions = len({n for n in route if isinstance(n, AndNode)})
        print(f"Route {idx + 1} consists of {num_reactions} reactions")

    # for idx, route in enumerate(routes):
    #     visualize_andor(
    #         output_graph, filename=os.path.join('experiments', f"route_{idx + 1}.pdf"), nodes=route
    #     )

if __name__ == "__main__":
    main()