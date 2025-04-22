import hydra
import os
import pickle
from syntheseus.search.visualization import visualize_andor
from pathlib import Path
from syntheseus.search.analysis.route_extraction import (
    iter_routes_time_order,
)

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[0]

@hydra.main(config_path='./configs', config_name='config.yaml')
def visualize(config):
    print(config)
    experiment_path = os.path.join(PROJECT_ROOT, 
                          'experiments', 
                          'test1_root_aligned_callsLimit100_timeLimit600',
                          'graphs')
    output_graph_path = os.path.join(experiment_path, 'output_graph.pkl')
    output_graph = pickle.load(open(output_graph_path, 'rb'))

    routes = list(iter_routes_time_order(output_graph, max_routes=10))
    print(f'found {len(routes)} routes')

    routes_output_dir = os.path.join(experiment_path, 'routes')
    os.makedirs(routes_output_dir, exist_ok=True)

    for idx, route in enumerate(routes):
        visualize_andor(
            output_graph, filename=os.path.join(routes_output_dir, f"route_{idx + 1}.pdf"), nodes=route
        )

if __name__ == "__main__":
    visualize()