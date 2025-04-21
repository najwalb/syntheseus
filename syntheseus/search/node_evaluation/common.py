"""Common node evaluation functions."""

from __future__ import annotations

from typing import Sequence, Union
import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem import AllChem
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.search.graph.and_or import AndNode
from syntheseus.search.graph.molset import MolSetNode
from syntheseus.search.node_evaluation.base import NoCacheNodeEvaluator, ReactionModelBasedEvaluator

supp_act = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "swish": nn.SiLU(),
}

def smiles_to_fp(smiles, fp_size=2048):
    """
    Convert a SMILES string to a fingerprint.
    Args:
        smiles (str): SMILES string
    Returns:
        np.array: fingerprint of the SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=fp_size, useChirality=True
    )
    fp = torch.tensor(fp, dtype=torch.uint8)
    return fp

class CustomLoss(nn.Module):
    def __init__(self, max_label):
        super().__init__()
        self.max_label = max_label

    def forward(self, logits, target):
        # if target <= self.max_label, return MSELoss(logits, target)
        # otherwise, return (max(0, self.max_label - logits + 1))^2
        loss = torch.mean(
            torch.where(
                target <= self.max_label,
                (logits - target) ** 2,
                torch.max(torch.zeros_like(logits), self.max_label - logits + 1) ** 2,
            )
        )
        return loss
    
class Dense(nn.Module):
    """
    Dense layer with activation function.

    Args:
        in_features (int): input feature size
        out_features (int): output feature size
        hidden_act (nn.Module): activation function (e.g. nn.ReLU())
    """

    def __init__(self, in_features, out_features, hidden_act):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.hidden_act = hidden_act

    def forward(self, x):
        return self.hidden_act(self.linear(x))

    
class SyntheticDistance(nn.Module):
    """
    Synthetic distance model. On input it takes either concatenated
    fingerprints (starting (+) target) or difference fingerprint (target - starting)
    and outputs the predicted distance.

    Args:
        input_type (str): input type (e.g. "concat", "diff")
        max_label (int): maximum label value
        fp_size (int): fingerprint size
        output_dim (int): output dimension
        hidden_sizes (str): string representnig list of hidden layer sizes (e.g. '1024,1024')
        hidden_activation (str): activation function (e.g. "relu")
        dropout (float): dropout probability (e.g. 0.3)
    """

    def __init__(self, args):
        super().__init__()
        self.hidden_sizes = [int(size) for size in args.hidden_sizes.split(",")]
        if args.model_type == "retro":
            self.criterion = nn.MSELoss()
        elif args.model_type == "dist":
            self.criterion = CustomLoss(args.max_label)
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")
        if args.input_type == "concat":
            input_dim = args.fp_size * 2
        elif args.input_type == "diff":
            input_dim = args.fp_size
        else:
            raise ValueError(f"Unsupported input type: {args.input_type}")
        self.hidden_activation = supp_act[args.hidden_activation]
        self.layers = self._build_layers(
            input_dim, self.hidden_sizes, self.hidden_activation
        )
        self.output_layer = nn.Linear(self.hidden_sizes[-1], args.output_dim, bias=True)
        self.dropout = nn.Dropout(args.dropout)

    def _build_layers(self, fp_size, hidden_sizes, hidden_activation):
        layers = nn.ModuleList(
            [Dense(fp_size, hidden_sizes[0], hidden_act=hidden_activation)]
        )

        for layer_i in range(len(hidden_sizes) - 1):
            in_features = hidden_sizes[layer_i]
            out_features = hidden_sizes[layer_i + 1]
            layer = Dense(in_features, out_features, hidden_act=hidden_activation)
            layers.append(layer)

        return layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)

        return self.output_layer(x)

    def get_loss(self, logits, target):
        loss = self.criterion(logits, target.float())
        return loss


class ValueNodeEvaluator(NoCacheNodeEvaluator):
    def __init__(self, value_model_path: str, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # define model
        checkpoint = torch.load(value_model_path, map_location="cpu", weights_only=False)
        pretrain_args = checkpoint["args"]
        pretrain_args.output_dim = 1
        self.model = SyntheticDistance(pretrain_args).to(self.device)
        # Load the checkpoint
        state_dict = checkpoint["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("Loaded retro value model!")

    def _evaluate_nodes(self, nodes, graph=None):
        # get model estimates for each node
        return [self.predict(node.reaction.product) for node in nodes]

    def predict(self, target, as_item=True):
        """
        Predict the synthetic cost of 'target'.

        Args:
            target (str): target molecule SMILES

        Returns:
            float: synthetic distance
        """
        target_fp = smiles_to_fp(target, fp_size=2048).float().unsqueeze(0)
        with torch.no_grad():
            dist = self.model(target_fp)
        if as_item:
            return dist.item()
        else:
            return dist

class ConstantNodeEvaluator(NoCacheNodeEvaluator):
    def __init__(self, constant: float, **kwargs):
        super().__init__(**kwargs)
        self.constant = constant

    def _evaluate_nodes(self, nodes, graph=None):
        return [self.constant] * len(nodes)


class HasSolutionValueFunction(NoCacheNodeEvaluator):
    def _evaluate_nodes(self, nodes, graph=None):
        return [float(n.has_solution) for n in nodes]


class ReactionModelLogProbCost(ReactionModelBasedEvaluator[AndNode]):
    """Evaluator that uses the reactions' negative logprob to form a cost (useful for Retro*)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(return_log=True, **kwargs)

    def _get_reaction(self, node: AndNode, graph) -> SingleProductReaction:
        return node.reaction

    def _evaluate_nodes(self, nodes, graph=None) -> Sequence[float]:
        return [-v for v in super()._evaluate_nodes(nodes, graph)]


class ReactionModelProbPolicy(ReactionModelBasedEvaluator[Union[MolSetNode, AndNode]]):
    """Evaluator that uses the reactions' probability to form a policy (useful for MCTS)."""

    def __init__(self, **kwargs) -> None:
        kwargs["normalize"] = kwargs.get("normalize", True)  # set `normalize = True` by default
        super().__init__(return_log=False, **kwargs)

    def _get_reaction(self, node: Union[MolSetNode, AndNode], graph) -> SingleProductReaction:
        if isinstance(node, MolSetNode):
            parents = list(graph.predecessors(node))
            assert len(parents) == 1, "Graph must be a tree"
            return graph._graph.edges[parents[0], node]["reaction"]
        elif isinstance(node, AndNode):
            return node.reaction
        else:
            raise ValueError(f"ReactionModelProbPolicy does not support nodes of type {type(node)}")
