"""Inference wrapper for the RootAligned model.

Paper: https://arxiv.org/abs/2203.11444
Code: https://github.com/otori-bird/retrosynthesis

The original RootAligned code is released under the MIT license.
Parts of this file are based on code from the GitHub repository above.
"""

import argparse
import math
import multiprocessing
import random
import warnings
import torch
import torch.nn as nn
import os
from collections import defaultdict
from typing import Any, List, Optional, Sequence
from functools import partial
import yaml
from rdkit import Chem

from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import ReactionMetaData, SingleProductReaction
from syntheseus.reaction_prediction.inference.base import ExternalBackwardReactionModel
from syntheseus.reaction_prediction.utils.inference import (
    get_unique_file_in_dir,
    process_raw_smiles_outputs_backwards,
)

# class AtomCountClassifier(nn.Module):
#     def __init__(self, a, b, sigma):
#         super().__init__()
#         self.a = a
#         self.b = b
#         self.sigma = sigma

#     def forward(self, *args, **kwargs):
#         out = torch.sigmoid(((args[0] - self.a) / self.b)**self.sigma)
#         return out
    
    
class AtomCountClassifier(nn.Module):
    def __init__(self, target_size=20, scale=5.0, gamma=2.0, tokenizer=None):
        """
        A simple heuristic for estimating if a molecule will be small based on current sequence.
        
        Args:
            target_size: Target number of atoms (smaller values prefer smaller molecules)
            scale: Scaling factor for the sigmoid function
            gamma: Sharpness parameter for the sigmoid
        """
        super().__init__()
        self.target_size = target_size
        self.scale = scale
        self.gamma = gamma
        self.tokenizer = tokenizer

    # def forward(self, *args, **kwargs):
    #     return 1.
        
    def forward(self, partial_seq):
        """
        Apply a heuristic to predict if the molecule will be small.
        
        Args:
            input_ids: The current token sequence
            tokenizer: The tokenizer for decoding (if needed)
            
        Returns:
            Scores favoring smaller molecules
        """
        # 1. Simple token counting strategy
        # Count tokens that likely indicate atoms/bonds
        
        # Estimate current size and projected final size
        input_ids = partial_seq
        seq_length = input_ids.shape[1]
        
        # Heuristic 1: Use sequence length directly
        # The longer the sequence, the larger the molecule is likely to be
        estimated_size = seq_length
        
        # Heuristic 2: If you have a way to count atoms specifically in your tokens
        # This is a placeholder - adjust based on your token patterns
        if self.tokenizer:
            # Count tokens representing atoms (this is highly tokenizer-specific)
            # Example implementation - adapt to your tokenization scheme
            atom_tokens = set([self.tokenizer.convert_tokens_to_ids(t) for t in ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'P']])
            
            # Count atom tokens in sequence
            atom_count = sum(1 for t in input_ids[0].tolist() if t in atom_tokens)
            estimated_size = atom_count
        
        # Calculate how far we are from target size
        # Positive values mean we're below target (good)
        # Negative values mean we're above target (bad)
        size_difference = self.target_size - estimated_size
        
        # Convert to probability using sigmoid
        # Higher probability = more likely to be a small molecule
        probability = torch.sigmoid(self.gamma * torch.tensor(size_difference) / self.scale)
        
        # Ensure tensor shape and type
        return torch.tensor(probability, device=input_ids.device).view(-1, 1)
    
# Create a wrapper for the generator that adds your classifier guidance
class ClassifierGuidedGenerator(nn.Module):
    def __init__(self, original_generator, translator, src, classifier_model, guidance_scale=1.0):
        super().__init__()
        self.original_generator = original_generator
        self.translator = translator
        self.classifier_model = classifier_model
        self.guidance_scale = guidance_scale
        self.src = src

    def forward(self, *args, **kwargs):
        # Get the original scores from the translator's generator
        # TODO: what input goes here? need partially completed sequence
        original_scores = self.original_generator(*args, **kwargs)

        # Apply your classifier guidance/bias
        # The exact implementation depends on your classifier and how you want to combine scores
        # HACK: need to choose topn candidates here to evaluate for the classifier, and from which we choose topk for the beam search
        # choose topn candidates to evaluate for the classifier
        classifier_scores = self.classifier_model(self.translator.alive_seq)  # or whatever inputs your classifier needs


        # HACK: need to choose topn candidates here to evaluate for the classifier, and from which we choose topk for the beam search
        
        # Combine scores (e.g., add, multiply, etc.)
        # This is similar to your pseudocode: original_scores + my_own_bias
        modified_scores = self.combine_scores(original_scores, classifier_scores)
        
        return modified_scores
    
    def combine_scores(self, original_scores, classifier_scores):
        # Implement your score combination logic
        # Could be simple addition with a weight parameter
        # return original_scores + (lambda_weight * classifier_scores)
        # Or more complex combinations
        combined_log_probs = original_scores + self.guidance_scale * classifier_scores  # Simplified example
        log_sum = torch.logsumexp(combined_log_probs, dim=-1, keepdim=True)
        normalized_log_probs = combined_log_probs - log_sum
        return normalized_log_probs

class RootAlignedModel(ExternalBackwardReactionModel):
    def __init__(
        self,
        *args,
        num_augmentations: int = 20,
        probability_from_score_temperature: Optional[float] = 2.0,
        with_classifier_guidance: bool = False,
        **kwargs,
    ) -> None:
        """Initializes the RootAligned model wrapper.

        Assumed format of the model directory:
        - `model_dir` contains the model checkpoint as the only `*.pt` file
        - `model_dir` contains the config as the only `*.yml` file
        """
        super().__init__(*args, **kwargs)

        #Parse arguments for calling external functions from `root_aligned/OpenNMT.py`
        config_file_path = get_unique_file_in_dir(self.model_dir, pattern="*.yml")
        print(f'======= config_file_path: {config_file_path}')
        with open(config_file_path, "r") as f:
            opt_from_config = yaml.safe_load(f)

        import torch

        opt = argparse.Namespace()
        for key, value in opt_from_config.items():
            setattr(opt, key, value)
        opt.models = [get_unique_file_in_dir(self.model_dir, pattern="*.pt")]
        print(f'======= opt.models: {opt.models}')
        opt.output = "/dev/null"
        print(f'========= self.device: {self.device}')
        opt.gpu = -1 if self.device == "cpu" else torch.device(self.device).index

        setattr(opt, "synthon", False)

        self.opt = opt

        from root_aligned import score

        score.opt = opt

        # Import external functions from `root_aligned/OpenNMT.py`.
        from onmt.translate.translator import build_translator
        from onmt.utils.parse import ArgumentParser

        ArgumentParser.validate_translate_opts(opt)

        self.translator = build_translator(opt, report_score=False)
        self.num_augmentations = num_augmentations
        self.probability_from_score_temperature = probability_from_score_temperature
        self.beam_size = opt.beam_size
        self.with_classifier_guidance = with_classifier_guidance
        if self.with_classifier_guidance:
            self.classifier = AtomCountClassifier()

    def get_parameters(self):
        """Return the model parameters."""
        return self.translator.model.parameters()

    def _mols_to_batch(self, inputs) -> List[bytes]:
        """Map `Molecule`s into SMILES bytes."""
        from root_aligned.score import smi_tokenizer

        # Example outcome: b'C C ( = O ) c 1 c c c 2 c ( c c n 2 C ( = O ) O C ( C ) ( C ) C ) c 1\n'.
        return [bytes(smi_tokenizer(input.smiles) + "\n", "utf-8") for input in inputs]

    def _build_kwargs_from_scores(self, scores: List[float]) -> List[ReactionMetaData]:
        """Compute kwargs to save in the predictions given raw scores from the RootAligned model.

        The scores we get from the model cannot be directly interpreted as a (log) probability.
        In general, the model produces an array of `[num_augmentations, beam_size]` predictions, and
        computes the score of a given prediction `p` as `total_rr - best_pos * 1e8` where:
            `total_rr = sum_{(j, k) such that prediction[j, k] = p} (1 / (k + 1))`
            `best_pos = min_{(j, k) such that prediction[j, k] = p} k`
        In other words, the predictions are ranked first on how early they appear during beam search
        for any augmentation (`best_pos`), with ties broken by how much they repeat (`total_rr`).
        See `compute_rank` in the RootAlign repository for details.

        This function recovers `total_rr` and `best_pos`, puts them into metadata, and also computes
        a different combined score which is supposed to be better-behaved (smaller range of values).
        """
        import torch

        # Make sure the scores are sorted as expected.
        for score, next_score in zip(scores, scores[1:]):
            assert score >= next_score

        # Maximum possible value `total_rr` could have.
        max_possible_total_rr = self.num_augmentations * sum(
            1.0 / (k + 1) for k in range(self.beam_size)
        )

        kwargs_list: List[ReactionMetaData] = []
        for score in scores:
            best_pos = -math.floor(score / 1e8)
            total_rr = score + best_pos * 1e8

            assert 0 <= best_pos < self.beam_size
            assert 0.0 < total_rr <= max_possible_total_rr

            new_score = total_rr - (best_pos + 1) * max_possible_total_rr
            assert new_score <= 0.0

            kwargs_list.append(
                {  # type: ignore[typeddict-unknown-key]
                    "original_score": score,
                    "best_pos": best_pos,
                    "total_rr": total_rr,
                    "score": new_score,
                }
            )

        # Make sure the new scores produce the same ranking.
        for kwargs, next_kwargs in zip(kwargs_list, kwargs_list[1:]):
            assert kwargs["score"] >= next_kwargs["score"]

        if self.probability_from_score_temperature is not None:
            scaled_scores = [
                self.probability_from_score_temperature * kwargs["score"] / max_possible_total_rr
                for kwargs in kwargs_list
            ]
            probabilities = torch.nn.functional.softmax(torch.as_tensor(scaled_scores), dim=-1)

            for kwargs, probability in zip(kwargs_list, probabilities.numpy().tolist()):
                kwargs["probability"] = probability

        return kwargs_list

    # Create a custom translation function with guidance
    def translate_with_classifier_guidance(self,
                                           model,
                                           src,
                                           property='synthesizability',
                                           guidance_scale=1.0):
        # Set to eval mode but enable gradients
        model.eval()
        batch_size = len(src)

        # Save the original generator
        original_generator = self.translator.model.generator

        # Create your wrapper with your classifier
        guided_generator = ClassifierGuidedGenerator(original_generator, self.translator, src, self.classifier)

        # Replace the generator in the translator
        self.translator.model.generator = guided_generator

        # Now when you call translate(), it will use your guided probabilities
        translations = self.translator.translate(src, batch_size=batch_size)

        # You can always restore the original generator if needed
        # translator.generator = original_generator

        return translations


    def _get_reactions(
        self, inputs, num_results: int, random_augmentation=False
    ) -> List[Sequence[SingleProductReaction]]:
        # Step 1: Perform data augmentation.
        augmented_inputs = []
        if random_augmentation:
            for input in inputs:
                augmented_inputs.append(input)
                for i in range(self.num_augmentations - 1):
                    randomized_smi = Chem.MolToSmiles(input.rdkit_mol, doRandom=True)
                    randomized_mol = Molecule(smiles=randomized_smi, canonicalize=False)
                    augmented_inputs.append(randomized_mol)
        else:
            from root_aligned.preprocessing.generate_PtoR_data import clear_map_canonical_smiles

            for input in inputs:
                product_atom_map_numbers = [i + 1 for i in range(input.rdkit_mol.GetNumAtoms())]
                max_times = len(product_atom_map_numbers)
                product_roots = [-1]
                times = min(self.num_augmentations, max_times)
                if times < self.num_augmentations:  # times = max_times
                    product_roots.extend(product_atom_map_numbers)
                    product_roots.extend(
                        random.choices(product_roots, k=self.num_augmentations - len(product_roots))
                    )
                else:  # times = num_augmentations
                    while len(product_roots) < times:
                        product_roots.append(random.sample(product_atom_map_numbers, 1)[0])
                        if product_roots[-1] in product_roots[:-1]:
                            product_roots.pop()
                times = len(product_roots)
                assert times == self.num_augmentations
                for k in range(times):
                    pro_root_atom_map = product_roots[k]
                    pro_root = pro_root_atom_map - 1
                    if pro_root_atom_map <= 0:
                        pro_root = -1
                    pro_smi = clear_map_canonical_smiles(
                        input.smiles, canonical=True, root=pro_root
                    )
                    randomized_mol = Molecule(smiles=pro_smi, canonicalize=False)
                    augmented_inputs.append(randomized_mol)

        assert len(augmented_inputs) == len(inputs) * self.num_augmentations

        # Step 2: Map from `Molecule`s to SMILES bytes to align with `root_aligned/OpenNMT.py`.
        augmented_batch = self._mols_to_batch(augmented_inputs)

        # Step 3: Translate.
        print(f'self.with_classifier_guidance: {self.with_classifier_guidance}')
        if self.with_classifier_guidance:
            _, augmented_predictions = self.translate_with_classifier_guidance(model=self.translator.model,
                                                                            src=augmented_batch,
                                                                            guidance_scale=1.0)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="__floordiv__ is deprecated")

                _, augmented_predictions = self.translator.translate(
                    src=augmented_batch,
                    src_feats=defaultdict(list),
                    tgt=None,
                    batch_size=2048,
                    batch_type="tokens",
                    attn_debug=False,
                    align_debug=False,
                )  # shape: `[data_size x augmentation_size, beam_size]`

        # Step 4: Unravel and canonicalize.
        lines = []  # shape: `[data_size x augmentation_size x beam_size]`
        for i in range(len(augmented_predictions)):
            for j in range(len(augmented_predictions[i])):
                lines.append(augmented_predictions[i][j].replace(" ", ""))

        from root_aligned.score import canonicalize_smiles_clear_map

        raw_predictions = []
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        wrapper = partial(canonicalize_smiles_clear_map, opt=self.opt)
        raw_predictions = pool.map(
            func=wrapper, iterable=lines
        )  # Canonicalize reactants and modify illegal reactants into empty strings.
        pool.close()
        pool.join()

        # From `[data_size x augmentation_size x beam_size]` to `[data_size, augmentation_size, beam_size]`.
        predictions: List[List[Any]] = [
            [[] for _ in range(self.num_augmentations)] for _ in range(len(inputs))
        ]

        for i, prediction in enumerate(raw_predictions):
            predictions[i // (self.beam_size * self.num_augmentations)][
                i % (self.beam_size * self.num_augmentations) // self.beam_size
            ].append(prediction)

        # Step 5: Rank legal reactants from all augmentations and beams.
        ranked_results = []  # shape: `[data_size, augmentation_size x beam_size]`
        ranked_scores = []

        from root_aligned.score import compute_rank

        for i in range(len(predictions)):
            rank, _ = compute_rank(predictions[i], opt=self.opt)
            rank = list(zip(rank.keys(), rank.values()))
            rank.sort(key=lambda x: x[1], reverse=True)
            rank = rank[:num_results]  # Truncate to `num_results` results.
            ranked_results.append([item[0][0] for item in rank])  # Output reactant SMILES.
            ranked_scores.append([item[1] for item in rank])  # Output scores used for ranking.

        return [
            process_raw_smiles_outputs_backwards(
                input, outputs, self._build_kwargs_from_scores(scores)
            )
            for input, outputs, scores in zip(inputs, ranked_results, ranked_scores)
        ]
