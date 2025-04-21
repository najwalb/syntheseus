# Basic experiment
- theoretical questions:
    - bestfirst: 
        - uses a priority queue to guarantee best node is always first
            - requires updating oder upon insertion/deletion
    - understand best first algorithm and the types of value functions proposed by different works
        - + what models they use
    - is it possible to guide other types of algorithms?
    - how well do other algorithms perform as they are?
        - compute solve rate
    - what kind of filtering does syntheseus do my default (e.g. do we need to factor in the probability of duplicate molecules etc)
- add hydra to be able to run experiments
- get basic results with rsmiles + correct checkpoint + multiple routes
    - save routes
    - evaluate diversity of routes
    - include metrics to evaluate the molecules making up the routes + average route metrics
        - NLL/synthesizability
        - toxicity
        - SAScore/logS etc

# Guidance experiment
- make a new rsmiles that can be guided
    - using custom model wrapper
        - careful with what changed in the rsmiles implementation => maybe can use my own directly here
    - add synthesizability model to the weights before sampling
- see what other baselines to guide and how
- see how to add filtering step to use as a baseline too