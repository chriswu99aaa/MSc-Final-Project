# MSc-Final-Project


### Run

set cfg/config.yaml and run main.py for heuristic evaluations.

**If you want to run several evaluations simultaneously, please create multiple environments, including copied problems and prompts, and a copied cfg file in cfg/problems with changing the problem name** (See tsp_constructive_copy for reference)

### Report Runs

We also provide all the reported runs of BOTree-AHD in outputs/. The runs under the step-by-step construction framework are in [Google Drive](https://drive.google.com/file/d/1mWBiWwi4u9FBMXVxOTfZVuvdMSrX50af/view?usp=sharing) and BOTree-AHD runs under the ACO framework is in [Google Drive](https://drive.google.com/file/d/1UhiSlNP6crQvtZfeNEXGFWTI0B1e2yq-/view?usp=sharing).

Moreover, the ``gpt.py`` for each problem in this repository contains a leading heuristic function designed by MCTS-AHD.

### Acknowledge

Thanks to the implementations of [EoH](https://github.com/FeiLiu36/EoH) and [ReEvo](https://github.com/ai4co/reevo), and [MCTS-AHD](https://github.com/zz1358m/MCTS-AHD-master)

