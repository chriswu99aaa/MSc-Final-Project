from source.mcts_ahd import MCTS_AHD
from source.getParas import Paras
from source import prob_rank, pop_greedy
from problem_adapter import Problem

from utils.utils import init_client

from source.bo import BayesianOptimizer
from source.bo_local import BayesianOptimizer_Local
from source.bo_baseline import BayesianOptimizer_Baseline
class AHD:
    def __init__(self, cfg, root_dir, workdir, client) -> None:
        self.cfg = cfg
        self.root_dir = root_dir
        self.problem = Problem(cfg, root_dir)

        self.paras = Paras() 
        # self.paras.set_paras(method = "mcts_ahd",
        #                      init_size = self.cfg.init_pop_size,
        #                      pop_size = self.cfg.pop_size,
        #                      llm_model = client,
        #                      ec_fe_max = self.cfg.max_fe,
        #                      exp_output_path = f"{workdir}/",
        #                     exp_debug_mode = False,
        #                      eva_timeout=cfg.timeout)
        self.paras.set_paras(method = "bo_ahd",
                            n_init = self.cfg.n_init,
                             n_iters = self.cfg.n_iters,

                             llm_model = client,
                             ec_fe_max = self.cfg.max_fe,
                             exp_output_path = f"{workdir}/",
                             exp_debug_mode = False,
                             eva_timeout=cfg.timeout,
                             max_train_size=self.cfg.max_train_size,
                             gp_fit_sample_size=self.cfg.gp_fit_sample_size,
                             sampling_strategy=self.cfg.sampling_strategy,
                             use_sparse_gp=self.cfg.use_sparse_gp,
                             num_inducing_points=self.cfg.num_inducing_points,
                             exp_use_continue=self.cfg.exp_use_continue,
                             exp_continue_path=self.cfg.exp_continue_path,
                             parent_sample_size=self.cfg.parent_sample_size,)

        init_client(self.cfg)
    
    def evolve(self):
        print("- Evolution Start -")

        method = MCTS_AHD(self.paras, self.problem, prob_rank, pop_greedy)

        results = method.run()

        print("> End of Evolution! ")
        print("-----------------------------------------")
        print("---  MCTS-AHD successfully finished!  ---")
        print("-----------------------------------------")

        return results

    def bo_evolve(self):
        print("- Bayesian Optimization Start -")
        if self.cfg.algorithm == "bo_local":
            print("Using Local Bayesian Optimizer (Algorithm B)")
            method = BayesianOptimizer_Local(paras=self.paras, problem=self.problem)
        elif self.cfg.algorithm == "bo_baseline":
            print("Using Global Bayesian Optimizer (Algorithm C)")
            method = BayesianOptimizer_Baseline(paras=self.paras, problem=self.problem)
        else:
            print("Using Global Bayesian Optimizer (Algorithm A)")
            method = BayesianOptimizer(paras=self.paras, problem=self.problem)

        results = method.run()

        print("> End of Bayesian Optimization! ")
        print("-----------------------------------------")
        print("---  BO successfully finished!  ---")
        print("-----------------------------------------")

        return results
