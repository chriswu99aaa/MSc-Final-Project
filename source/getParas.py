import os


class Paras():
    def __init__(self):
        #####################
        ### General settings  ###
        #####################
        self.method = 'mcts_ahd'
        self.problem = 'tsp_construct'
        self.management = 'pop_greedy'
        self.selection = 'prob_rank'

        #####################
        ###  EC settings  ###
        #####################
        self.pop_size = 10  # Size of Elite set E, default = 10
        self.init_size = 4  # Number of initial nodes N_I, default = 4
        self.ec_fe_max = 1000  # Number of evaluations, default = 1000
        self.ec_operators = ['e1', 'e2', 'm1', 'm2', 's1']
        self.ec_m = 2
        self.ec_operator_weights = [0, 1, 2, 2, 1]  # weights for operators default = [0,1,k,k,1], default = [0,1,2,2,1]

        #####################
        ### LLM settings  ###
        #####################
        self.llm_use_local = False  # if use local model
        self.llm_local_url = None  # your local server 'http://127.0.0.1:11012/completions'
        self.llm_api_endpoint = "chat.openai.com"
        self.llm_api_key = "Not used"  # Not used
        self.llm_model = "gpt-3.5-turbo-1106"

        #####################
        ###  Exp settings  ###
        #####################
        self.exp_debug_mode = True  # if debug
        self.exp_output_path = "/"  # default folder for ael outputs
        self.exp_use_seed = False
        self.exp_seed_path = "./seeds/seeds.json"
        self.exp_use_continue = True
        self.exp_continue_id = 0
        self.exp_continue_path = "./results/pops/population_generation_0.json"
        self.exp_n_proc = -1
        
        #####################
        ###  Evaluation settings  ###
        #####################
        self.eva_timeout = 60
        self.eva_numba_decorator = False
        
   # these must match whatever's in your hydra config under `bo:`
        self.n_init   = 10      # number of initial random samples
        self.n_iters  = 20      # number of BO iterations
        self.bounds   = [[0.5,1],[0.5,1]]  # parameter bounds
        self.noise   = 1e-6     # noise level for GP regression

        self.n_instances = 20  # number of instances to evaluate
        self.num_samples = 20  # number of samples to evaluate in acquisition function
        self.max_candidate_size = 50 # acquisition function's candidate size
        self.max_train_size = 100 # max training size for GP
        self.gp_fit_sample_size = 100 # number of samples to fit GP
        self.sampling_strategy = 'sobol'  # sampling strategy GP uses, e.g. 'random', 'sobol', 'latin_hypercube', 'stratified', 'diversity', 'ucb', 'thompson', 'adaptive'
        self.use_sparse_gp = False  # whether to use sparse GP
        self.num_inducing_points = 64  # number
        self.lambda_value = 0.7  # weight between train_length and gap in the objective function
        self.parent_sample_size = 5  # Number of parents to sample for expansion in BO

    def set_paras(self, *args, **kwargs):

        # Map paras
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Identify and set parallel 
        # self.set_parallel()

        # Initialize method and ec settings
        # self.set_ec()

        # Initialize evaluation settings
        # self.set_evaluation()

        # assert self.llm_api_key!="XXX", "Please set the environment variable `OPENAI_API_KEY`"


if __name__ == "__main__":
    # Create an instance of the Paras class
    paras_instance = Paras()

    # Setting parameters using the set_paras method
    paras_instance.set_paras(llm_use_local=True, llm_local_url='http://example.com', ec_pop_size=8)

    # Accessing the updated parameters
    print(paras_instance.llm_use_local)  # Output: True
    print(paras_instance.llm_local_url)  # Output: http://example.com
    print(paras_instance.ec_pop_size)  # Output: 8
