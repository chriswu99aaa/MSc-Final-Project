
import logging
import numpy as np

from source.bo_interface import BOInterface

from os import path
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.decomposition import PCA
import traceback
from botorch.models import SingleTaskGP, FullyBayesianSingleTaskGP
from botorch.fit import fit_gpytorch_mll, fit_fully_bayesian_model_nuts
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.objective import IdentityMCObjective
from botorch.optim import optimize_acqf,optimize_acqf_discrete
from botorch.acquisition.logei import qLogExpectedImprovement
from gpytorch.kernels import ScaleKernel, RBFKernel,MaternKernel,RQKernel
import torch
from source.bo import BayesianOptimizer
from botorch.models.transforms import Normalize, Standardize
from botorch import settings as bts

class HeuristicNode:
    # def __init__(self, code, algorithm, parent=None, action=None, train_length=None, expanded_nodes=None):
    def __init__(self, code, algorithm, parent=None, action=None, train_length=None):
        self.code = code
        self.algorithm = algorithm
        self.parent = parent
        self.action = action  # action taken to generate this node
        self.children = []
        self.train_length = train_length
        # self.expanded_nodes = expanded_nodes
        self.feature_vector = None  # embedding vector for the code
    
    def add_child(self, child):
        self.children.append(child)

    def __repr__(self):
        return f"HeuristicNode(description={self.algorithm}, path_length={self.train_length})"

    def set_train_length(self, train_length):
        self.train_length = train_length

    def set_expanded_nodes(self, expanded_nodes):
        self.expanded_nodes = expanded_nodes


import random

class BayesianOptimizer_Local(BayesianOptimizer):
    def __init__(self,  paras, problem, **kwargs):
        super().__init__(paras, problem, **kwargs)
        self.initializing = False

        self.initial_epsilon = 1.0  # Start with full exploration
        self.epsilon_decay = 0.995 # Decay rate for exploration
        # self.ws = workspace_dir
        self.problem = problem
        self.paras = paras 
        self.parent_sample_size = paras.parent_sample_size #parent sample size for BO
        self.use_local_llm = paras.llm_use_local
        self.url = paras.llm_local_url
        self.api_endpoint = paras.llm_api_endpoint  # currently only API2D + GPT
        self.api_key = paras.llm_api_key
        self.llm_model = paras.llm_model

        # BO settings
        self.n_init = paras.n_init               # e.g. 10
        self.n_iters = paras.n_iters             # e.g. 20
        self.bounds = paras.bounds                # e.g. [[0,1],[0,1],...]
        self.dim = len(self.bounds)
        self.num_samples = paras.num_samples  # e.g. 10
        self.max_candidate_size = paras.max_candidate_size # e.g. 50
        self.parent_ucb_beta = 2.0
        # LLM settings
        self.use_local_llm = kwargs.get('use_local_llm', False)
        assert isinstance(self.use_local_llm, bool)
        if self.use_local_llm:
            assert 'url' in kwargs, 'The keyword "url" should be provided when use_local_llm is True.'
            assert isinstance(kwargs.get('url'), str)
            self.url = kwargs.get('url')

        # containers
        self.X = []  # shape (n_obs, d)
        self.Y = []  # scalar cost

        self.count = 0

        # Experimental settings
        self.use_seed = paras.exp_use_seed
        self.seed_path = paras.exp_seed_path
        self.load_pop = paras.exp_use_continue
        self.load_pop_path = paras.exp_continue_path
        self.load_pop_id = paras.exp_continue_id
        self.output_path = paras.exp_output_path
        self.debug_mode = paras.exp_debug_mode  # if debug
        self.use_continue = paras.exp_use_continue
        self.continue_path = paras.exp_continue_path
        self.timeout = self.paras.eva_timeout

        self.heuristics = []

        self.population = []

        self.action_types = ['i1', 'e1', 'e2', 'm1', 'm2', 's1']  # action types for evolution

        self.train_lengths = []  # to store tour lengths
        self.expanded_nodes = []  # to store expanded nodes

        self.pca = PCA(n_components=64)  # PCA for embedding dimensionality reduction

    def embed_code(self, code):
        """
        Use OpenAI API to embed the code snippet.
        Returns a numpy array of the embedding.
        """
        try:
            embedding = self.bo_interface.get_embedding(code)
            return np.array(embedding)
        except Exception as e:
            print(f"OpenAI embedding Fail: {e}")
            traceback.print_exc()
 
            return np.zeros(1536)  # text-embedding-ada-002的维度

    def encode_action(self, action):
        """
        Encode the action using one-hot encoding.
        """
        if not hasattr(self, 'action_encoder'):
            self.action_encoder = OneHotEncoder(sparse_output=False)
            # 使用所有可能的action类型进行fit
            actions_array = np.array(self.action_types).reshape(-1, 1)
            self.action_encoder.fit(actions_array)
        

        if action is None:
            action = 'i1'  # 默认action
        action_vector = self.action_encoder.transform([[action]])[0]
        return action_vector


    def create_feature_vector(self, code, action):
        """
        Create a feature vector by combining the code embedding and action one-hot encoding.
        The feature vector will be of shape (64 + 6,) after PCA reduction.
        """
        code_embedding = self.embed_code(code)
        action_onehot = self.encode_action(action)

        # # Apply PCA transform if PCA is fitted, else return original embedding
        # if hasattr(self, 'pca') and hasattr(self.pca, 'components_'):
        #     try:
        #         code_embedding_reduced = self.pca.transform(code_embedding.reshape(1, -1)).flatten()
        #     except Exception as e:
        #         print(f"PCA transform failed: {e}")
        #         code_embedding_reduced = code_embedding
        # else:
        #     code_embedding_reduced = code_embedding
        
        # 拼接特征：[code_embedding_reduced(64维) + action_onehot(6维)]
        feature_vector = np.concatenate([code_embedding, action_onehot])
        return feature_vector

    
    def initialize(self):
        print("- Initialization Start -")
        # interface_problem = self.problem

        # self.bo_interface = BOInterface(self.api_endpoint, self.api_key, self.llm_model,
        #                                 self.use_local_llm, interface_problem, use_local_llm=self.use_local_llm, url=self.url,
        #                                 timeout=self.timeout, population=self.population)
        
        # Retry logic for initial heuristic generation if NaN detected
        max_retries = 5
        for attempt in range(max_retries):
            code, description = self.bo_interface.generate_heuristic_by_action(action="i1")
            print(f"Initial response: {code}, {description}")

            train_length = self.problem.batch_evaluate([code], 0)[0]  # Evaluate the initial heuristic

            print(f"Initial objective value: {train_length if (not hasattr(self.problem, 'obj_type') or self.problem.obj_type == 'min') else -train_length}")
            
            # Calculate and cache feature vector
            feature_vector = self.create_feature_vector(code, "i1")
            if not (np.isnan(feature_vector).any() or (isinstance(train_length, float) and np.isnan(train_length))):
                # Create root node
                root = HeuristicNode(algorithm=description, code=code, parent=None, action="i1", train_length=train_length)
                root.feature_vector = feature_vector  # Cache the feature vector
                print(f"Root node created: {root}")
                self.population.append(root)
                self.train_lengths.append(train_length)
                self.X.append(feature_vector)
                self.Y.append(train_length)
                print(f"- Initialization Completed - Population size: {len(self.population)}")
                break
            else:
                print(f"Initialization attempt {attempt+1}: Detected NaN in feature vector or train_length. Retrying...")
        else:
            raise RuntimeError("Failed to generate valid initial heuristic after multiple attempts.")


    def fit(self):
        """
        Fit the Fully Bayesian Gaussian Process model to the collected data using NUTS.
        """
        import numpy as np

        # Debug: Check for NaNs or infs in self.X and self.Y
        X_arr = np.vstack(self.X)
        Y_arr = np.array(self.Y)

        train_x = torch.tensor(X_arr, dtype=torch.double)
        # 关键：训练目标取负（最小化 f 变成最大化 -f）
        train_y = torch.tensor(-Y_arr, dtype=torch.double).unsqueeze(-1)

        model = SingleTaskGP(
            train_x,
            train_y,
            covar_module=ScaleKernel(RQKernel()),
            input_transform=Normalize(train_x.shape[-1]),
            outcome_transform=Standardize(1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def acquisition_function(self, model, candidate_features):
        """
        the input should be the five action generated heuristics
        Define the acquisition function for Bayesian optimization.
        """

        input_dim = candidate_features.shape[1]
        num_samples = self.num_samples
        num_candidates = candidate_features.shape[0]
        if num_candidates > 0:
            indices = torch.randperm(num_candidates)[:min(num_samples, num_candidates)]
            optimal_inputs = candidate_features[indices]

            if len(optimal_inputs) < num_samples:
                repeat_times = (num_samples + len(optimal_inputs) - 1) // len(optimal_inputs)
                optimal_inputs = optimal_inputs.repeat(repeat_times, 1)[:num_samples]
        else:
            optimal_inputs = torch.randn((num_samples, input_dim), dtype=torch.double)
        
        # Retry logic for qPredictiveEntropySearch with increasing jitter/threshold
        search_success = False
        jitter_vals = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        threshold_vals = [1e-2]
        for jitter in jitter_vals:
            for threshold in threshold_vals:
                try:
                    acq_func = qPredictiveEntropySearch(
                        model=model,
                        optimal_inputs=optimal_inputs,
                        maximize=True,
                        X_pending=None,
                        max_ep_iterations=30,  # previous 50
                        ep_jitter=jitter,
                        test_jitter=jitter,
                        threshold=threshold
                    )
                    logging.info(f"qPredictiveEntropySearch Start Search")
                    best_candidate, _ = optimize_acqf_discrete(
                        acq_function=acq_func,
                        q=1,
                        choices=candidate_features,
                    )
                    logging.info(f"qPredictiveEntropySearch Start Search Success")

                    # Directly find the index of the best_candidate
                    squeezed_best_candidate = best_candidate.squeeze()

                    # Compare each candidate with the best one
                    # The result of isclose is a boolean tensor, all(dim=1) checks for each row
                    matches = torch.isclose(candidate_features, squeezed_best_candidate, atol=1e-6).all(dim=1)
                    logging.info(f"Matches Found")

                    # Get the index of the first match
                    indices = matches.nonzero(as_tuple=True)[0]
                    if indices.numel() > 0:
                        search_success = True
                        return indices[0].item()
                except Exception as e:
                    print(f"qPredictiveEntropySearch failed with jitter={jitter}, threshold={threshold}: {e}")
                    continue
        if not search_success:
            print("All qPredictiveEntropySearch attempts failed. Falling back to basic acquisition function.")
            return self._fallback_acquisition(model, candidate_features)

        print("No matching candidate found.")
        logging.error("No matching candidate found.")
        return 0

    def _fallback_acquisition(self, model, candidate_features):
        """
        Fallback acquisition function that selects the best candidate based on the model's predictions.
        """

        # check if we have any valid Y values
        if len(self.Y) > 0:
            best_f = torch.tensor(-min(self.Y), dtype=torch.double)
        else:
            best_f = torch.tensor(float('-inf'), dtype=torch.double)
        
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        
        acq_func = qLogExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=IdentityMCObjective(),
        )
        k = 5  # Number of candidates to select
        try:
            # Ensure candidates are in the right shape for batch processing
            with torch.no_grad():
                # Only unsqueeze once to get the correct shape [N, 1, D] for q=1 batch evaluation
                candidate_features_shaped = candidate_features.unsqueeze(1)
                vals = acq_func(candidate_features_shaped)  # Should return [N] tensor
                topk = torch.topk(vals.squeeze(-1), k=min(k, candidate_features.shape[0]), largest=True).indices.tolist()
                return topk
        except Exception as e:
            print(f"Fallback acquisition also failed: {e}")
            return 0


    def expand(self, node: HeuristicNode):
        """
        Expand the node by generating new heuristics for each action, and return them as a list.
        This version only generates nodes without evaluating them.
        """
        max_retries = 5
        new_nodes = []

        for action in self.action_types:
            if action == 'i1':
                continue
            print(f"Generating heuristic for action: {action}")

            # 1. generate heuristic code for the action
            code, algorithm = None, None
            for retry in range(max_retries):
                try:
                    code, algorithm = self.bo_interface.generate_heuristic_by_action(action)
                    if self.check_duplicate_code(self.population, code):
                        print(f"Duplicate code detected for action {action}, retry {retry + 1}")
                        code, algorithm = None, None
                        continue
                    break
                except Exception as e:
                    print(f"Error generating heuristic for action {action}, retry {retry + 1}: {e}")
                    continue
            
            if not code:
                print(f"Failed to generate candidate for action {action} after {max_retries} retries")
                continue

            # Do not evaluate here, just create the node with None train_length
            child = HeuristicNode(algorithm=algorithm, code=code, parent=node, action=action, train_length=None)
            feature_vector = self.create_feature_vector(code, action)

            # if self.check_similar(feature_vector) and action not in ['m1', 'm2']:
            #     print(f"Similar feature vector detected for action {action}, skipping child.")
            #     continue
            
            child.feature_vector = feature_vector
            new_nodes.append(child)
            print(f"Successfully generated heuristic for action {action} (unevaluated)")
        

        return new_nodes


    def select_parents_ucb(self, model, beta:float=2.0):
        """
        Select parents using Upper Confidence Bound (UCB) strategy.
        """
        if len(self.X) == 0:
            return []
        k = min(self.parent_sample_size, 3) 
        self.initial_epsilon *= self.epsilon_decay  # Decay epsilon for exploration
        eps = max(self.initial_epsilon,0.3)
        # 分支一：随机探索
        if np.random.rand() < eps:
            idx = np.random.choice(len(self.population), size=k, replace=False).tolist()
            print(f"[UCB-ε] Random pick (ε={eps:.3f}) -> nodes: {[self.population[i] for i in idx]}")
            return [self.population[i] for i in idx]
        else:
            # Convert X to tensor
            X_tensor = torch.tensor(np.vstack(self.X), dtype=torch.double)

            # Predict mean and variance
            with torch.no_grad():
                post = model.posterior(X_tensor)
                mu = post.mean.squeeze(-1)                  # [-Y 的均值]
                sigma = post.variance.sqrt().squeeze(-1)    # 标准差

            ucb = mu + beta * sigma 

            # Select top k parents based on UCB values
            top_indices = torch.topk(ucb, k=k, largest=True).indices.tolist()
            
            print(f"Top K parents selected by UCB node: {[self.population[i] for i in top_indices]}")
            return [self.population[i] for i in top_indices]

    def check_duplicate_obj(self, population, obj):
        for ind in population:
            if obj == ind.train_length:
                return True
        return False

    def check_duplicate_code(self, population, code):
        for ind in population:
            if code == ind.code:
                return True
        return False
    
    def check_similar(self, fv, thr=0.98):
        if not self.X:
            return False

        X = np.vstack(self.X).astype(np.float64)
        fv = fv.astype(np.float64)

        # 原来是 v.shape[0]，变量未定义
        D = min(X.shape[1], fv.shape[0])
        E_code = X[:, :D]
        v_code = fv[:D]

        denom = (np.linalg.norm(E_code, axis=1) * (np.linalg.norm(v_code) + 1e-12) + 1e-12)
        cos = (E_code @ v_code) / denom
        return cos.max() > thr

    def _get_node_feature_vector(self, node: HeuristicNode):
        if node.feature_vector is None:
            node.feature_vector = self.create_feature_vector(node.code, node.action)
        return node.feature_vector

    def plot_performance(self, iteration):
        import matplotlib.pyplot as plt
        
        if not self.Y:
            return

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.Y) + 1), self.Y, marker='o', linestyle='-', label='Performance (Tour Length)')
        
        # Find the best performance up to the current iteration
        best_performance = [min(self.Y[:i+1]) for i in range(len(self.Y))]
        plt.plot(range(1, len(self.Y) + 1), best_performance, marker='x', linestyle='--', label='Best Performance so far')

        plt.xlabel('Evaluation')
        plt.ylabel('Tour Length')
        plt.title(f'Performance over Iterations (Up to Iteration {iteration})')
        plt.legend()
        plt.grid(True)
        
        plot_filename = self.output_path + f"performance_iteration_{iteration}.png"
        plt.savefig(plot_filename)
        plt.close()
        logging.info(f"Performance plot saved to {plot_filename}")


    def run(self):
        import json

        interface_problem = self.problem
        self.bo_interface = BOInterface(self.api_endpoint, self.api_key, self.llm_model,
                                        self.use_local_llm, interface_problem, use_local_llm=self.use_local_llm, url=self.url,
                                        timeout=self.timeout, population=self.population)

        if self.use_continue == True and self.continue_path is not None and path.exists(self.continue_path):
            self.load_population_from_file(self.continue_path)
            self.bo_interface.population = self.population
            print(f"Continuing training from iteration {self.count} to {self.n_iters}")
        else:
            print("Starting new Bayesian Optimization run.")
            self.initialize()  # Initialize the optimizer with the first heuristic
            self.count = 0

            # === Bayesian Optimization Main Loop ===
        for iteration in range(self.count, self.n_iters):
            print(f"\n=== Bayesian Optimization Iteration {iteration + 1}/{self.n_iters} ===")

            # 1. Fit GP model with current population
            with bts.debug(True):
                if len(self.X) > 0:
                    model = self.fit()
                    print(f"GP Model Fitted with Data: {len(self.X)}")
                else:
                    print("Inadequate Data")
                    continue

            # 2. 采样部分父节点进行扩展，降低计算量
            all_child_nodes = []
            if len(self.population) <= self.parent_sample_size:
                sampled_parents = self.population
                print('Randomly Sampling')
                print(f"poplation size: {len(self.population)}")
            else:
                sampled_parents = self.select_parents_ucb(
                    model,
                    beta=self.parent_ucb_beta,
                )

            for parent_node in sampled_parents:
                children = self.expand(parent_node)
                if children:
                    all_child_nodes.extend(children)

            if not all_child_nodes:
                print("Expansion did not yield any new nodes. Continuing to next iteration.")
                continue

            # 3. Prepare candidate set from all new child nodes
            candidate_features = torch.stack([torch.tensor(self._get_node_feature_vector(node), dtype=torch.double) for node in all_child_nodes])
            print(f"Candidate Set Size (from all children): {candidate_features.shape}")

            # 4. Use acquisition function to select the best child node indices
            try:
                # best_child_indices = self._fallback_acquisition(model, candidate_features)
                best_child_indices = self._fallback_acquisition(model, candidate_features)
                # Handle case where fallback returns a single index
                if isinstance(best_child_indices, int):
                    best_child_indices = [best_child_indices]
                selected_children = [all_child_nodes[idx] for idx in best_child_indices]
                print(f"Selected {len(selected_children)} Child Nodes for batch evaluation")
            except Exception as e:
                print(f"Acquisition Function Execution Failed: {e}")
                # Fallback to random selection
                selected_children = [random.choice(all_child_nodes)]
                print(f"Fallback To Random Selection: selected 1 child")

            # 5. Batch evaluate all selected children using multithreading
            selected_codes = [child.code for child in selected_children]
            try:
                
                train_lengths = self.problem.batch_evaluate(selected_codes, 0)
                print(f"Batch evaluated {len(selected_children)} children")
                
                # Assign evaluation results to children and filter valid ones
                valid_children = []
                for child, train_length in zip(selected_children, train_lengths):
                    child.train_length = train_length
                    print(f"Child objective value: {train_length}")
                    
                    # # Check for duplicates and validity
                    # if self.check_duplicate_obj(self.population, train_length):
                    #     print(f"Duplicate evaluated train_length {train_length} detected, skipping child.")
                    #     continue
                    
                    if not (isinstance(train_length, float) and np.isfinite(train_length)):
                        print(f"Invalid evaluated train_length {train_length}, skipping child.")
                        continue

                    if self.check_duplicate_code(self.population, child.code):
                        print(f"Duplicate evaluated code detected for action {child.action}, skipping child.")
                        continue
                    
                    # if self.check_similar(child.feature_vector,0.98):
                    #     print(f"Similar feature vector detected for action {child.action}, skipping child.")
                    #     continue
                    valid_children.append(child)
                
            except Exception as e:
                print(f"Batch evaluation failed: {e}")
                continue

            if not valid_children:
                print("No valid children after batch evaluation, continuing to next iteration.")
                continue

            # 6. Add all valid children to the population and training data
            for selected_child in valid_children:
                selected_child.parent.add_child(selected_child)
                self.population.append(selected_child)
                self.train_lengths.append(selected_child.train_length)
                self.X.append(selected_child.feature_vector)
                self.Y.append(selected_child.train_length)

            # 7. Output current best performance
            if self.Y:
                if self.problem.obj_type == "min":
                    current_best = min(self.Y)
                else:
                    # For maximization, negate back to show the true best value
                    current_best = -min(self.Y)
                print(f"Current Best Performance: {current_best}")

            # 8. Save population and best individual
            self.count = iteration + 1

            population_data = []
            for idx, node in enumerate(self.population):
                node_data = {
                    "algorithm": node.algorithm,
                    "code": node.code,
                    "objective": node.train_length,
                    "action": node.action,
                    "parent": self.population.index(node.parent) if node.parent is not None else None,
                    "children": [self.population.index(child) for child in node.children]
                }
                population_data.append(node_data)

            filename = self.output_path + "population_generation_" + str(self.count) + ".json"
            with open(filename, 'w') as f:
                json.dump(population_data, f, indent=5)

            if self.problem.obj_type == "min":
                best_node = min(self.population, key=lambda x: x.train_length if x.train_length is not None else float('-inf'))
            else:
                # For maximization, best is the node with the minimum (negated) value
                best_node = min(self.population, key=lambda x: x.train_length if x.train_length is not None else float('inf'))
            best_data = {
                "algorithm": best_node.algorithm,
                "code": best_node.code,
                # For maximization, negate back to show the true value
                "objective": best_node.train_length if self.problem.obj_type == "min" else -best_node.train_length,
                "action": best_node.action
            }

            best_filename = self.output_path + "best_population_generation_" + str(self.count) + ".json"
            with open(best_filename, 'w') as f:
                json.dump(best_data, f, indent=5)

            # Plot performance
            # self.plot_performance(self.count)

        # === Return The Final Result (following MCTS format) ===
        if not self.population:
            return "", ""

        if self.problem.obj_type == "min":
            best_node = min(self.population, key=lambda x: x.train_length if x.train_length is not None else float('inf'))
            best_value = best_node.train_length
        else:
            # For maximization, best is the node with the minimum (negated) value
            best_node = min(self.population, key=lambda x: x.train_length if x.train_length is not None else float('inf'))
            best_value = -best_node.train_length

        print(f"\n=== Optimization Complete ===")
        print(f"Best Performance: {best_value}")
        print(f"Total Evaluations: {len(self.Y)}")
        print(f"Population Size: {len(self.population)}")

        # Return best code and the filename of the last saved best individual (like MCTS)
        final_filename = self.output_path + "best_population_generation_" + str(self.count) + ".json"
        return best_node.code, final_filename


    def load_population_from_file(self, filename):
        import json
        print(f"Loading population from file: {filename}")
        with open(filename, 'r') as f:
            data = json.load(f)

        self.population = []
        self.X = []
        self.Y = []

        for item in data:
            algorithm = item.get("algorithm")
            code = item.get("code")
            action = item.get("action")
            objective = item.get("objective")

            # The new format has objective as a single value.
            # We'll use it as train_length and assume gap is 0 or None.
            train_length = objective

            node = HeuristicNode(
                algorithm=algorithm,
                code=code,
                parent=None,
                action=action,
                train_length=train_length,
            )
            feature_vector = self.create_feature_vector(code, action)
            node.feature_vector = feature_vector

            self.population.append(node)
            self.X.append(feature_vector)
            self.Y.append(objective)

        # restore parent-child relationships
        for idx, item in enumerate(data):
            parent_idx = item.get("parent")
            if parent_idx is not None:
                node = self.population[idx]
                parent_node = self.population[parent_idx]
                node.parent = parent_node
                parent_node.children.append(node)


        import re
        match = re.search(r'_(\d+)\.json$', filename)
        if match:
            self.count = int(match.group(1))
        else:
            self.count = 0

        self.initial_epsilon = 1.0 * 0.995 ** self.count  # Reset epsilon based on count
        print(f"Loaded population size: {len(self.population)}, starting iteration: {self.count}")
