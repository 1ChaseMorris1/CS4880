Project Assignment 4: LLM-Guided Monte Carlo Tree Search for Tic-Tac-Toe
Background
Monte Carlo Tree Search (MCTS) builds a game tree incrementally through repeated iterations, each consisting of four phases: Selection (choosing a promising path using UCB1), Expansion (adding a new child node), Simulation (playing out to a terminal state), and Backpropagation (updating statistics along the path). An interactive visualization of this process is available at:

https://drchangliu.github.io/RL/MCTS.html

Explore this visualization before starting the assignment. Use the Step Phase button to observe each of the four MCTS phases in detail, and experiment with the Setup Board feature to see how MCTS searches from different board positions. Pay attention to how the UCB1 formula balances exploitation (choosing moves with high win rates) and exploration (trying moves that have been visited less often), and how random rollouts in the Simulation phase provide value estimates for newly expanded nodes.

The Limitation of Random Rollouts
In standard MCTS, the Simulation phase evaluates a newly expanded node by playing random moves until the game ends. While this works reasonably well given enough iterations, random rollouts can be noisy and wasteful—many simulated games follow nonsensical move sequences that do not reflect intelligent play. This raises a natural question: Can we replace (or augment) random rollouts with a more informed evaluation?

The RAP Framework: LLMs as World Models
Hao et al. (2023) propose Reasoning via Planning (RAP), a framework that repurposes a Large Language Model to serve dual roles within an MCTS-based planner:

LLM as World Model: The LLM predicts the next state given a current state and an action (i.e., it models state transitions).
LLM as Reasoning Agent: The LLM generates candidate actions and provides reward signals that assess how promising a given state is.
In RAP, the LLM's reward estimates replace or supplement the random rollout step of traditional MCTS. Rather than simulating random playouts to a terminal state, the framework queries the LLM to evaluate how favorable a position is, enabling more informed backpropagation of value estimates. RAP combines multiple reward signals—such as action likelihood, state confidence, and LLM self-evaluation—to guide the tree search efficiently.

Reference: Hao, S., Gu, Y., Ma, H., Hong, J., Wang, Z., Wang, D., & Hu, Z. (2023). Reasoning with language model is planning with world model. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (pp. 8154–8173). Association for Computational Linguistics.

Your Task
Modify an MCTS solver for Tic-Tac-Toe by incorporating an LLM-guided evaluation in the spirit of the RAP framework. Specifically:

Implement a baseline MCTS agent for Tic-Tac-Toe that uses standard random rollouts in the Simulation phase (or use the existing implementation from the course visualization as your starting point).

Replace the random rollout with an LLM-based evaluation. Inspired by RAP, use an LLM (e.g., via the OpenAI or Anthropic API) to estimate the value of a board position without performing a full random playout. You may design this in one or more of the following ways:

Direct position evaluation: Prompt the LLM with the current board state and ask it to estimate the win probability for the current player.
Informed action prior: Use the LLM to assign prior probabilities to candidate moves, biasing the Selection phase (similar to how AlphaGo uses a policy network alongside UCB).
LLM-as-world-model simulation: Have the LLM predict the likely outcome of a sequence of moves from the current position, producing a more informed (but not fully random) rollout. (For very simple games such as Tic-Tac-Toe, it would almost always be easier to just use the game engine.)
Compare the two approaches. Run both the standard MCTS agent and the LLM-guided MCTS agent under controlled conditions. You should:

Fix the number of MCTS iterations (e.g., 10, 50, 100, 500) and compare move quality at each level.
Measure win/draw/loss rates when the two agents play against each other, and optionally against a perfect (minimax) player.
Track per-move metrics such as the win-rate estimate for the chosen move and the time per iteration.
Analyze and reflect. Write a discussion that addresses the following:

Did the LLM-guided approach improve the quality of play? At what iteration counts is the difference most or least pronounced?
Tic-Tac-Toe is a solved game with a small state space. How does this affect whether an LLM evaluation adds value compared to random rollouts, which converge quickly in small games?
What are the trade-offs in terms of computational cost (API latency and token usage vs. the speed of random rollouts)?
Under what circumstances—game complexity, tree depth, branching factor—would you expect the LLM-guided approach to provide a larger advantage?
Relate your findings back to the RAP paper's results on more complex tasks (plan generation, math reasoning) and discuss why the benefits may differ for Tic-Tac-Toe.
Deliverables
Source code for both the baseline and LLM-guided MCTS implementations.
A results table or chart comparing performance across different iteration counts.
A written analysis (approximately 1–2 pages) reflecting on the experimental outcomes and connecting them to the ideas in the RAP paper.
A slide deck that you will use for in-class presentation. Make sure basic contextual info (e.g. which LLM you used, the prompts you used, etc.) is included.
Tips
Start by carefully reading the RAP paper (especially Sections 3.2–3.3 on the reward design and the MCTS algorithm) to understand how LLM confidence signals are integrated into the tree search.
Keep your LLM prompts simple and well-structured. For example, represent the board as a 3×3 grid in text and ask the LLM to rate the position on a scale of 0 to 1.
Be mindful of API costs. You may want to cache LLM evaluations for board states you have already queried, since many MCTS iterations will revisit the same positions.
Consider running experiments with a small, locally hosted model (i.e. Ollama models) to avoid latency bottlenecks and to save token cost.