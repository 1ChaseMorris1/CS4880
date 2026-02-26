Monte Carlo Tree Search (MCTS) builds a search tree incrementally by running simulated playouts. Each iteration has four phases: Selection picks the most promising path using a tree policy; Expansion adds a new child node; Simulation plays a random rollout to a terminal state; Backpropagation updates statistics along the path back to the root.

What is UCB? UCB stands for Upper Confidence Bound — a strategy from the multi-armed bandit literature (the same exploration-vs-exploitation tradeoff from Chapter 2). When choosing which tree branch to explore next, we want to balance two goals: exploitation (prefer nodes with high win rates) and exploration (revisit nodes we haven't tried much — they might be better than they appear). UCB achieves this by adding an “optimism bonus” to each node's estimated value.

The UCB1 formula applies this idea to tree search. The “1” in UCB1 denotes the first and simplest variant in a family of UCB algorithms introduced by Auer, Cesa-Bianchi & Fischer (2002). Their paper also proposed UCB2 (a more complex epoch-based variant) and UCB1-Tuned (which incorporates reward variance). UCB1 became the most widely adopted because it is simple, has strong theoretical guarantees (logarithmic regret), and works well in practice — which is why MCTS adopted it. During the Selection phase, MCTS picks the child with the highest UCB1 score:

UCB1(j) = Q̅j + C · √ln Nparent / Nj
↑ Exploitation term     ↑ Exploration bonus

Q̅j = average win rate of child j (exploitation term — prefer nodes that have been winning)
Nparent = parent visit count — the total number of times the parent node has been visited. As the parent accumulates more visits, the exploration bonus grows logarithmically, ensuring that even well-visited children are periodically re-explored.
Nj = child visit count — the number of times child j has been visited. A low visit count relative to the parent makes the exploration term large, giving under-explored children a higher UCB1 score so they are prioritized for selection.
C · √ln Nparent / Nj = exploration bonus (large when j has few visits relative to its parent — encourages trying under-explored moves)
C = exploration constant (default √2 ≈ 1.41; higher = more exploration, lower = more exploitation)
Intuitive explanation. Imagine each move is a slot machine and you don’t know which one pays best. Each move has an estimated value (how well it has performed so far) and uncertainty (how little you’ve tried it). UCB says: “Assume each move might be as good as its optimistic upper bound, then pick the move whose plausible best-case estimate is highest.” Moves you haven’t tried much get a large uncertainty bonus, so they are tried soon; moves that keep winning maintain a high estimated value, so they are tried often. Over time the uncertainty shrinks and the best move emerges.

Why “Upper Confidence Bound”? The name comes from statistics. If the true value of a move is μ, then with high probability μ ≤ Q̅ + confidence radius. That confidence radius shrinks as you sample more. So UCB literally means: “Pick the move whose upper confidence estimate is highest.” This is not worst-case reasoning, not adversarial, and not minimax — it is optimism in the face of uncertainty, a principle that provably minimises regret.

How to use this demo

The tree starts with a single root node showing “0” — this represents the current board position (an empty board by default) with zero visits so far.
Step Phase walks through one MCTS iteration phase by phase (Selection → Expansion → Simulation → Backpropagation). Watch the colored highlights on the tree and the phase bar above it.
Step Iteration runs one full 4-phase cycle at once.
Auto runs iterations continuously — watch the tree grow and the win-rate chart converge. Use the speed slider to control how fast.
+10 / +100 / +1000 run many iterations instantly (no animation, but faster results).
To analyze a specific position: click Setup Board, click cells to cycle through empty/X/O, then click Done Editing. The tree resets to search from your position.
Step Phase
Step Iteration
Auto

10/s
+10
+100
+1000
C

1.41

Random
Reset Tree
Setup Board
Clear Board
1. Selection
2. Expansion
3. Simulation
4. Backpropagation
Fit
+
−
Top number = visit count (N)
Bottom number = win rate (Q/N) from root player's view
Label above = board move (e.g. B2 = column B, row 2)
Color: red = losing, yellow = neutral, green = winning
Max depth:

6
Board Position
X to move
A
B
C
1
2
3
The green cell shows the best move found so far.
Statistics
Iterations
0
Tree Nodes
1
Best Move
—
Win Rate
—
Move Rankings
Move	Visits	Win%	Q	UCB1
No data yet
UCB1 Live
UCB1 (Upper Confidence Bound 1) scores each child node as UCB1 = Q̅ + C · √ln Np / Nj, where Q̅ is the node's average reward (wins / visits), Np is how often the parent was visited, Nj is how often this child was visited, and C is a tunable constant (typically √2). The score is an optimistic estimate of a node's true value: the first term (exploitation) reflects how promising a move has been so far, while the second term (exploration) acts as a confidence bonus that grows for less-visited nodes — ensuring no move is neglected. During selection, MCTS always picks the child with the highest UCB1, which automatically balances exploiting known-good moves against exploring uncertain ones. Click any tree node to inspect its breakdown below.

Run an iteration to see UCB1 values.
Iteration Log
Waiting for first iteration...
Best-Move Win Rate Over Iterations