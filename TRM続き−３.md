=== PAGE 7 ===

Recursive Reasoning with Tiny Networks
4.4. Less is more
We attempted to increase capacity by increasing the
number of layers in order to scale the model. Sur-
prisingly, we found that adding layers decreased gen-
eralization due to overfitting. In doing the oppo-
site, decreasing the number of layers while scaling
the number of recursions ( n) proportionally (to keep
the amount of compute and emulated depth approxi-
mately the same), we found that using 2 layers (instead
of 4 layers) maximized generalization. In doing so, we
obtain better generalization on Sudoku-Extreme (im-
proving TRM from 79.5% to 87.4%; see Table 1) while
reducing the number of parameters by half (again).
It is quite surprising that smaller networks are bet-
ter, but 2 layers seems to be the optimal choice. Bai
& Melas-Kyriazi (2024) also observed optimal perfor-
mance for 2-layers in the context of deep equilibrium
diffusion models; however, they had similar perfor-
mance to the bigger networks, while we instead ob-
serve better performance with 2 layers. This may ap-
pear unusual, as with modern neural networks, gener-
alization tends to directly correlate with model sizes.
However, when data is too scarce and model size is
large, there can be an overfitting penalty (Kaplan et al.,
2020). This is likely an indication that there is too little
data. Thus, using tiny networks with deep recursion
and deep supervision appears to allow us to bypass a
lot of the overfitting.
4.5. attention-free architecture for tasks with small
fixed context length
Self-attention is particularly good for long-context
lengths when L≫D since it only requires a matrix of
[D, 3D]parameters, even though it can account for the
whole sequence. However, when focusing on tasks
whereL≤D, a linear layer is cheap, requiring only a
matrix of [L,L]parameters. Taking inspiration from
the MLP-Mixer (Tolstikhin et al., 2021), we can replace
the self-attention layer with a multilayer perceptron
(MLP) applied on the sequence length. Using an MLP
instead of self-attention, we obtain better generaliza-
tion on Sudoku-Extreme (improving from 74.7% to
87.4%; see Table 1). This worked well on Sudoku 9x9
grids, given the small and fixed context length; how-
ever, we found this architecture to be suboptimal for
tasks with large context length, such as Maze-Hard
and ARC-AGI (both using 30x30 grids). We show
results with and without self-attention for all experi-
ments.4.6. No additional forward pass needed with ACT
As previously mentioned, the implementation of ACT
in HRM through Q-learning requires two forward
passes, which slows down training. We propose a
simple solution, which is to get rid of the continue loss
(from the Q-learning) and only learn a halting proba-
bility through a Binary-Cross-Entropy loss of having
reached the correct solution. By removing the continue
loss, we remove the need for the expensive second for-
ward pass, while still being able to determine when to
halt with relatively good accuracy. We found no sig-
nificant difference in generalization from this change
(going from 86.1% to 87.4%; see Table 1).
4.7. Exponential Moving Average (EMA)
On small data (such as Sudoku-Extreme and Maze-
Hard), HRM tends to overfit quickly and then diverge.
To reduce this problem and improves stability, we
integrate Exponential Moving Average (EMA) of the
weights, a common technique in GANs and diffusion
models to improve stability (Brock et al., 2018; Song &
Ermon, 2020). We find that it prevents sharp collapse
and leads to higher generalization (going from 79.9%
to 87.4%; see Table 1).
4.8. Optimal the number of recursions
We experimented with different number of recursions
by varying Tand nand found that T= 3,n= 3
(equivalent to 48 recursions) in HRM and T= 3,n= 6
in TRM (equivalent to 42 recursions) to lead to optimal
generalization on Sudoku-Extreme. More recursions
could be helpful for harder problems (we have not
tested it, given our limited resources); however, in-
creasing eitherTornincurs massive slowdowns. We
show results at different nand Tfor HRM and TRM
in Table 3. Note that TRM requires backpropagation
through a full recursion process, thus increasing ntoo
much leads to Out Of Memory (OOM) errors. How-
ever, this memory cost is well worth its price in gold.
In the following section, we show our main results on
multiple datasets comparing HRM, TRM, and LLMs.
5. Results
Following Wang et al. (2025), we test our approach
on the following datasets: Sudoku-Extreme (Wang
et al., 2025), Maze-Hard (Wang et al., 2025), ARC-AGI-
1 (Chollet, 2019) and, ARC-AGI-2 (Chollet et al., 2025).
Results are presented in Tables 4 and 5. Hyperparame-
ters are detailed in Section 6. Datasets are discussed
below.
7

=== PAGE 8 ===

Recursive Reasoning with Tiny Networks
Table 3. % Test accuracy on Sudoku-Extreme dataset. HRM
versus TRM matched at a similar effective depth per super-
vision step(T(n+1)n layers)
HRM TRM
n=k, 4 layersn=2k, 2 layers
k TDepth Acc (%) Depth Acc (%)
1 1 9 46.4 7 63.2
2 2 24 55.0 20 81.9
3 3 48 61.6 42 87.4
4 4 80 59.5 72 84.2
6 3 84 62.3 78 OOM
3 6 96 58.8 84 85.8
6 6 168 57.5 156 OOM
Sudoku-Extreme consists of extremely difficult Su-
doku puzzles (Dillion, 2025; Palm et al., 2018; Park,
2018) (9x9 grid), for which only 1K training samples
are used to test small-sample learning. Testing is done
on 423K samples. Maze-Hard consists of 30x30 mazes
generated by the procedure by Lehnert et al. (2024)
whose shortest path is of length above 110; both the
training set and test set include 1000 mazes.
ARC-AGI-1 and ARC-AGI-2 are geometric puzzles in-
volving monetary prizes. Each puzzle is designed to
be easy for a human, yet hard for current AI models.
Each puzzle task consists of 2-3 input–output demon-
stration pairs and 1-2 test inputs to be solved. The final
score is computed as the accuracy over all test inputs
from two attempts to produce the correct output grid.
The maximum grid size is 30x30. ARC-AGI-1 con-
tains 800 tasks, while ARC-AGI-2 contains 1120 tasks.
We also augment our data with the 160 tasks from
the closely related ConceptARC dataset (Moskvichev
et al., 2023). We provide results on the public evalua-
tion set for both ARC-AGI-1 and ARC-AGI-2.
While these datasets are small, heavy data-
augmentation is used in order to improve gen-
eralization. Sudoku-Extreme uses 1000 shuffling
(done without breaking the Sudoku rules) augmenta-
tions per data example. Maze-Hard uses 8 dihedral
transformations per data example. ARC-AGI uses
1000 data augmentations (color permutation, dihedral-
group, and translations transformations) per data
example. The dihedral-group transformations consist
of random 90-degree rotations, horizontal/vertical
flips, and reflections.
From the results, we see that TRM without self-
attention obtains the best generalization on Sudoku-
Extreme (87.4% test accuracy). Meanwhile, TRM with
self-attention generalizes better on the other tasks
(probably due to inductive biases and the overcapac-ity of the MLP on large 30x30 grids). TRM with self-
attention obtains 85.3% accuracy on Maze-Hard, 44.6%
accuracy on ARC-AGI-1, and 7.8% accuracy on ARC-
AGI-2 with 7M parameters. This is significantly higher
than the 74.5%, 40.3%, and 5.0% obtained by HRM us-
ing 4 times the number of parameters (27M).
Table 4. % Test accuracy on Puzzle Benchmarks (Sudoku-
Extreme and Maze-Hard)
Method # Params Sudoku Maze
Chain-of-thought, pretrained
Deepseek R1 671B 0.0 0.0
Claude 3.7 8K ? 0.0 0.0
O3-mini-high ? 0.0 0.0
Direct prediction, small-sample training
Direct pred 27M 0.0 0.0
HRM 27M 55.0 74.5
TRM-Att (Ours) 7M 74.7 85.3
TRM-MLP (Ours) 5M/19M187.4 0.0
Table 5.% Test accuracy on ARC-AGI Benchmarks (2 tries)
Method # Params ARC-1 ARC-2
Chain-of-thought, pretrained
Deepseek R1 671B 15.8 1.3
Claude 3.7 16K ? 28.6 0.7
o3-mini-high ? 34.5 3.0
Gemini 2.5 Pro 32K ? 37.0 4.9
Grok-4-thinking 1.7T 66.7 16.0
Bespoke (Grok-4) 1.7T 79.6 29.4
Direct prediction, small-sample training
Direct pred 27M 21.0 0.0
HRM 27M 40.3 5.0
TRM-Att (Ours) 7M 44.6 7.8
TRM-MLP (Ours) 19M 29.6 2.4
6. Conclusion
We propose Tiny Recursion Models (TRM), a simple
recursive reasoning approach that achieves strong gen-
eralization on hard tasks using a single tiny network
recursing on its latent reasoning feature and progres-
sively improving its final answer. Contrary to the
Hierarchical Reasoning Model (HRM), TRM requires
no fixed-point theorem, no complex biological justi-
fications, and no hierarchy. It significantly reduces
the number of parameters by halving the number of
layers and replacing the two networks with a single
tiny network. It also simplifies the halting process,
removing the need for the extra forward pass. Over-
15M on Sudoku and 19M on Maze
8