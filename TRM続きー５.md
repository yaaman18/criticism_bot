=== PAGE 11 ===

Recursive Reasoning with Tiny Networks
Hyper-parameters and setup
All models are trained with the AdamW opti-
mizer(Loshchilov & Hutter, 2017; Kingma & Ba, 2014)
with β1=0.9,β2=0.95, small learning rate warm-
up (2K iterations), batch-size 768, hidden-size of 512,
Nsup=16 max supervision steps, and stable-max loss
(Prieto et al., 2025) for improved stability. TRM uses an
Exponential Moving Average (EMA) of 0.999. HRM
uses n= 2,T= 2 with two 4-layers networks, while
we usen=6,T=3 with one 2-layer network.
For Sudoku-Extreme and Maze-Hard, we train for 60k
epochs with learning rate 1e-4 and weight decay 1.0.
For ARC-AGI, we train for 100K epochs with learning
rate 1e-4 (with 1e-2 learning rate for the embeddings)
and weight decay 0.1. The numbers for Deepseek R1,
Claude 3.7 8K, O3-mini-high, Direct prediction, and
HRM from the Table 4 and 5 are taken from Wang et al.
(2025). Both HRM and TRM add an embedding of
shape [0, 1,D]on Sudoku-Extreme and Maze-Hard to
the input. For ARC-AGI, each puzzle (containing 2-3
training examples and 1-2 test examples) at each data-
augmentation is given a specific embedding of shape
[0, 1,D]and, at test-time, the most common answer
out of the 1000 data augmentations is given as answer.
Experiments on Sudoku-Extreme were ran with 1 L40S
with 40Gb of RAM for generally less than 36 hours.
Experiments on Maze-Hard were ran with 4 L40S with
40Gb of RAM for less than 24 hours. Experiments on
ARC-AGI were ran for around 3 days with 4 H100
with 80Gb of RAM.
Ideas that failed
In this section, we quickly mention a few ideas that
did not work to prevent others from making the same
mistake.
We tried replacing the SwiGLU MLPs by SwiGLU
Mixture-of-Experts (MoEs) (Shazeer et al., 2017; Fedus
et al., 2022), but we found generalization to decrease
massively. MoEs clearly add too much unnecessary
capacity, just like increasing the number of layers does.
Instead of back-propagating through the whole n+1
recursions, we tried a compromise between HRM 1-
step gradient approximation, which back-propagates
through the last 2 recursions. We did so by decou-
pling nfrom thenumber of last recursions kthat we
back-propagate through. For example, while n= 6
requires 7 steps with gradients in TRM, we can use
gradients for only the k= 4 last steps. However, we
found that this did not help generalization in any way,
and it made the approach more complicated. Back-propagating through the whole n+1 recursions makes
the most sense and works best.
We tried removing ACT with the option of stopping
when the solution is reached, but we found that gen-
eralization dropped significantly. This can probably
be attributed to the fact that the model is spending
too much time on the same data samples rather than
focusing on learning on a wide range of data samples.
We tried weight tying the input embedding and out-
put head, but this was too constraining and led to a
massive generalization drop.
We tried using TorchDEQ (Geng & Kolter, 2023) to
replace the recursion steps by fixed-point iteration as
done by Deep Equilibrium Models (Bai et al., 2019).
This would provide a better justification for the 1-step
gradient approximation. However, this slowed down
training due to the fixed-point iteration and led to
worse generalization. This highlights the fact that
converging to a fixed-point is not essential.
11

=== PAGE 12 ===

Recursive Reasoning with Tiny Networks
Algorithms with different number of latent
features
def latent recursion(x, z, n=6):
for i in range(n+1): # latent recursion
z = net(x, z)
return z
def deep recursion(x, z, n=6, T=3):
# recursing T−1 times to improve z (no gradients needed)
with torch.no grad():
for j in range(T−1):
z = latent recursion(x, z, n)
# recursing once to improve z
z = latent recursion(x, z, n)
return z.detach(), output head(y), Q head(y)
# Deep Supervision
for x input, y true in train dataloader:
z = z init
for step in range(N supervision):
x = input embedding(x input)
z, y hat, q hat = deep recursion(x, z)
loss = softmax cross entropy(y hat, y true)
loss += binary cross entropy(q hat, (y hat == y true))
z = z.detach()
loss.backward()
opt.step()
opt.zero grad()
if q[0]>0: # early−stopping
break
Figure 4. Pseudocode of TRM using a single- zwith deep
supervision training in PyTorch.
def latent recursion(x, y, z, n=6):
for i in range(n): # latent recursion
z[i] = net(x, y, z[0], ... , z[n−1])
y = net(y, z[0], ... , z[n−1]) # refine output answer
return y, z
def deep recursion(x, y, z, n=6, T=3):
# recursing T−1 times to improve y and z (no gradients needed)
with torch.no grad():
for j in range(T−1):
y, z = latent recursion(x, y, z, n)
# recursing once to improve y and z
y, z = latent recursion(x, y, z, n)
return (y.detach(), z.detach()), output head(y), Q head(y)
# Deep Supervision
for x input, y true in train dataloader:
y, z = y init, z init
for step in range(N supervision):
x = input embedding(x input)
(y, z), y hat, q hat = deep recursion(x, y, z)
loss = softmax cross entropy(y hat, y true)
loss += binary cross entropy(q hat, (y hat == y true))
loss.backward()
opt.step()
opt.zero grad()
if q[0]>0: # early−stopping
break
Figure 5. Pseudocode of TRM using multi-scale zwith deep
supervision training in PyTorch.Example on Sudoku-Extreme
831
9 68 7
3 5
68
6 2
74 3
9 4
2 4 1
6 2 57
Inputx
526794831
391268475
487315296
168532749
935476182
742981563
873159624
259647318
614853957
Outputy
526794831
391268475
487315296
168532749
935476182
742981563
873159624
259647318
614853957
Tokenizedz H(denotedyin TRM)
5 5494 63
4 31 465
484 3 664
9 653 54
3543 544
6 3 33588
33365 664
75 6 3366
4348 3664
Tokenizedz L(denotedzin TRM)
Figure 6. This Sudoku-Extreme example shows an input, ex-
pected output, and the tokenized zHandzL(after reversing
the embedding and using argmax) for a pretrained model.
This highlights the fact that zHcorresponds to the predicted
response, while zLis a latent feature that cannot be decoded
to a sensible output unless transformed intoz Hbyf H.
12