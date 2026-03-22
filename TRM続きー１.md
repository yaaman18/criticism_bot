=== PAGE 3 ===

Recursive Reasoning with Tiny Networks
def hrm(z, x, n=2, T=2): # hierarchical reasoning
zH, zL = z
with torch.no grad():
for i in range(nT−2):
zL = L net(zL, zH, x)
if (i + 1) % T == 0:
zH = H net(zH, zL)
# 1−step grad
zL = L net(zL, zH, x)
zH = H net(zH, zL)
return (zH, zL), output head(zH), Q head(zH)
def ACT halt(q, y hat, y true):
target halt = (y hat == y true)
loss = 0.5∗binary cross entropy(q[0], target halt)
return loss
def ACT continue(q, last step):
if last step:
target continue = sigmoid(q[0])
else:
target continue = sigmoid(max(q[0], q[1])))
loss = 0.5∗binary cross entropy(q[1], target continue)
return loss
# Deep Supervision
for x input, y true in train dataloader:
z = z init
for step in range(N sup): # deep supervision
x = input embedding(x input)
z, y pred, q = hrm(z, x)
loss = softmax cross entropy(y pred, y true)
# Adaptive computational time (ACT) using Q−learning
loss += ACT halt(q, y pred, y true)
,, qnext = hrm(z, x) # extra forward pass
loss += ACT continue(q next, step == N sup−1)
z = z.detach()
loss.backward()
opt.step()
opt.zero grad()
if q[0]>q[1]: # early−stopping
break
Figure 2. Pseudocode of Hierarchical Reasoning Models
(HRMs).
a forward pass of HRM consists of applying 6 function
evaluations, where the first 4 function evaluations are
detached from the computational graph and are not
back-propagated through. The authors uses n= 2
with T= 2 in all experiments, but HRM can be gener-
alized by allowing for an arbitrary number of L steps
(n) and recursions (T) as shown in Algorithm 2.
2.3. Fixed-point recursion with 1-step gradient
approximation
Assuming that ( zL,zH) reaches a fixed-point ( z∗
L,z∗
H)
through recursing from bothf Landf H,
z∗
L≈f L(z∗
L+z H+x)
z∗
H≈f H(zL+z∗
H),
the Implicit Function Theorem (Krantz & Parks, 2002)
with the 1-step gradient approximation (Bai et al.,
2019) is used to approximate the gradient by back-
propagating only the last fLand fHsteps. This theo-
rem is used to justify only tracking the gradients of
the last two steps (out of 6), which greatly reduces
memory demands.2.4. Deep supervision
To improve effective depth, deep supervision is used.
This consists of reusing the previous latent features
(zHandzL) as initialization for the next forward pass.
This allows the model to reason over many iterations
and improve its latent features ( zLand zH) until it
(hopefully) converges to the correct solution. At most
Nsup=16 supervision steps are used.
2.5. Adaptive computational time (ACT)
With deep supervision, each mini-batch of data sam-
ples must be used for Nsup=16 supervision steps
before moving to the next mini-batch. This is expen-
sive, and there is a balance to be reached between
optimizing a few data examples for many supervision
steps versus optimizing many data examples with less
supervision steps. To reach a better balance, a halting
mechanism is incorporated to determine whether the
model should terminate early. It is learned through
a Q-learning objective that requires passing the zH
through an additional head and running an additional
forward pass (to determine if halting now rather than
later would have been preferable). They call this
method Adaptive computational time (ACT). It is only
used during training, while the full Nsup=16 super-
vision steps are done at test time to maximize down-
stream performance. ACT greatly diminishes the time
spent per example (on average spending less than 2
steps on the Sudoku-Extreme dataset rather than the
fullNsup=16 steps), allowing more coverage of the
dataset given a fixed number of training iterations.
2.6. Deep supervision and 1-step gradient
approximations replaces BPTT
Deep supervision and the 1-step gradient approxima-
tion provide a more biologically plausible and less
computationally-expansive alternative to Backpropa-
gation Through Time (BPTT) (Werbos, 1974; Rumel-
hart et al., 1985; LeCun, 1985) for solving the temporal
credit assignment (TCA) (Rumelhart et al., 1985; Wer-
bos, 1988; Elman, 1990) problem (Lillicrap & Santoro,
2019). The implication is that HRM can learn what
would normally require an extremely large network
without having to back-propagate through its entire
depth. Given the hyperparameters used by Jang et al.
(2023) in all their experiments, HRM effectively rea-
sons over nlayers(n+ 1)TN sup=4∗(2+1)∗2∗16=
384 layers of effective depth.
3

=== PAGE 4 ===

Recursive Reasoning with Tiny Networks
2.7. Summary of HRM
HRM leverages recursion from two networks at dif-
ferent frequencies (high frequency versus low fre-
quency) and deep supervision to learn to improve
its answer over multiple supervision steps (with ACT
to reduce time spent per data example). This enables
the model to imitate extremely large depth without
requiring backpropagation through all layers. This
approach obtains significantly higher performance on
hard question-answer tasks that regular supervised
models struggle with. However, this method is quite
complicated, relying a bit too heavily on uncertain
biological arguments and fixed-point theorems that
are not guaranteed to be applicable. In the next sec-
tion, we discuss those issues and potential targets for
improvements in HRM.
3. Target for improvements in Hierarchical
Reasoning Models
In this section, we identify key targets for improve-
ments in HRM, which will be addressed by our pro-
posed method, Tiny Recursion Models (TRMs).
3.1. Implicit Function Theorem (IFT) with 1-step
gradient approximation
HRM only back-propagates through the last 2 of the 6
recursions. The authors justify this by leveraging the
Implicit Function Theorem (IFT) and one-step approx-
imation, which states that when a recurrent function
converges to a fixed point, backpropagation can be
applied in a single step at that equilibrium point.
There are concerns about applying this theorem to
HRM. Most importantly, there is no guarantee that
a fixed-point is reached. Deep equilibrium models
normally do fixed-point iteration to solve for the fixed
point z∗=f(z∗)(Bai et al., 2019). However, in the case
of HRM, it is not iterating to the fixed-point but simply
doing forward passes of fLand fH. To make matters
worse, HRM is only doing 4 recursions before stopping
to apply the one-step approximation. After its first
loop of two fLand 1 fHevaluations, it only apply a
single fLevaluation before assuming that a fixed-point
is reached for both zLand zH(z∗
L=f L(z∗
L+z H+x)
and z∗
H=f H(z∗
L+z∗
H)). Then, the one-step gradient
approximation is applied to both latent variables in
succession.
The authors justify that a fixed-point is reached by
depicting an example with n= 7 and T= 7 where
the forward residuals is reduced over time (Figure 3
in Wang et al. (2025)). Even in this setting, which isdifferent from the much smaller n= 2 and T= 2 used
in every experiment of their paper, we observe the
following:
1.the residual for zHis clearly well above 0 at every
step
2.the residual for zLonly becomes closer to 0 after
many cycles, but it remains significantly above 0
3.z Lis very far from converged after one fLevalu-
ation at Tcycles, which is when the fixed-point
is assumed to be reached and the 1-step gradient
approximation is used
Thus, while the application of the IFT theorem and
1-step gradient approximation to HRM has some basis
since the residuals do generally reduce over time, a
fixed point is unlikely to be reached when the theorem
is actually applied.
In the next section, we show that we can bypass the
need for the IFT theorem and 1-step gradient approxi-
mation, thus bypassing the issue entirely.
3.2. Twice the forward passes with Adaptive
computational time (ACT)
HRM uses Adaptive computational time (ACT) during
training to optimize the time spent of each data sam-
ple. Without ACT, Nsup=16 supervision steps would
be spent on the same data sample, which is highly in-
efficient. They implement ACT through an additional
Q-learning objective, which decides when to halt and
move to a new data sample rather than keep iterating
on the same data. This allows much more efficient
use of time especially since the average number of su-
pervision steps during training is quite low with ACT
(less than 2 steps on the Sudoku-Extreme dataset as
per their reported numbers).
However, ACT comes at a cost. This cost is not directly
shown in the HRM’s paper, but it is shown in their of-
ficial code. The Q-learning objective relies on a halting
loss and a continue loss. The continue loss requires an
extra forward pass through HRM (with all 6 function
evaluations). This means that while ACT optimizes
time more efficiently per sample, it requires 2 forward
passes per optimization step. The exact formulation is
shown in Algorithm 2.
In the next section, we show that we can bypass the
need for two forward passes in ACT.
4