=== PAGE 5 ===

Recursive Reasoning with Tiny Networks
3.3. Hierarchical interpretation based on complex
biological arguments
The HRM’s authors justify the two latent variables
and two networks operating at different hierarchies
based on biological arguments, which are very far
from artificial neural networks. They even try to match
HRM to actual brain experiments on mice. While in-
teresting, this sort of explanation makes it incredibly
hard to parse out why HRM is designed the way it
is. Given the lack of ablation table in their paper, the
over-reliance on biological arguments and fixed-point
theorems (that are not perfectly applicable), it is hard
to determine what parts of HRM is helping what and
why. Furthermore, it is not clear why they use two
latent features rather than other combinations of fea-
tures.
In the next section, we show that the recursive process
can be greatly simplified and understood in a much
simpler manner that does not require any biological
argument, fixed-point theorem, hierarchical interpre-
tation, nor using two networks. It also explains why 2
is the optimal number of features (z Landz H).
def latent recursion(x, y, z, n=6):
for i in range(n): # latent reasoning
z = net(x, y, z)
y = net(y, z) # refine output answer
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
if q hat>0: # early−stopping
break
Figure 3.Pseudocode of Tiny Recursion Models (TRMs).
4. Tiny Recursion Models
In this section, we present Tiny Recursion Models
(TRMs). Contrary to HRM, TRM requires no com-
plex mathematical theorem, hierarchy, nor biological
arguments. It generalizes better while requiring only
a single tiny network (instead of two medium-size net-
works) and a single forward pass for the ACT (insteadof 2 passes). Our approach is described in Algorithm 3
and illustrated in Figure 1. We also provide an ablation
in Table 1 on the Sudoku-Extreme dataset (a dataset
of difficult Sudokus with only 1K training examples,
but 423K test examples). Below, we explain the key
components of TRMs.
Table 1. Ablation of TRM on Sudoku-Extreme comparing %
Test accuracy, effective depth per supervision step (T(n+
1)nlayers), number of Forward Passes (NFP) per optimization
step, and number of parameters
Method Acc (%) Depth NFP # Params
HRM 55.0 24 2 27M
TRM (T=3,n=6) 87.4 42 1 5M
w/ ACT 86.1 42 2 5M
w/ separatef H,fL 82.4 42 1 10M
no EMA 79.9 42 1 5M
w/ 4-layers,n=3 79.5 48 1 10M
w/ self-attention 74.7 42 1 7M
w/T=2,n=2 73.7 12 1 5M
w/ 1-step gradient 56.5 42 1 5M
4.1. No fixed-point theorem required
HRM assumes that the recursions converge to a fixed-
point for both zLandzHin order to leverage the 1-step
gradient approximation (Bai et al., 2019). This allows
the authors to justify only back-propagating through
the last two function evaluations (1 fLand 1 fH). To
bypass this theoretical requirement, we define a full
recursion process as containing nevaluations of fL
and 1 evaluation off H:
zL←f L(zL+z H+x)
...
zL←f L(zL+z H+x)
zH←f H(zL+z H).
Then, we simply back-propagate through the full re-
cursion process.
Through deep supervision, the models learns to take
any(zL,zH)and improve it through a full recursion
process, hopefully making zHcloser to the solution.
This means that by the design of the deep supervi-
sion goal, running a few full recursion processes (even
without gradients) is expected to bring us closer to the
solution. We propose to run T− 1 recursion processes
without gradient to improve (zL,zH)before running
one recursion process with backpropagation.
Thus, instead of using the 1-step gradient approxi-
mation, we apply a full recursion process containing
nevaluations of fLand 1 evaluation of fH. This re-
moves entirely the need to assume that a fixed-point
5

=== PAGE 6 ===

Recursive Reasoning with Tiny Networks
is reached and the use of the IFT theorem with 1-step
gradient approximation. Yet, we can still leverage
multiple backpropagation-free recursion processes to
improve (zL,zH). With this approach, we obtain a
massive boost in generalization on Sudoku-Extreme
(improving TRM from 56.5% to 87.4%; see Table 1).
4.2. Simpler reinterpretation ofz Handz L
HRM is interpreted as doing hierarchical reasoning
over two latent features of different hierarchies due to
arguments from biology. However, one might wonder
why use two latent features instead of 1, 3, or more?
And do we really need to justify these so-called ”hier-
archical” features based on biology to make sense of
them? We propose a simple non-biological explana-
tion, which is more natural, and directly answers the
question of why there are 2 features.
The fact of the matter is: zHis simply the current
(embedded) solution. The embedding is reversed by
applying the output head and rounding to the nearest
token using the argmax operation. On the other hand,
zLis a latent feature that does not directly correspond
to a solution, but it can be transformed into a solution
by applying zH←f H(x,zL,zH). We show an example
on Sudoku-Extreme in Figure 6 to highlight the fact
that zHdoes correspond to the solution, but zLdoes
not.
Once this is understood, hierarchy is not needed; there
is simply an input x, a proposed solution y(previously
called zH), and a latent reasoning feature z(previously
called zL). Given the input question x, current solution
y, and current latent reasoning z, the model recursively
improves its latent z. Then, given the current latent z
and the previous solution y, the model proposes a new
solution y(or stay at the current solution if its already
good).
Although this has no direct influence on the algorithm,
this re-interpretation is much simpler and natural. It
answers the question about why two features: remem-
bering in context the question x, previous reasoning
z, and previous answer yhelps the model iterate on
the next reasoning zand then the next answer y. If
we were not passing the previous reasoning z, the
model would forget how it got to the previous solu-
tion y(since zacts similarly as a chain-of-thought). If
we were not passing the previous solution y, then the
model would forget what solution it had and would
be forced to store the solution ywithin zinstead of
using it for latent reasoning. Thus, we need both yand
zseparately, and there is no apparent reason why one
would need to splitzinto multiple features.While this is intuitive, we wanted to verify whether
using more or less features could be helpful. Results
are shown in Table 2.
More features( >2): We tested splitting zinto dif-
ferent features by treating each of the nrecursions as
producing a different zifori= 1, ..., n. Then, each
ziis carried across supervision steps. The approach
is described in Algorithm 5. In doing so, we found
performance to drop. This is expected because, as dis-
cussed, there is no apparent need for splitting zinto
multiple parts. It does not have to be hierarchical.
Single feature: Similarly, we tested the idea of taking
a single feature by only carrying zHacross supervi-
sion steps. The approach is described in Algorithm 4.
In doing so, we found performance to drop. This is
expected because, as discussed, it forces the model to
store the solutionywithinz.
Thus, we explored using more or less latent variables
on Sudoku-Extreme, but found that having only yand
zlead to better test accuracy in addition to being the
simplest more natural approach.
Table 2. TRM on Sudoku-Extreme comparing % Test accu-
racy when using more or less latent features
Method # of features Acc (%)
TRMy,z(Ours) 2 87.4
TRM multi-scalez n+1=7 77.6
TRM singlez1 71.9
4.3. Single network
HRM uses two networks, one applied frequently as a
low-levelmodule fHand one applied rarely as anhigh-
levelmodule ( fH). This requires twice the number of
parameters compared to regular supervised learning
with a single network.
As mentioned previously, while fLiterates on the la-
tent reasoning feature z(zLin HRM), the goal of fH
is to update the solution y(zHin HRM) given the la-
tent reasoning and current solution. Importantly, since
z←f L(x+y+z) contains xbuty←f H(y+z) does
not contains x, the task to achieve (iterating on zversus
using zto update y) is directly specified by the inclu-
sion or lack of xin the inputs. Thus, we considered
the possibility that both networks could be replaced
by a single network doing both tasks. In doing so, we
obtain better generalization on Sudoku-Extreme (im-
proving TRM from 82.4% to 87.4%; see Table 1) while
reducing the number of parameters by half. It turns
out that a single network is enough.
6

