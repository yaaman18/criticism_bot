=== PAGE 9 ===

Recursive Reasoning with Tiny Networks
all, TRM is much simpler than HRM, while achieving
better generalization.
While our approach led to better generalization on 4
benchmarks, every choice made is not guaranteed to
be optimal on every dataset. For example, we found
that replacing the self-attention with an MLP worked
extremely well on Sudoku-Extreme (improving test ac-
curacy by 10%), but poorly on other datasets. Different
problem settings may require different architectures
or number of parameters. Scaling laws are needed
to parametrize these networks optimally. Although
we simplified and improved on deep recursion, the
question of why recursion helps so much compared
to using a larger and deeper network remains to be
explained; we suspect it has to do with overfitting, but
we have no theory to back this explaination. Not all
our ideas made the cut; we briefly discuss some of the
failed ideas that we tried but did not work in Section 6.
Currently, recursive reasoning models such as HRM
and TRM are supervised learning methods rather than
generative models. This means that given an input
question, they can only provide a single deterministic
answer. In many settings, multiple answers exist for a
question. Thus, it would be interesting to extend TRM
to generative tasks.
Acknowledgements
Thank you Emy Gervais for your invaluable support
and extra push. This research was enabled in part
by computing resources, software, and technical as-
sistance provided by Mila and the Digital Research
Alliance of Canada.
References
ARC Prize Foundation. The Hidden Drivers of HRM’s
Performance on ARC-AGI. https://arcprize.
org/blog/hrm-analysis , 2025a. [Online; ac-
cessed 2025-09-15].
ARC Prize Foundation. ARC-AGI Leaderboard.
https://arcprize.org/leaderboard , 2025b.
[Online; accessed 2025-09-24].
Bai, S., Kolter, J. Z., and Koltun, V . Deep equilibrium
models.Advances in neural information processing
systems, 32, 2019.
Bai, X. and Melas-Kyriazi, L. Fixed point diffusion
models. InProceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pp. 9430–
9440, 2024.
Brock, A., Donahue, J., and Simonyan, K. Large scalegan training for high fidelity natural image synthe-
sis.arXiv preprint arXiv:1809.11096, 2018.
Chollet, F. On the measure of intelligence.arXiv
preprint arXiv:1911.01547, 2019.
Chollet, F., Knoop, M., Kamradt, G., Landers, B.,
and Pinkard, H. Arc-agi-2: A new challenge
for frontier ai reasoning systems.arXiv preprint
arXiv:2505.11831, 2025.
Chowdhery, A., Narang, S., Devlin, J., Bosma, M.,
Mishra, G., Roberts, A., Barham, P ., Chung, H. W.,
Sutton, C., Gehrmann, S., et al. Palm: Scaling lan-
guage modeling with pathways.Journal of Machine
Learning Research, 24(240):1–113, 2023.
Dillion, T. Tdoku: A fast sudoku solver and gener-
ator. https://t-dillon.github.io/tdoku/ ,
2025.
Elman, J. L. Finding structure in time.Cognitive science,
14(2):179–211, 1990.
Fedus, W., Zoph, B., and Shazeer, N. Switch transform-
ers: Scaling to trillion parameter models with simple
and efficient sparsity.Journal of Machine Learning Re-
search, 23(120):1–39, 2022.
Geng, Z. and Kolter, J. Z. Torchdeq: A library for deep
equilibrium models.arXiv preprint arXiv:2310.18605,
2023.
Hendrycks, D. and Gimpel, K. Gaussian error linear
units (gelus).arXiv preprint arXiv:1606.08415, 2016.
Jang, Y., Kim, D., and Ahn, S. Hierarchical graph
generation with k2-trees. InICML 2023 Workshop on
Structured Probabilistic Inference Generative Modeling,
2023.
Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B.,
Chess, B., Child, R., Gray, S., Radford, A., Wu, J.,
and Amodei, D. Scaling laws for neural language
models.arXiv preprint arXiv:2001.08361, 2020.
Kingma, D. P . and Ba, J. Adam: A method for stochas-
tic optimization.arXiv preprint arXiv:1412.6980,
2014.
Krantz, S. G. and Parks, H. R.The implicit function
theorem: history, theory, and applications. Springer
Science & Business Media, 2002.
LeCun, Y. Une procedure d’apprentissage ponr reseau
a seuil asymetrique.Proceedings of cognitiva 85, pp.
599–604, 1985.
9

=== PAGE 10 ===

Recursive Reasoning with Tiny Networks
Lehnert, L., Sukhbaatar, S., Su, D., Zheng, Q., Mcvay, P .,
Rabbat, M., and Tian, Y. Beyond a*: Better planning
with transformers via search dynamics bootstrap-
ping.arXiv preprint arXiv:2402.14083, 2024.
Lillicrap, T. P . and Santoro, A. Backpropagation
through time and the brain.Current opinion in neuro-
biology, 55:82–89, 2019.
Loshchilov, I. and Hutter, F. Decoupled weight decay
regularization.arXiv preprint arXiv:1711.05101, 2017.
Moskvichev, A., Odouard, V . V ., and Mitchell, M. The
conceptarc benchmark: Evaluating understanding
and generalization in the arc domain.arXiv preprint
arXiv:2305.07141, 2023.
Palm, R., Paquet, U., and Winther, O. Recurrent re-
lational networks.Advances in neural information
processing systems, 31, 2018.
Park, K. Can convolutional neural networks
crack sudoku puzzles? https://github.com/
Kyubyong/sudoku, 2018.
Prieto, L., Barsbey, M., Mediano, P . A., and Birdal, T.
Grokking at the edge of numerical stability.arXiv
preprint arXiv:2501.04697, 2025.
Rumelhart, D. E., Hinton, G. E., and Williams, R. J.
Learning internal representations by error propaga-
tion. Technical report, 1985.
Shazeer, N. Glu variants improve transformer.arXiv
preprint arXiv:2002.05202, 2020.
Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le,
Q., Hinton, G., and Dean, J. Outrageously large neu-
ral networks: The sparsely-gated mixture-of-experts
layer.arXiv preprint arXiv:1701.06538, 2017.
Snell, C., Lee, J., Xu, K., and Kumar, A. Scaling
llm test-time compute optimally can be more effec-
tive than scaling model parameters.arXiv preprint
arXiv:2408.03314, 2024.
Song, Y. and Ermon, S. Improved techniques for train-
ing score-based generative models.Advances in
neural information processing systems, 33:12438–12448,
2020.
Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu,
Y. Roformer: Enhanced transformer with rotary
position embedding.Neurocomputing, 568:127063,
2024.
Tolstikhin, I. O., Houlsby, N., Kolesnikov, A., Beyer,
L., Zhai, X., Unterthiner, T., Yung, J., Steiner, A.,
Keysers, D., Uszkoreit, J., et al. Mlp-mixer: Anall-mlp architecture for vision.Advances in neural
information processing systems, 34:24261–24272, 2021.
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J.,
Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin,
I. Attention is all you need.Advances in neural
information processing systems, 30, 2017.
Wang, G., Li, J., Sun, Y., Chen, X., Liu, C., Wu, Y.,
Lu, M., Song, S., and Yadkori, Y. A. Hierarchical
reasoning model.arXiv preprint arXiv:2506.21734,
2025.
Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F.,
Chi, E., Le, Q. V ., Zhou, D., et al. Chain-of-thought
prompting elicits reasoning in large language mod-
els.Advances in neural information processing systems,
35:24824–24837, 2022.
Werbos, P . Beyond regression: New tools for predic-
tion and analysis in the behavioral sciences.PhD
thesis, Committee on Applied Mathematics, Harvard
University, Cambridge, MA, 1974.
Werbos, P . J. Generalization of backpropagation with
application to a recurrent gas market model.Neural
networks, 1(4):339–356, 1988.
Zhang, B. and Sennrich, R. Root mean square layer
normalization.Advances in Neural Information Pro-
cessing Systems, 32, 2019.
10