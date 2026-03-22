Making ERIE a Self‑Maintaining Autopoietic‑Like Agent
Executive summary
ERIE’s current blueprint (TRM‑A world model with uncertainty output, TRM‑B boundary, and a future GNW-like integration layer) already covers representation (world modeling) and segregation (self/non‑self separation). What is still missing—based on your own gap list—is the closed control loop that keeps ERIE “alive” in a concrete, testable sense: it must (i) hold explicit belief states, (ii) update them via an explicit inference loop, (iii) choose actions under an action-conditioned transition model, and (iv) maintain a set of viability variables within a viable region, with explicit death/failure criteria when constraints are violated. This is exactly the engineering decomposition used across (a) Active Inference “agent” implementations (belief update + policy inference + action sampling), (b) viability theory and safe-control formalisms (viability kernel, failure set, action shielding), and (c) autopoiesis/protocell/artificial-life lines (organizational closure and boundary-maintaining loops). 

A source-grounded way to “complete” ERIE as a self-maintaining machine is to combine three mathematically compatible layers:

Belief dynamics layer (Active Inference / predictive coding): implement explicit belief states (posterior over latent states) and belief updates (precision-weighted prediction errors or message passing), plus policy inference and action selection (e.g., discrete POMDP active inference as in step-by-step tutorials and libraries, or deep active inference for neural world models). 
Viability constraints layer (viability theory / safe control): define a constraint set over internal variables (“alive set”), define a failure set, and reason about the viability kernel (states from which the agent can keep constraints satisfied via some admissible control). This gives a rigorous, implementation-friendly “what decays / what is death” answer. 
Organizational closure layer (autopoiesis / artificial life / protocells): define the minimal closed loop that continuously regenerates (or repairs) the structures that realize the boundary and the internal conditions needed for ongoing inference/action (e.g., boundary integrity variable + maintenance actions + resource uptake). Autopoiesis is operationalized here as self-producing constraints, not as metaphysics. 
Finally, the precision question in your gap list (#7) has a clean bridge: in the predictive processing / free-energy tradition, precision is explicitly equated with gain on prediction error units, and this claim is supported not only conceptually but also in concrete process models and empirical modeling work. This is the most defensible route for making TRM‑A’s uncertainty output “do work” in an actual belief update loop. 

Literature map for ERIE’s missing elements
This section maps each missing ERIE element to (i) theories worth using, (ii) representative primary/review sources, and (iii) concrete ways to reuse them in ERIE.

ERIE gap	Theories / math that directly apply	Representative sources (primary/review)	How it plugs into ERIE (engineering translation)
Belief state + belief update loop	Active Inference (POMDP), predictive coding / variational inference	Karl Friston “Active inference and learning”; step-by-step active inference POMDP tutorial; pymdp library paper; predictive coding under FEP; free-energy principle overview 
Implement explicit posterior beliefs (states, optionally parameters) and explicit update steps (message passing / gradient flows). Treat TRM‑A as the generative model component (or as the learned transition/likelihood inside one).
Viability variables (“what decays”)	Homeostasis/allostasis, interoceptive inference; viability constraints and kernels	Pezzulo et al. on homeostasis under active inference; Barrett et al. (allostasis/interoception); Seth & Friston (interoceptive inference); viability theory book (viability kernel); safe RL/viability kernel papers 
Introduce explicit internal variables with target ranges (setpoints/densities). Define alive set and failure set. Couple these variables into preferences/priors (active inference) and constraints (viability kernel).
Minimal action space + action-conditioned transition	POMDP control states; policy inference; action shielding under constraints	Friston “Active inference and learning”; pymdp (policy inference, action sampling); deep active inference; viability-based admissible actions (shielding) 
Define a minimal discrete action basis (intake, avoid, repair, sense) or continuous controls. Make TRM‑A explicitly action-conditioned: (p(s_{t+1}\mid s_t,a_t)). Add a “shield” that prevents actions that violate viability constraints.
Sensory-active boundary implementation	Markov blankets (engineering interface); process boundary variables	“Life as we know it”; “Markov blankets of life” (autonomy + active inference); “Free Energy Principle and Artificial Agency” review 
Implement boundary as explicit interface variables: sensory channels in, active channels out. Enforce no direct coupling internal↔external except via these channels. Make TRM‑B the enforcing/monitoring mechanism for this separation plus integrity variables.
Environment as resource/threat/learning opportunity (not only reward)	Allostasis & stress as control of energy resources; expected free energy (epistemic + pragmatic)	“Active Inference, homeostatic regulation…” review; stress active inference account; deep active inference (homeostatic motivation) 
Simulate environment with explicit resources (replenish viability variables), threats (damage integrity), and uncertainty pockets (learning opportunity). Use expected free energy decomposition to couple exploration to survival.
Minimal autopoietic “closed loop”	Autopoiesis operational closure; protocells / chemoton-inspired closure; artificial life analysis of boundary-maintaining patterns	Protocell models review; “Systems of Creation” review; Beer operationalizing autopoiesis in Game of Life; autopoiesis origin ALife paper 
Define closure as: (i) boundary integrity + (ii) internal maintenance variable + (iii) processes/actions that regenerate them using environment resources. This becomes the minimal “self-producing constraint” loop in software.
Precision as update gain in actual belief updates	Precision-weighted prediction errors; synaptic gain interpretation; precision control	Feldman & Friston attention/uncertainty/free-energy; predictive coding under FEP; free-energy principle overview; DCM precision & synaptic gain paper 
Use TRM‑A uncertainty outputs to compute precision proxies, then multiply prediction errors by these precisions in the update equations. Add expected-precision control as a separate modulatory state (attention-like).

Most important papers best 10–20 with ERIE relevance and tags
Below are 18 high-leverage sources emphasized for (a) primary literature weight, (b) direct implementability, and (c) coverage of your missing elements. Each item includes: (i) what it claims, (ii) which ERIE gap it fills, (iii) theory vs implementation tilt, and (iv) tags.

Friston et al., “Active inference and learning” (2016) 

What it claims: active inference gives a unified account of perception/action/learning via policy optimization and variational free energy; discusses habitual vs goal-directed behavior under policy inference. 

ERIE gaps: (1) belief update loop, (3) policy/action space formalization, (7) where precision-like quantities appear in inference/control. 

Tilt: theory + process model (highly reusable). 

Tags: active inference, precision, implementation-friendly

Smith et al., step-by-step Active Inference POMDP tutorial (2022) 

What it claims: provides practical construction and simulation of POMDP active inference models and fitting behavior; explicitly addresses how to build/run these models in practice. 

ERIE gaps: (1) concrete belief update loop scaffolding; (3) minimal action space and policy inference in discrete form. 

Tilt: implementation-oriented tutorial. 

Tags: active inference, implementation-friendly

Heins et al., “pymdp: A Python library for active inference in discrete state spaces” (2022) 

What it claims: provides a concrete software architecture for state inference and policy inference (e.g., infer_states(), infer_policies(), action sampling) in active inference agents. 

ERIE gaps: (1) belief update loop, (3) action/policy infrastructure, (7) precision-weighted updates (discrete message passing implementations). 

Tilt: implementation-first. 

Tags: active inference, implementation-friendly

Kai Ueltzhöffer, “Deep Active Inference” (2017, arXiv) 

What it claims: integrates active inference with deep latent variable models and recurrent nets; minimizes variational free energy of sensations, motivated by a homeostatic argument; demonstrates learning a generative model and sampling beliefs. 

ERIE gaps: (1) belief update loop, (3) action-conditioned transitions with learned models, (5) environment as structured sensory stream, (7) using uncertainty in inference loops. 

Tilt: theory-to-implementation bridge. 

Tags: active inference, embodiment, implementation-friendly

Giovanni Pezzulo et al., “Active Inference, homeostatic regulation and adaptive behavioural control” (2015) 

What it claims: reviews modeling homeostatic regulation and adaptive control within active inference, aiming to connect active inference with behavior-control traditions. 

ERIE gaps: (2) viability variables, (5) environment as resources/threats in homeostatic control framing, (6) minimal closure story at the level of regulation. 

Tilt: integrative review with computational orientation. 

Tags: viability, active inference, embodiment

Lisa Feldman Barrett et al., “An active inference theory of allostasis and interoception” (2016) 

What it claims: places metabolism/energy regulation (allostasis) and interoception at the core; integrates predictive coding/active inference accounts of interoception with a broader theory of mind. 

ERIE gaps: (2) viability variables (metabolic/energy-like variables), (5) reframing environment as conditions for regulation, not reward-only. 

Tilt: theory + synthesis; useful for defining “what decays.” 

Tags: viability, active inference, embodiment

Anil Seth & Friston, “Active interoceptive inference and the emotional brain” (2016) 

What it claims: frames bodily regulation as descending predictions enslaving autonomic reflexes; highlights interoceptive inference as control-relevant prediction. 

ERIE gaps: (2) viability variables + regulation loops; (6) minimal closure interpretation as continuous re-instantiation of controlled internal conditions. 

Tilt: theory with strong mechanistic commitments. 

Tags: viability, active inference, embodiment

Feldman & Friston, “Attention, Uncertainty, and Free-Energy” (2010) 

What it claims: argues attention can be seen as inference about precision; in predictive coding schemes precision is encoded by synaptic gain of prediction-error units; includes process simulations. 

ERIE gaps: (7) inferential precision as update gain; also supports precision-gated selection (useful for GNW-gating later). 

Tilt: mechanistic + computational. 

Tags: precision, predictive coding, implementation-friendly

Friston, “Predictive coding under the free-energy principle” (2009/2010 accessible as PDF/PMC) 

What it claims: provides explicit predictive coding / variational message passing formulation; frames perception as model inversion under a hierarchical dynamical generative model. 

ERIE gaps: (1) belief updates; (7) where precision weighting enters the update equations under Laplace assumptions. 

Tilt: primary mathematical source. 

Tags: predictive coding, precision, implementation-friendly

Friston, “The free-energy principle: a unified brain theory?” (2010 PDF) 

What it claims: summarizes FEP; explicitly states that in predictive coding precision modulates prediction error amplitude and corresponds to synaptic gain. 

ERIE gaps: (7) conceptual justification for precision-as-gain; supports putting inferred precision into update loops (not only logging uncertainty). 

Tilt: foundational synthesis. 

Tags: precision, predictive coding

Brown et al., “Dynamic causal modelling of precision and synaptic gain …” (2012) 

What it claims: tests the hypothesis that changes in sensory precision map to gain/excitability changes; uses DCM to relate changes (e.g., contrast) to precision/gain in cortical models. 

ERIE gaps: (7) supports “precision→gain” as an empirically testable mechanistic mapping, motivating explicit gain modulation modules. 

Tilt: empirical modeling anchored to predictive coding. 

Tags: precision, predictive coding

Jean-Pierre Aubin; Aubin, Bayen, Saint‑Pierre, “Viability Theory: New Directions” (Springer, 2011) 

What it claims: develops mathematical/algorithmic methods for evolutions under viability constraints (including computation/approximation of viability kernels and regulation maps). 

ERIE gaps: (2) viability variables formalization, (4) “alive set” and viability kernel, (5) death/failure as constraint violation, (3) action as regulation map. 

Tilt: formal math + algorithms (very reusable). 

Tags: viability, implementation-friendly

Turriff & Broucke, viability kernel construction for nonlinear control systems (2009 PDF) 

What it claims: proposes methodology to construct viability kernels for nonlinear control systems; frames viability as enforcing evolution in a safe set via control. 

ERIE gaps: (2) viability kernel computation ideas; (5) failure as leaving safe set; (3) action admissibility. 

Tilt: control-theory implementability. 

Tags: viability, implementation-friendly

“Safe Value Functions” (paper emphasizing failure set, viability kernel as largest safe set) 

What it claims: treats failure sets and viability kernel explicitly; uses viability kernel as “largest safe set,” reconciles failure within finite time by transitioning to sink states. 

ERIE gaps: (5) death/failure criterion design; (2) viability set definition; (3) action selection under safety. 

Tilt: implementation-friendly framing for RL/control. 

Tags: viability, implementation-friendly

Kirchhoff et al., “The Markov blankets of life: autonomy, active inference and the free energy principle” (2018) 

What it claims: discusses autonomous organization and boundaries via Markov blankets under active inference; emphasizes statistical boundary concept and nested blankets. 

ERIE gaps: (4) sensory-active boundary conceptual grounding + autonomy framing; (6) tying boundary to self-maintaining organization (with the usual caveats). 

Tilt: theory bridging autonomy and AI. 

Tags: boundary, active inference, autopoiesis (bridge), embodiment

Friston, “Life as we know it” (2013) 

What it claims: argues Markov blankets imply internal states appear to minimize free energy of blanket states; frames living systems as Markov-blanketed systems engaging in active inference-like dynamics. 

ERIE gaps: (4) boundary as interface; (6) self-maintenance narrative; (8) environment coupling as condition for persistence. 

Tilt: foundational, but must be used carefully to avoid over-claiming. 

Tags: boundary, active inference, autopoiesis

Stano et al., “Protocells Models in Origin of Life and Synthetic Biology” (2015 review) 

What it claims: reviews protocell research aimed at building compartmentalized systems with life-like organization; explicitly references chemoton and autopoietic frameworks as theoretical anchors in protocell modeling. 

ERIE gaps: (6) minimal closure patterns; (2) “what decays” analogs (membrane integrity, metabolic flux); (8) non-trivial environment dependence. 

Tilt: review with design patterns (good inspiration). 

Tags: autopoiesis, artificial life, implementation-friendly

Beer, “Characterizing autopoiesis in the Game of Life” (2015) 

What it claims: treats autopoiesis as organizational closure and explores how recurrent patterns can be analyzed as self-constructing networks maintaining boundaries in a cellular automaton. 

ERIE gaps: (6) operational/autopoiesis-like closure definition; (4) boundary integrity as a maintained structure; informs measurable “closure proxies.” 

Tilt: operationalization-heavy (useful even if substrate differs). 

Tags: autopoiesis, artificial life

(Additional ALife crosspoint references you asked for, though less directly “plug-and-play” than the above):

Chan’s “Lenia” (continuous CA with many resilient/adaptive lifeforms) is very relevant as a sandbox for “boundary integrity” and “metastability” experiments. 
Particle Lenia’s “energy-based formulation” is a potential bridge to variational/energy-based control intuitions, but the mapping to FEP should be treated as an analogy unless you make it explicit and testable. 
England’s nonequilibrium “statistical physics of self-replication” is a good theoretical constraint on what “self-maintenance” implies thermodynamically, and can motivate ERIE’s “metastable viability basin” design. 
ERIE design candidates aligned to the literature
This section proposes concrete design candidates for the missing pieces, expressed as (a) what to represent, (b) how to update/control, and (c) how to test—grounded in the research threads above.

Viability variable candidates
A recurring theme across active inference homeostasis/allostasis accounts is that “survival” is implemented as maintaining internal physiological variables (often energy/metabolic) within expected ranges, and these expectations shape perception and action. 
 In viability theory, the analogous object is a constraint set in state space that must not be violated; the viability kernel is the set of states from which some control policy can keep constraints satisfied. 

For ERIE, viable variables that map cleanly onto software and your architecture are:

Energy / resource budget (scalar or low-dimensional vector): decreases by computation + actuation; replenished by “intake” actions in resource-rich regions. This matches the allostasis/interoception framing (energy regulation as a core driver) and is the most direct “what decays.” 
Boundary integrity: a scalar capturing TRM‑B’s ability to maintain a clean sensory-active interface and resist “leaks” (e.g., adversarial perturbations or environment hazards that corrupt sensors/actuators). Autopoiesis operationalizations emphasize boundary maintenance as central; Markov-blanket/autonomy discussions tie boundaries to maintaining organization over time. 
Model integrity / internal order proxy: a variable tied to whether the belief update loop remains within a regime that produces coherent posteriors (e.g., bounded surprisal/free-energy proxy or bounded calibration loss). This matches the “metastable process” narrative in FEP and connects to the idea that living systems occupy stable attractor-like sets. 
A practical engineering constraint (aligned with viability-kernel thinking) is to separate (i) viability constraints from (ii) task preferences: keep the viability variables as hard constraints or lexicographically prioritized objectives, rather than folding them into a single reward. This mirrors safe-control/viability-kernel RL approaches that enforce hard safety independently of reward. 

Death / failure / collapse criteria candidates
Across viability-based control, a standard formal move is: define a failure set (X_F) (constraint violation), an alive set (constraint-satisfying subset), and treat “death” as entering (X_F) (often with absorbing terminal dynamics). 
 This is directly implementable and gives you a mathematically legible “death.” 

For ERIE, candidate death criteria:

Constraint breach: any viability variable leaves its allowable interval for longer than (T) steps (hysteresis avoids “single-frame noise deaths”). This matches viability theory’s constraint framing. 
Boundary collapse: TRM‑B integrity below threshold → sensory-active boundary no longer enforces conditional independence interface assumptions (e.g., sensor corruption or actuator hijack). This corresponds to the boundary-maintenance emphasis in autopoiesis operationalizations. 
Inference collapse: belief update diverges (e.g., posterior becomes numerically unstable; predictive error explodes under precision miscalibration). This connects to the “precision as gain” idea: high precision on wrong channels can destabilize inference, meaning precision control is itself a survival-critical mechanism. 
Minimal action space candidates (aligned with survival, not just reward)
Active inference implementations typically treat actions as control states that affect transitions and observations; viability-based control treats actions as moves that must keep the agent in safe sets. 
 A minimal action basis that explicitly supports self-maintenance is:

Intake / reject: actions that change energy budget by interacting with resources vs toxins (intake can raise energy but risk boundary damage). Motivated by allostasis/metabolism accounts. 
Approach / avoid: movement actions (or attention allocation actions) that move toward resources and away from threats; viability kernel framing makes this a “stay inside alive set” problem. 
Repair / shield: explicit actions that increase boundary integrity or reduce hazard exposure, operationalizing the boundary-maintenance aspect of autopoiesis. 
Active sensing (epistemic action): actions that reduce uncertainty where uncertainty reduction is instrumentally required for maintaining viability (e.g., find resources). This fits the “deep active inference” and active inference policy selection framing where sampling sensations under a generative model is part of action. 
Belief update loop candidates (how to make TRM‑A more than a predictor)
Because ERIE already has TRM‑A outputs (pred_mean, pred_logvar), the key missing step is to ensure these outputs participate in explicit inference updates:

Discrete-state active inference loop (fastest to ship): implement a minimal POMDP agent using established recipes and libraries, then replace (or hybridize) the transition/observation models with TRM‑A surrogates. This directly addresses the “belief update loop” gap with proven scaffolding (infer_states, infer_policies, sample_action). 
Predictive-coding-style continuous inference (closest to your precision interest): implement Laplace/predictive-coding updates where prediction errors are multiplied by precision (gain), explicitly reflecting the primary literature’s process theory. This is the most defensible route for “inferential precision enters actual belief updates.” 
Deep active inference loop (closest to TRM‑A neural world model): follow deep active inference designs that explicitly integrate variational deep generative models with active inference control and a homeostatic motivational framing. This directly reuses your neural world-model approach while adding the missing belief/action loop. 
Boundary implementation candidates (sensory-active interface as engineering object)
The Markov blanket framing used in active inference work treats boundaries as a statistical separation mediated by sensory and active states, and autonomy-oriented work emphasizes nested/self-sustaining boundary structures. 
 For ERIE engineering:

Interface-variable enforcement: represent the boundary explicitly as a pair of variable sets (s^{sens}) (incoming) and (s^{act}) (outgoing), and enforce at the software architecture level that internal states do not directly access external states except through (s^{sens}), and do not affect external except through (s^{act}). 
Boundary integrity as a monitored latent: treat integrity as a latent variable inferred from sensor/actuator consistency checks (e.g., mismatch between predicted and actual sensory consequences of action). This matches the idea that boundaries are functional/statistical and must be inferred/maintained. 
Caveat (important for engineering honesty): autonomy/Markov blanket papers explicitly note that statistical blankets need not match physical boundaries; for ERIE this implies you should claim “engineered interface blanket” rather than “the blanket is the self.” 
What to adopt now vs what to hold
Adopt now (high leverage, implementation-friendly, low metaphysical load)
Explicit belief update loop with policy inference (pymdp / tutorial patterns)
This directly fills gap (1) and (3) with well-defined computational steps and a reference implementation style. 

Viability defined as constraints + failure set (viability kernel framing)
This fills gaps (2), (4), (5) with a clean, testable definition: alive set, failure set, and optionally approximations to the viability kernel plus action shielding. 

Precision as update gain in inference (predictive coding + attention-as-precision control)
This fills gap (7) and makes TRM‑A’s uncertainty output operational: compute precision proxies and weight prediction errors; add a separate “expected precision” modulatory variable rather than collapsing everything into pred_logvar. 

Environment explicitly modeled as resources/threats/uncertainty pockets
Use the allostasis/stress framing to motivate energy/resource dynamics and threat costs, and active inference action selection to balance exploration and viability. This fills gap (5) and concretizes gap (2). 

Hold / defer (valuable but easy to over-claim or hard to translate directly)
Strong autopoiesis claims as “sufficient for life”
Autopoiesis is useful as a design heuristic for closure and boundary maintenance, but strong interpretations can drift into philosophy without discriminative engineering tests. Use operational/autopoiesis-inspired criteria (closure proxies) rather than metaphysical equivalence. 

Markov blankets as metaphysical selfhood
Markov blanket work supports boundary-as-interface design, but blanket talk can be overextended. ERIE should treat this as an architectural interface formalism and a statistical diagnostic, not as a proof of “self.” 

Lenia / Particle Lenia as direct self-maintenance implementations
These are excellent ALife sandboxes and can inspire metastability/boundary-integrity metrics, but the mapping into an active-inference belief/action loop is not direct unless you build it explicitly. 

Five papers to prioritize for ERIE’s next stage
These five are selected because each one directly unlocks a missing ERIE element with minimal translation overhead, and together they form a coherent “implementation reading order.”

Heins et al., pymdp (2022) 

Why most important: it provides an immediately reusable agent skeleton (state inference + policy inference + action sampling) that operationalizes the belief-update loop you identified as missing. 

What to read first: the “statement of need” and the sections describing infer_states() / infer_policies() and how generative model matrices are used. 

Implementation-order impact: implement this loop first, even with toy dynamics; then progressively replace its generative components with TRM‑A surrogates or hybrids.

Ueltzhöffer, Deep Active Inference (2017, arXiv) 

Why most important: it is the closest published bridge between deep generative models (like your TRM‑A design instincts) and an active-inference control loop, explicitly motivated by homeostasis. 

What to read first: the problem setup (“minimize variational free energy of sensations”), the generative model + variational posterior architecture, and action selection mechanism. 

Implementation-order impact: after you have a discrete loop running, use this as the template for a neural TRM‑A–driven loop.

Feldman & Friston, “Attention, Uncertainty, and Free‑Energy” (2010) 

Why most important: it gives the cleanest justification (with explicit process modeling) for treating precision as a gain-like control quantity that gates error influence—exactly your missing element (7). 

What to read first: the passages that identify precision with synaptic gain and show state-dependent precision explaining attentional gating/competition. 

Implementation-order impact: implement precision-weighted updates early, because wrong precision handling can destabilize inference and invalidate uncertainty outputs.

Aubin, Bayen, Saint‑Pierre, “Viability Theory: New Directions” (Springer 2011) 

Why most important: it gives the most rigorous answer to “what decays / what is death”—as constraint satisfaction over time—with explicit algorithms and the viability kernel concept.

What to read first: the definitions of viability constraints, viability kernel, and regulation maps (feedback selecting viable evolutions).

Implementation-order impact: define ERIE’s alive set + failure set early, because this clarifies action-space design (shielding/repair/intake must exist) and prevents “reward-only drift.”

Pezzulo et al., “Active Inference, homeostatic regulation and adaptive behavioural control” (2015)

Why most important: it directly targets your objective—self-maintaining agents—by synthesizing homeostatic regulation inside active inference and connecting it to behavioral control traditions.

What to read first: the sections that specify how homeostatic variables and adaptive control are represented in an active-inference framing.

Implementation-order impact: use it to choose your viability variables and to justify environment-as-resource/threat design rather than reward-punishment-only.

Reference set grouped by your priority themes
A (Viability / viability kernel / viability theory): Aubin et al. (book) 
; viability kernel construction method 
; safe RL/viability kernel as largest safe set 
; recent viability-based shielding framing 

B (Homeostasis / allostasis / self-maintenance): Pezzulo et al. 
; Barrett et al. 
; Seth & Friston 
; stress/allostatic collapse under active inference 
; homeostatic RL (useful but reward-adjacent) 

C (Autopoiesis / artificial life): Beer autopoiesis operationalization 
; autopoiesis origin perspective (ALife) 
; protocell models review (chemoton/autopoiesis references) 
; systems of creation / protocell criteria review 

D (Active inference for self-maintaining agents): Friston active inference & learning 
; deep active inference 
; active inference agency overview (arXiv) 

E (Predictive coding / precision as gain): Feldman & Friston 
; predictive coding under FEP 
; FEP unified brain theory 
; DCM precision & gain 

F (Embodied AI / adaptive agents / survival variables): viability-based safety shielding (arXiv) 
; safe control with viability kernel definitions 
; active inference practice tutorials/libraries 

G (Artificial cells / protocells / metabolism-inspired control): protocell models review 
; “Systems of Creation” 

H (Dynamical systems / nonequilibrium thermodynamics / metastability): Friston “Life as we know it” 
; England statistical physics of self-replication 

Lenia / Particle Lenia / ALife intersections: Lenia primary arXiv paper by Bert Wang-Chak Chan 
; Particle Lenia engineering description (Google Research) 
; Flow/other Lenia-related work as exploration sandbox 