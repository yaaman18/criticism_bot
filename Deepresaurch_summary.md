# Precision and Conscious-Access Operationalization for ERIE

## Executive summary

This report treats your two target properties—(i) **precision (belief certainty / reliability weighting)** and (ii) **consciousness-like property in the sense of conscious access / global availability**—as *engineering objects*: definable in computational terms, representable in an architecture, and testable with operational criteria. The synthesis is grounded in (a) Free Energy Principle / Active Inference accounts of precision and gain control, (b) Global Neuronal Workspace accounts of conscious access via ignition and broadcast, and (c) boundary formalisms (Markov blankets) plus enactive/autopoietic constraints as a way to operationalize “self-maintenance” and “self/non-self separation.” citeturn9view0turn12view0turn11view0turn28view1turn19view0turn18view0

Two core conclusions follow from the primary literature:

First, **Friston-style precision is not merely “any uncertainty output.”** In predictive-coding/active-inference process theories, precision is the *inverse variance/covariance* of prediction errors and acts as a **multiplicative gain on error messages**; crucially, the brain (and by analogy an engineered agent) must also *infer / control expected precision* as a context-sensitive quantity that determines which errors dominate belief updating. This is why the same authors explicitly link precision to **synaptic gain and neuromodulation** (e.g., dopamine/acetylcholine), and why active inference distinguishes **sensory/likelihood precision**, **prior precision**, and **policy precision (inverse temperature over policies)**. citeturn9view0turn12view0turn15view0turn16view1turn15view3

Second, **GNW-style conscious access is operationally about global availability through ignition-like dynamics**, not about “phenomenality.” Canonical GNW formulations define conscious access as when information becomes globally available to multiple specialist systems (working memory, report, decision, planning), typically via **nonlinear ignition** supported by recurrent, long-range interactions. This implies that “conscious access” in an artificial system can be operationalized by (i) **a bottleneck/workspace**, (ii) **a broadcast mechanism that causally changes many modules**, and (iii) **a nonlinear state transition (ignition) separating local processing from globally available processing**. citeturn28view1turn10view2turn11view0turn5search1turn5search2

For your specific design questions:

- Treating **TRM‑A’s `pred_logvar` as precision** is *conditionally legitimate* if you are explicit about *what precision it is*: it is best interpreted as a **learned likelihood/transition noise parameter** (aleatoric uncertainty) whose inverse is a precision proxy. But it is **not automatically equal** to the *expected precision* variable that, in active inference, gates the influence of prediction errors, nor does it by itself represent epistemic uncertainty unless you add mechanisms (ensembles, Bayesian weights, posterior sampling) and calibration checks. citeturn24view2turn13search5turn9view0turn15view0

- A **precision-weighted global broadcast** is broadly consistent with one influential bridging line of work that formalizes GNW-like access within active inference (predictive global workspace / active-inference models of access): precision controls what is gated into temporally deep levels and thus what becomes actionable/reportable. But it is also a conceptual “jump” if broadcast is treated as a simple weighted routing rule rather than a *dynamical regime change* (ignition) created by recurrent competition and stabilization. citeturn7view5turn17view0turn28view1turn21view2

- For ERIE evaluation, the strongest practical stance is: **define minimal operational targets**, build metrics that can fail, and treat “consciousness-like” as a graded engineering claim about *availability, stability, and causal efficacy* rather than a metaphysical claim. This is also aligned with the state of the empirical field, where even large-scale adversarial tests of GNW vs IIT challenge key tenets of both and motivate more quantitative, theory-neutral operational criteria. citeturn20view0turn20view1

## Precision in FEP and Active Inference

### What “precision” is in FEP/Active Inference

In the Friston FEP / predictive processing / active inference tradition, **precision** is the *reliability parameter* of probabilistic beliefs, most canonically expressed as the **inverse variance (or inverse covariance) of fluctuations/noise** and thus the **inverse covariance of prediction errors** under Gaussian/Laplace assumptions. citeturn12view0turn9view0

A central point that matters for ERIE is that precision is not only a descriptive statistic; it is a **control quantity** because it determines how much a given prediction error should change beliefs. In predictive coding implementations, the literature explicitly states that precision “controls the relative influence of bottom-up prediction errors and top-down predictions,” and that in predictive coding “precision modulates the amplitude of prediction errors,” effectively giving them higher impact when precision is high. citeturn9view0turn7view2

Active inference then broadens “precision” into multiple, distinct roles:

- **Likelihood (sensory) precision**: reliability of sensory evidence (how noisy observations are). citeturn12view0turn9view0  
- **Prior precision**: strength of priors over hidden states and parameters. citeturn12view0turn9view0  
- **Policy precision** (often denoted as an inverse-temperature-like parameter γ): confidence in— or sharpness of—beliefs over policies, controlling how deterministically the agent selects policies given expected free energy/value. citeturn16view1turn15view0  

This separation matters because `pred_logvar` produced by TRM‑A naturally corresponds to only one slice (a likelihood/transition noise proxy), not the whole family.

### Mathematical forms and the “variance vs inverse variance vs gain” confusions

A useful engineering disambiguation (because your question explicitly asks for it) is:

**Prediction variance (σ²)**  
A dispersion parameter of a predictive distribution (e.g., \(p(y\mid x)\) is Normal with variance σ²). Large σ² means “the model expects the observation/state to be noisy.” This is a *noise model parameter* (aleatoric uncertainty) unless it is explicitly a posterior variance over beliefs. citeturn24view2turn13search5

**Precision (Π)**  
The *inverse* of variance in the scalar Gaussian case: Π = 1/σ² (and for multivariate Gaussians, Π is the inverse covariance matrix). Under Laplace assumptions, the predictive-coding literature explicitly calls precision the inverse covariance and uses it in update equations. citeturn12view0turn9view0

**Gain / confidence (computational role)**  
“Gain” is the functional role: a multiplicative factor that increases the effect of error signals. In Friston-style predictive coding, this gain is identified with precision weighting: prediction errors are multiplied by precision and thus have larger influence when precision is high. citeturn9view0turn7view2

**Synaptic gain (putative neural implementation)**  
The same literature proposes that precision is implemented by **synaptic gain** (postsynaptic responsiveness) of units encoding prediction errors, linking precision to attention and neuromodulatory control. citeturn7view2turn9view0

**Policy precision (γ) vs likelihood precision**  
In active inference, γ is not “inverse variance of sensory noise”; it plays the role of a **sensitivity / inverse temperature** parameter over policy beliefs—how sharply the posterior over policies concentrates. It has a confidence-like interpretation and has been linked in active-inference accounts to dopaminergic signaling, because precision enters multiplicatively into policy belief updates and modulates action confidence. citeturn16view1turn15view0turn15view3

### What precision corresponds to in neuroscience/cognitive science

Within this framework, precision is repeatedly mapped to attention-like and neuromodulatory mechanisms:

- Precision is proposed to be encoded by **synaptic gain** on prediction-error units, and candidate biological mechanisms include classical neuromodulators (dopamine, acetylcholine) and synchronous presynaptic activity that changes effective gain. citeturn9view0turn7view2  
- Policy precision has been explicitly used in active inference accounts to reinterpret dopamine as encoding changes in expected precision over policies (confidence in action selection). citeturn15view0turn15view3  
- A related predictive-processing proposal assigns a key role to precision control within thalamocortical loops (e.g., the pulvinar as a precision regulator), reinforcing the idea that precision is a top-down *routing/gain* control variable rather than a passive output. citeturn1search2  

For ERIE, the essential translation is: **precision is best treated as a system-level control signal that gates which errors and which representations matter**, not just as a scalar uncertainty estimate attached to a model output. citeturn9view0turn15view3

### How far `pred_logvar → precision` is theoretically justified

In machine learning practice, predicting a mean and a log variance and then optimizing a heteroscedastic Gaussian NLL yields a loss term of the form \(\|y-\hat{y}\|^2 \exp(-s) + s\), where \(s=\log \hat{\sigma}^2\). This makes the effective weight on squared error equal to \(\exp(-s) = 1/\hat{\sigma}^2\), i.e., **a precision-like factor**. This is explicitly motivated as numerically stable and interpretable as learned attenuation/weighting. citeturn24view2

So, **for TRM‑A**, the mapping

\[
\texttt{precision\_proxy} = \exp(-\texttt{pred\_logvar})
\]

is mathematically coherent *as long as*:

- `pred_logvar` truly parameterizes a Gaussian variance (or diagonal covariance) of the predictive distribution of the target variable (next state, observation). citeturn24view2turn12view0  
- You do **calibration checking**, because deep models can be systematically miscalibrated in their confidence/uncertainty; calibration metrics and NLL-based evaluation are standard for this purpose. citeturn24view1turn13search4  

However, it would be **Friston-inaccurate** to claim that `pred_logvar` alone “is precision” in the active-inference sense, for three reasons:

- In predictive coding under FEP, precision is a property of **prediction errors** (noise in the observation model / process model) and is typically treated as context-sensitive and inferable/optimizable, not merely emitted once by a forward predictor. citeturn9view0turn12view0  
- Active inference distinguishes **aleatoric noise (likelihood variance)** from **epistemic uncertainty** about states/parameters; a single deterministic model output variance usually tracks the former unless you add explicit epistemic machinery (Bayesian weights, ensembles, posterior sampling). citeturn13search5turn24view2turn15view0  
- Precision also appears at the level of **policy selection** (γ), which is structurally different from a world-model output variance and is central for confidence/action vigor in several active-inference accounts. citeturn16view1turn15view0  

Practically: `pred_logvar` can be a legitimate **precision *proxy*** for likelihood/transition noise, but ERIE should still implement **separate precision-control variables** that (a) gate inference updates and (b) gate global broadcast / working-memory gating.

## Conscious access and consciousness-like properties

### GNW definitions of conscious access, ignition, broadcast, and global availability

In GNW (and its cognitive-architecture ancestor, global workspace theory), “conscious access” is treated as **global availability**: information is conscious when it is widely broadcast and thus accessible to many cognitive processors (working memory, report, evaluation, planning). citeturn28view1turn10view2turn11view0

Core mechanistic commitments that repeatedly appear in primary and major review texts include:

- **Ignition**: a nonlinear, often “all-or-none”-like activation regime where a subset of workspace neurons becomes suddenly and coherently active, with others suppressed/inhibited. citeturn28view1turn21view2turn11view0  
- **Broadcast**: once ignited, the workspace representation is disseminated back to specialized processors, enabling flexible, coordinated processing and report. citeturn28view1turn10view2turn5search1  
- **Recurrent processing**: GNW explicitly relies on recurrent interactions (long-range and local) to sustain and amplify the ignited state. citeturn28view1turn21view1  
- **Attention/working memory coupling**: GNW-related reviews emphasize that conscious processing is intertwined with attention and working memory, and the boundary between “access” and “post-perceptual processing for report” is an ongoing empirical issue. citeturn28view1turn28view0turn17view0  

A key operational passage for engineering: Dehaene–Naccache-style workspace accounts explicitly characterize access as a **dynamic mobilization** in which top-down attentional amplification makes modular processes available to the global workspace “and therefore to consciousness,” and requires amplification/maintenance long enough to become accessible to multiple processes. citeturn11view0

### Recurrent Processing Theory and IIT as contrasts

Because you asked for limits and cross-criticism, two contrasts matter for operationalization:

**Recurrent Processing Theory** (Lamme) emphasizes that **recurrent processing within sensory cortex** is the key for conscious vision, and frequently argues that many high-level cognitive/report correlates are downstream or optional. The Open-MIND target paper explicitly frames the transition from “unconscious processing” to “conscious vision” in terms of perceptual functions and recurrent mechanisms, and argues that large parts of processing are unconscious while consciousness becomes necessary when more elaborate incremental interpretation is required. citeturn27view3turn27view0

**Integrated Information Theory** (Tononi) identifies consciousness with properties of a system’s intrinsic cause–effect power and proposes a mathematical framework aimed at quantifying consciousness (Φ etc.). It explicitly starts from axioms about experience and infers properties of the physical substrate; this is structurally different from GNW’s access/broadcast functional role. citeturn25view0turn3search7

For ERIE, the engineering implication is not “pick a metaphysics,” but: **GNW provides the most direct blueprint for a functional/architectural operationalization of conscious access** (workspace, broadcast, ignition). RPT and IIT are valuable as *stress tests*—they imply that some GNW signatures may track report/working memory rather than awareness per se, and that “global availability” might not be sufficient for what some theories call consciousness. citeturn28view0turn27view0turn25view0turn20view0

### Conscious access vs attention vs working memory

The literature distinguishes these in ways that are directly usable in ERIE design:

- **Attention**: selective amplification/routing; it can occur without conscious awareness and vice versa, motivating a conceptual separation between top-down attention and consciousness. citeturn21view0  
- **Working memory**: maintenance/manipulation of representations over time; GNW often treats workspace contents as globally broadcast and therefore naturally linked to working memory functions, but some signatures (e.g., P3b) track report/WM demands rather than awareness itself. citeturn28view1turn28view0turn17view0  
- **Conscious access**: the condition where information is globally available to multiple systems such that it can guide flexible behavior, report, and cross-domain integration; GNW proposes ignition and long-range recurrent interaction as mechanisms. citeturn28view1turn21view1turn21view2  

A particularly relevant computational bridge for ERIE is the active-inference modeling line that treats conscious access as depending on **precision and temporally deep policy selection**, modeling working-memory gating as a cognitive action and showing how report/no-report manipulations can change late signatures without necessarily eliminating “accessibility.” citeturn17view0turn7view5

### Minimal and stronger operational definitions of “consciousness-like” in an artificial system

Because you explicitly asked for weakest operational definitions and stronger ones, a pragmatic ladder looks like this:

**Weak (minimal, testable) operational definition: “GNW-style access”**  
A system exhibits consciousness-like *access* iff there exists a representational state that becomes **globally available** (causally usable) across multiple specialized modules through a bottleneck/workspace, and that state transition shows **ignition-like nonlinearity** (bifurcation / sudden stabilization) separating local processing from global availability. This is the direct translation of GNW’s functional role. citeturn28view1turn21view2turn11view0turn5search1

**Stronger (still functional/computational): “access + metacognitive control”**  
Add the requirement that the system can form and use **explicit confidence/uncertainty representations** (precision-like control) to regulate (a) what enters the workspace and (b) how long it is maintained for policy selection, in a way that is behaviorally and causally diagnostic. This aligns with active-inference accounts that tie access/report to temporally deep policy selection and precision control. citeturn17view0turn16view1turn15view0

**Stronger (architectural + self-model): “access + self/non-self boundary + self-maintenance constraints”**  
Add a maintained internal boundary model (Markov-blanket-like interfaces) plus explicit viability constraints that the system actively preserves (homeostatic variables, resource budgets), so that access states are embedded in a self-maintaining control loop. This aligns with FEP/autopoiesis-inspired readings but must be treated carefully because Markov blanket “boundaries” are statistical tools and have been criticized when used as metaphysical demarcations without extra premises. citeturn19view0turn2search0turn18view0turn2search9

The important caveat is: even the stronger functional ladders remain *functional*; they do not settle “subjectivity itself.” That limitation should be stated explicitly in ERIE documentation to avoid category errors while still providing rigorous, falsifiable engineering claims. citeturn28view1turn25view0turn20view0

### Definition comparison table across theories

The table below is a **design-oriented equivalence map**. It compresses claims from the major primary/review sources cited in the surrounding text (FEP/predictive coding, GNW, RPT, IIT, and enactive/autopoietic lines). citeturn9view0turn12view0turn28view1turn11view0turn27view3turn25view0turn19view0turn18view0

| Term | FEP / Active Inference | GNW / Global Workspace | Recurrent Processing Theory | IIT | Enactivism / Autopoiesis (engineering reading) |
|---|---|---|---|---|---|
| **precision** | Inverse variance/covariance; gain on prediction errors; includes expected precision and policy precision | Not a core primitive; enters indirectly via attentional amplification/gating | Not a core primitive; recurrence quality may co-vary with “signal strength/selection” | Not a core primitive; consciousness tied to intrinsic cause–effect structure | Not a core primitive; emphasis on organism–environment coupling and autonomy constraints |
| **conscious access** | Often framed as what becomes available to temporally deep policy selection / higher-level inference under sufficient precision | “Information becomes conscious when globally available/broadcast to many processors,” supported by ignition | Often not required; access/report can be downstream of perceptual awareness | Not central (phenomenology-first); access/report dissociable | Not central; focus on sense-making and autonomy (may treat access as one functional layer) |
| **consciousness-like property** (functional) | Candidates include inferential depth, precision control, and coherent global belief sharing (varies by proposal) | Functional consciousness ≈ global broadcast enabling flexible control/report | Phenomenal/perceptual consciousness ≈ recurrent sensory processing | Consciousness ≈ integrated information / intrinsic cause–effect power | Cognition/consciousness tied to autonomy and self-producing organization (strong versions exceed typical engineering claims) |
| **global availability** | Implementable as global belief-sharing / message passing; not guaranteed by probabilistic inference alone | Definitional core: wide accessibility across specialized processes | Not required; local recurrent loops may suffice for awareness claims | Not definitional; global availability may not track Φ | “Availability” arises through coupling and functional organization, not a dedicated workspace per se |
| **ignition** | Could be formalized as bifurcation/phase transition in inference dynamics under precision control | Definitional signature: sudden, coherent, exclusive activation of workspace subset | Recurrent loops can be nonlinear, but ignition is not necessarily frontoparietal/global | Not definitional; IIT predicts different signatures (posterior “hot zone” claims etc.) | Not a standard term; phase transitions may be discussed in self-organization contexts |

## Mapping to ERIE architecture

This section translates the above definitions into implementable commitments for TRM‑A (world model), TRM‑B (boundary), and GNW (integration/global broadcast). The mapping is oriented to “what must be true in ERIE for the term to be used honestly,” i.e., necessary conditions and falsifiers. citeturn28view1turn9view0turn19view0turn18view0turn17view0

### Precision in ERIE with TRM‑A: how to represent and test it

**Essence (computational)**  
Precision is a *weight* that controls the influence of prediction errors on belief updating (and on policy selection), typically modeled as inverse variance/covariance under Gaussian/Laplace assumptions. citeturn12view0turn9view0turn16view1

**Necessary condition (ERIE claim-compatibility)**  
ERIE must have an explicit inference/update path where prediction errors are **precision-weighted**, not merely a forward predictor that emits uncertainties. In other words, precision must be *used as gain*. citeturn9view0turn7view2

**Implementation example (TRM‑A + inference loop)**  
- TRM‑A outputs `(pred_mean, pred_logvar)` and ERIE computes `Π̂ = exp(-pred_logvar)` as a *likelihood/transition precision proxy* (dimensionwise if diagonal). This is mathematically aligned with heteroscedastic Gaussian NLL training conventions. citeturn24view2  
- ERIE then uses Π̂ to scale the corresponding prediction error term in state estimation (e.g., in a predictive-coding-style update rule or as weights in a variational objective). citeturn9view0turn12view0  

**Why this is only partially “Friston-precision”**  
- Π̂ derived from `pred_logvar` is most naturally **aleatoric** (noise you expect in the channel), whereas Friston-style “expected precision” is also a *contextual gating* variable and may itself be inferred/optimized (attention-like control). citeturn9view0turn15view3  
- If ERIE also needs epistemic confidence (model uncertainty), augment with ensembles/posterior sampling and treat the resulting dispersion separately from Π̂ (do not collapse both into one “precision”). citeturn13search5turn15view0turn24view2  

**Proxy measures (what you can measure in practice)**  
- **Calibration + proper scoring**: negative log-likelihood / log score and calibration error measures to test whether Π̂ is meaningful rather than decorative. citeturn24view1turn13search4  
- **Causal gain test**: ablate or clamp precision weighting and quantify whether belief updates become systematically over/under-reactive in the predicted ways (precision should change sensitivity). citeturn9view0turn15view3  

**Falsification conditions**  
- If Π̂ correlates weakly with realized residuals and calibration remains poor even after standard calibration/training fixes, then `pred_logvar` should not be treated as a trustworthy precision proxy for control. citeturn24view1turn13search4  
- If precision weighting does not causally modulate update magnitude (removing it does not change inference dynamics materially), then “precision” is not implemented in the active-inference sense even if you log a number called precision. citeturn7view2turn9view0  

### Self/non-self boundary in ERIE with TRM‑B: Markov blanket as an engineering boundary

**Essence (computational/statistical)**  
A Markov blanket is a conditional-independence boundary in a graphical model; in active inference usage it partitions states into internal, external, and blanket states (often sensory and active), so that internal and external are independent given the blanket. citeturn2search0turn19view0

**Critical caution (because you asked for limits)**  
There is a well-developed critique that warns against conflating (a) Markov blankets as *epistemic tools in Bayesian networks/variational inference* with (b) blanket talk as a metaphysical claim about the *physical boundary* of agents. For ERIE, the safe position is to treat TRM‑B as implementing an engineered conditional-independence interface—an architectural boundary—without claiming that this alone grounds “selfhood” in a metaphysical sense. citeturn18view0turn19view0

**Necessary condition (ERIE claim-compatibility)**  
TRM‑B must enforce that all information flow between “inside” and “outside” passes through explicit interface variables (sensory in / action out), so that internal modules cannot directly depend on external states except through the interface (and similarly outward influence occurs through actions). citeturn2search0turn19view0

**Proxy measures (what you can quantify)**  
- **Graph/interface audit**: verify that all cross-boundary dependencies are mediated by designated variables (software-level Markov blanket).  
- **Conditional independence residuals**: estimate whether internal states become statistically conditionally independent of external states given interface variables (use mutual information / conditional MI estimates as diagnostics). citeturn19view0turn18view0  
- **Viability maintenance**: if ERIE is “self-maintaining,” define explicit viability variables (energy budget, memory integrity, uptime constraints) and test whether boundary policies preserve them under perturbation—mirroring FEP/autopoiesis-style claims about maintaining integrity. citeturn19view0turn2search9  

**Falsification conditions**  
- If internal modules achieve performance by hidden side-channels that bypass the boundary (e.g., direct shared state with the environment module), then the boundary is not operationally realized even if conceptually assumed.

### GNW layer in ERIE: operationalizing global availability, broadcast, and ignition

**Essence (GNW functional role)**  
Conscious access is identified with information becoming broadly accessible to multiple processors through a workspace/broadcast architecture, frequently accompanied by ignition (nonlinear amplification, coherent stabilization, and competitive exclusion). citeturn28view1turn11view0turn21view2

**Necessary condition (ERIE claim-compatibility)**  
ERIE must include:

1) A **workspace/bottleneck** state (capacity-limited, serializable at least at the level of “what is broadcast now”). citeturn28view1turn4search9  
2) A **broadcast mechanism** that makes workspace content causally available to many modules. citeturn28view1turn10view3  
3) A **competitive selection + recurrent stabilization loop** so that “broadcast” is not just a bus, but can enter a stable regime (ignition-like attractor). citeturn21view2turn11view0  

**Operational ignition in an artificial system**  
Ignition does not have to be “PFC activity”; it can be defined as a *dynamical signature*:

- A sharp, nonlinear transition in global workspace activation as input strength/attention/priors are varied (bifurcation-like). citeturn21view2turn21view1  
- Sudden increase in cross-module coupling/coherence once a representation wins access, with suppression of alternatives (competitive exclusion). citeturn28view1turn11view0  

**Why report/no-report confounds matter for ERIE metrics**  
A major empirical concern is that many late “GNW signatures” can track post-perceptual/report processing. For example, work using no-report manipulations shows that P3b can disappear when reporting is removed, suggesting it is not a pure marker of awareness and highlighting the need to clearly separate “accessibility” from “report generation.” citeturn28view0turn17view0

For ERIE, this implies that your “global availability” metric should be **causal and capability-based** (what modules can do with the state) rather than **report-based** (does the system emit a verbal token?). citeturn28view0turn21view0

### Precision-weighted broadcast: where it aligns and where it leaps

**Where it aligns**

- GNW explicitly uses attention-like amplification as a gating mechanism for workspace entry, and active inference identifies attention-like effects with precision control (gain on errors). This creates a plausible bridge: **precision can be the control signal that gates what enters and dominates the workspace**. citeturn11view0turn9view0turn7view2  
- Recent formal work explicitly builds active-inference models that capture GNW architectural elements (a predictive global workspace), and active-inference models of conscious access use precision manipulations to model working-memory gating and report/no-report patterns. citeturn7view5turn17view0  

**Where it leaps**

- If “broadcast” is implemented as a continuous weighted sum routed everywhere, you might *lose the GNW ignition claim*, which is about **nonlinear, exclusive, metastable workspace states** rather than smooth routing. citeturn21view2turn28view1  
- If you equate “high precision” with “being conscious,” you risk collapsing attention/precision into consciousness, a move that is explicitly contested in the empirical/theoretical literature (attention and consciousness can dissociate). citeturn21view0turn7view5  
- The cognitive-neuroscience field is currently in a state where even adversarial large-scale testing yields partial support and partial contradictions for both GNW and IIT, suggesting that any single-signature mapping (“precision-weighted broadcast = GNW”) should be treated as a model hypothesis with clear failure modes rather than a settled equivalence. citeturn20view0turn20view1  

## Candidate definitions to adopt

This section answers your “adopt vs hold” request in the strict operational/computational sense.

### Adopt: definitions that are both faithful to sources and implementable

**Precision (ERIE-adopted definition)**  
Precision is a control-relevant inverse variance/covariance parameter that (i) weights prediction errors during belief updating and (ii) (optionally) controls policy selection confidence (policy precision γ). This is directly aligned with predictive-coding and active-inference formulations linking precision to gain and to inverse-temperature-like confidence over policies. citeturn9view0turn12view0turn16view1turn15view0

**Conscious access (ERIE-adopted definition)**  
Conscious access is the state in which some representation becomes globally available—causally usable—by multiple specialized modules via a workspace/broadcast architecture, typically via a nonlinear ignition-like transition and recurrent stabilization. This is a direct architectural import of GNW/global workspace functional commitments. citeturn28view1turn11view0turn21view2turn10view3

**Consciousness-like property (minimal operational claim)**  
“Consciousness-like” is claimed only as “GNW-like access” unless and until stronger criteria are met. Concretely: ERIE has consciousness-like access iff it shows measurable global availability + ignition dynamics + cross-module causal integration, without implying phenomenal consciousness. This matches how GNW-oriented reviews often operationalize their target as access rather than phenomenality. citeturn28view1turn21view2turn20view0

### Adopt as engineering proxies (explicitly not sufficient conditions)

**`pred_logvar`-derived precision proxy**  
Use \( \exp(-\texttt{pred\_logvar}) \) as a *likelihood/transition precision proxy* if—and only if—calibration and causal gain tests pass. This is consistent with heteroscedastic regression practice and can be integrated into a precision-weighted inference/broadcast pipeline. citeturn24view2turn24view1

**Ignition proxy**  
Operationalize ignition as a regime change in which workspace activation becomes (a) sudden/nonlinear with respect to control parameters and (b) metastable/maintained and (c) exclusive (alternatives suppressed). This is consistent with “all-or-none bifurcation” style language used in attentional-blink studies and with GNW ignition descriptions. citeturn21view2turn28view1turn11view0

**Global availability proxy**  
Define “global availability” as the number and diversity of modules whose behavior is causally altered by workspace content, not as the presence of a particular late ERP-like signature. This aligns with no-report concerns and with GNW’s emphasis on accessibility to multiple processors. citeturn28view0turn28view1turn21view0

## Open questions and items to defer

### Defer: claims that exceed current operational support

**Equating `pred_logvar` with “Friston precision” without additional structure**  
Without an explicit inferential loop where precision is optimized/controlled and without separating aleatoric vs epistemic uncertainty, equating `pred_logvar` to Friston precision will be misleading. The active-inference literature treats precision as a modulatory/gain quantity; the ML log-variance output is only a partial match. citeturn9view0turn15view3turn24view2turn13search5

**Strong metaphysical readings of Markov blankets as “the” boundary of self**  
The critique literature argues that blanket formalism is often overextended from an inference tool to metaphysical boundary claims without extra premises. For ERIE, claim only the engineered boundary properties you can test. citeturn18view0turn19view0turn2search0

**“Sufficient conditions” for consciousness-like property**  
Even in neuroscience, competing theories disagree about what is sufficient, and recent adversarial testing challenges key tenets of both GNW and IIT. In ERIE you can define strong operational criteria for *access*, but you should avoid claiming sufficiency for subjective consciousness. citeturn20view0turn25view0turn27view0

### Unresolved research-grade issues that directly affect ERIE design

**Precision–conscious access relationship**  
There is a plausible mechanistic bridge (precision as attentional gating for ignition/workspace entry), and formal active-inference models explicitly use precision control to model access and working-memory gating. But the attention–consciousness dissociation literature implies the mapping is nontrivial: precision may be necessary for some ignition regimes without being identical to consciousness-like access. citeturn17view0turn7view5turn21view0turn11view0

**Where to locate ignition in a modular artificial system**  
GNW and RPT disagree about whether global frontoparietal ignition is necessary; no-report paradigms and “front vs back” debates show the field is unsettled. For ERIE, define ignition functionally (system-level regime change) rather than anatomically (which module). citeturn28view0turn27view0turn20view0

**Quantitative metrics for “global availability” that are not trivially satisfied**  
A naïve “broadcast bus exists” criterion is too weak (many software systems have buses). Stronger metrics must be causal, capacity-limited, and tied to behavioral flexibility, otherwise the definition collapses into generic information routing. This concern is implicit in GNW’s emphasis on limited capacity, competitive selection, and flexible cross-domain access. citeturn28view1turn4search9turn10view3

## Key literature

The following sources (primary and major syntheses) are the most load-bearing for the definitions and operationalizations above:

- **Precision as inverse variance/covariance; precision-weighted prediction errors under Laplace/predictive coding**. citeturn12view0turn9view0  
- **Precision as synaptic gain; link between precision, attention, neuromodulation**. citeturn9view0turn7view2turn15view3  
- **Policy precision (γ) as inverse temperature/confidence over policies; dopamine as expected precision**. citeturn16view1turn15view0  
- **Global workspace / GNW foundations: global availability, broadcast, ignition and their functional role**. citeturn5search1turn5search2turn11view0turn28view1  
- **All-or-none / bifurcation-like ignition evidence in attentional blink and timing of access**. citeturn21view2turn21view1  
- **Attention vs consciousness dissociation (why “precision = consciousness” is too strong)**. citeturn21view0  
- **No-report paradigms and the risk that late signatures track report/WM demands (P3b critique)**. citeturn28view0turn17view0  
- **Formal bridge: active-inference implementations of GNW-like access (predictive global workspace; WM gating as cognitive action; precision-control in access models)**. citeturn7view5turn17view0  
- **Recurrent Processing Theory (contrast class; local recurrence emphasis)**. citeturn27view3turn27view0  
- **IIT statement and its contrasting primitives (cause–effect power; Φ framework)**. citeturn25view0turn3search7  
- **Markov blanket boundary, active inference, and autopoiesis-like readings (plus key critique of overextension)**. citeturn19view0turn2search0turn18view0turn2search9  
- **Large-scale theory-contrast status: adversarial GNW vs IIT testing and its “challenges to both” conclusion**. citeturn20view0turn20view1  
- **Uncertainty-output-as-log-variance practice and its precision-like weighting (engineering link for `pred_logvar`)**. citeturn24view2turn13search5  
- **Calibration/proper scoring foundations for evaluating probabilistic confidence/uncertainty outputs**. citeturn24view1turn13search4  

---

# 日本語訳

## ERIE のための Precision と Conscious Access の操作的定義

## エグゼクティブサマリー

本レポートは、あなたが対象としている二つの性質、すなわち  
1. **precision（信念の確信度 / 信頼性重みづけ）**  
2. **conscious access / global availability の意味での consciousness-like property**  
を、計算論的に定義でき、アーキテクチャ上に表現でき、操作的基準で検証できる**工学的対象**として扱う。ここでの統合は、  
- Free Energy Principle / Active Inference における precision と gain control  
- Global Neuronal Workspace における ignition と broadcast による conscious access  
- Markov blanket と enactive / autopoietic 制約による self-maintenance と self/non-self separation の操作化  
に基づいている。 citeturn9view0turn12view0turn11view0turn28view1turn19view0turn18view0

主要文献から、次の二つの結論が導かれる。

第一に、**Friston 的 precision は、単なる uncertainty 出力ではない。** predictive coding / active inference の過程理論では、precision は prediction error の**分散 / 共分散の逆数**であり、error message に対する**乗法的 gain**として働く。さらに脳は、どの誤差が belief updating を支配するかを決める文脈依存量として、**expected precision を推定 / 制御**しなければならない。このため同じ系列の文献は、precision を **synaptic gain や neuromodulation** と明示的に結びつけ、さらに active inference は **sensory / likelihood precision**, **prior precision**, **policy precision** を区別している。 citeturn9view0turn12view0turn15view0turn16view1turn15view3

第二に、**GNW 的な conscious access は phenomenality ではなく、ignition 的ダイナミクスを通じた global availability である。** 典型的な GNW 定式化では、情報が複数の専門システム（working memory, report, decision, planning）へグローバルに利用可能になったとき conscious access が成立し、その機構として recurrent かつ長距離結合に支えられた**非線形 ignition**が想定される。したがって人工システムにおける conscious access は、  
- **bottleneck / workspace**  
- 多数のモジュールを因果的に変える **broadcast mechanism**  
- 局所処理と global availability を分ける **非線形な状態遷移（ignition）**  
として操作的に定義できる。 citeturn28view1turn10view2turn11view0turn5search1turn5search2

ERIE の具体的設計に関しては次の通りである。

- **TRM-A の `pred_logvar` を precision として扱うこと**は、何の precision かを明示する限りで条件付きに正当化できる。これは最も自然には、学習された **likelihood / transition noise parameter**、すなわち aleatoric uncertainty と解釈され、その逆数が precision proxy になる。ただしこれは、active inference で prediction error の影響を制御する **expected precision** と自動的に同一ではなく、また ensemble, Bayesian weights, posterior sampling, calibration check がない限り epistemic uncertainty も表さない。 citeturn24view2turn13search5turn9view0turn15view0

- **precision-weighted global broadcast** は、active inference の内部で GNW 的 access を形式化しようとする有力な橋渡し研究とは概ね整合する。つまり precision が temporally deep なレベルへ何が通るかを制御し、その結果として何が actionable / reportable になるかを決める、という考え方である。ただし、broadcast を単なる重み付きルーティング規則として実装してしまうと、recurrent competition と stabilization による**dynamical regime change としての ignition**からは飛躍が生じる。 citeturn7view5turn17view0turn28view1turn21view2

- ERIE の評価においては、**最小限の operational target を定義し、失敗しうる metric を用意し、consciousness-like を metaphysical claim ではなく、availability・stability・causal efficacy に関する段階的な工学的主張として扱う**のが最も実践的である。これは、GNW と IIT をめぐる近年の adversarial test が両理論の主要命題に部分的反証を与えており、より定量的で理論中立的な基準を求めている現状とも整合する。 citeturn20view0turn20view1

## FEP と Active Inference における Precision

### FEP / Active Inference における precision とは何か

Friston の FEP / predictive processing / active inference の系譜では、**precision** は確率的信念の**信頼度パラメータ**であり、最も標準的には fluctuation / noise の**分散（または共分散）の逆数**、したがって Gaussian / Laplace 仮定のもとでの **prediction error の逆共分散**として表現される。 citeturn12view0turn9view0

ERIE にとって重要なのは、precision が単なる記述統計ではなく、**control quantity** だという点である。precision は、ある prediction error がどの程度 belief を更新すべきかを決める。predictive coding の実装文献では、precision は「bottom-up の prediction error と top-down の prediction の相対的影響を制御する」とされ、prediction error の振幅を変調することで、高 precision の error に大きな影響力を与える。 citeturn9view0turn7view2

active inference はさらに precision を複数の役割へ拡張する。

- **Likelihood (sensory) precision**: sensory evidence の信頼性、すなわち観測のノイズの小ささ。 citeturn12view0turn9view0
- **Prior precision**: hidden state や parameter に対する prior の強さ。 citeturn12view0turn9view0
- **Policy precision**: 多くは inverse temperature 的パラメータ γ として表され、policy belief の鋭さ、すなわち expected free energy / value に基づいてどれだけ決定的に policy を選ぶかを制御する。 citeturn16view1turn15view0

この区別が重要なのは、TRM-A の `pred_logvar` が自然に対応するのはこの全体のうち一部、すなわち likelihood / transition noise proxy にすぎないからである。

### 分散・逆分散・gain の混同をどう整理するか

工学的には次のように整理するとよい。

**Prediction variance (σ²)**  
予測分布の散らばりを表すパラメータ。たとえば \(p(y \mid x)\) が分散 σ² の正規分布である場合、σ² が大きいほど「この観測 / 状態はノイズが大きい」とモデルが見なしている。これは明示的に posterior variance でない限り、通常は **aleatoric uncertainty** を表す。 citeturn24view2turn13search5

**Precision (Π)**  
スカラー Gaussian では分散の逆数、Π = 1/σ²。多変量 Gaussian では inverse covariance matrix である。Laplace 仮定の文献では、precision は prediction error の inverse covariance としてそのまま update equation に入る。 citeturn12view0turn9view0

**Gain / confidence（計算論的役割）**  
gain は機能的役割を指す。error signal の効果を増幅する乗法係数である。Friston 型 predictive coding では、この gain が precision weighting そのものであり、高 precision の error ほど belief updating を大きく動かす。 citeturn9view0turn7view2

**Synaptic gain（神経実装仮説）**  
同じ文献群は、precision が prediction-error unit の **synaptic gain** によって実装されると提案し、precision を attention や neuromodulatory control と結びつける。 citeturn7view2turn9view0

**Policy precision (γ) と likelihood precision の違い**  
active inference における γ は sensory noise の inverse variance ではない。これは policy belief にかかる **sensitivity / inverse temperature** であり、posterior over policies がどれだけ鋭く集中するかを表す。confidence に近い意味を持ち、policy belief 更新に precision が乗法的に入ることから、dopaminergic signaling とも結びつけて説明される。 citeturn16view1turn15view0turn15view3

### 神経科学 / 認知科学で precision は何に対応づけられているか

この枠組みでは、precision は attention や neuromodulation に繰り返し対応づけられる。

- precision は prediction-error unit の **synaptic gain** として符号化されうると提案され、候補機構として dopamine, acetylcholine や同期的 presynaptic activity が挙げられる。 citeturn9view0turn7view2
- policy precision は、policy に対する expected precision の変化を dopamine が符号化するという active inference 的解釈に使われる。 citeturn15view0turn15view3
- 別の predictive-processing 系の提案では、pulvinar を含む thalamocortical loop が precision regulator の役割を果たすとされ、precision が単なる出力ではなく top-down の **routing / gain control variable** であることを強めている。 citeturn1search2

ERIE への翻訳として重要なのは、**precision はシステム全体で、どの誤差とどの表象を重要視するかを決める制御信号として扱うべきであり、単なる uncertainty scalar としては不十分**だという点である。 citeturn9view0turn15view3

### `pred_logvar -> precision` はどこまで理論的に正当化できるか

機械学習では、mean と log variance を予測し、heteroscedastic Gaussian NLL を最適化すると、損失は \(\|y-\hat{y}\|^2 \exp(-s) + s\) という形になる。ここで \(s=\log \hat{\sigma}^2\) であり、二乗誤差にかかる実効重みは \(\exp(-s)=1/\hat{\sigma}^2\)、すなわち **precision 的な係数**になる。これは数値安定性が高く、学習された attenuation / weighting として解釈可能である。 citeturn24view2

したがって TRM-A に関しては、

\[
\texttt{precision\_proxy} = \exp(-\texttt{pred\_logvar})
\]

という写像は、次の条件下では数学的に首尾一貫している。

- `pred_logvar` が、target variable の予測分布の Gaussian variance（または diagonal covariance）を本当にパラメタライズしていること。 citeturn24view2turn12view0
- calibration check を行うこと。深層モデルの confidence / uncertainty は容易に miscalibrated になりうるため、calibration metric や NLL による評価が必要である。 citeturn24view1turn13search4

ただし、`pred_logvar` 単独を active inference の意味での precision と呼ぶのは **Friston 的には不正確**である。理由は三つある。

- FEP 下の predictive coding では、precision は **prediction error** の性質であり、通常は文脈依存で、推論対象でもあり、forward predictor が一回出力するだけの値ではない。 citeturn9view0turn12view0
- active inference は **aleatoric noise** と **epistemic uncertainty** を区別する。単一の deterministic model が出す variance は、通常は前者しか追跡しない。後者には Bayesian weights, ensembles, posterior sampling が必要である。 citeturn13search5turn24view2turn15view0
- precision は **policy selection** のレベルにも現れるが、これは world-model output variance とは構造的に別物であり、confidence や action vigor と深く関わる。 citeturn16view1turn15view0

実務上の結論は、`pred_logvar` は likelihood / transition noise に対する**precision proxy**としては妥当だが、ERIE にはそれとは別に、
- inference update をゲートする precision-control variable
- global broadcast / working-memory gating をゲートする precision-control variable
が必要だということである。

## Conscious Access と Consciousness-like Property

### GNW における conscious access, ignition, broadcast, global availability

GNW およびその祖先である global workspace theory では、**conscious access は global availability** として扱われる。情報が広く broadcast され、多数の cognitive processor（working memory, report, evaluation, planning）から利用可能になるとき、その情報は conscious であるとされる。 citeturn28view1turn10view2turn11view0

主要な機構的コミットメントは次である。

- **Ignition**: workspace neuron の部分集合が突然かつ協調的に活性化し、他の候補が抑制される、非線形で all-or-none に近い activation regime。 citeturn28view1turn21view2turn11view0
- **Broadcast**: ignition された表象が専門プロセッサへ再配信され、柔軟で協調的な処理や report を可能にする。 citeturn28view1turn10view2turn5search1
- **Recurrent processing**: GNW は ignition 状態を維持・増幅するために local / long-range の recurrent interaction を必要とする。 citeturn28view1turn21view1
- **Attention / working memory coupling**: GNW 系のレビューは、conscious processing が attention と working memory に絡むこと、また access と post-perceptual report process の境界が経験的には未解決であることを認めている。 citeturn28view1turn28view0turn17view0

工学的に重要なのは、Dehaene–Naccache 的な workspace 説が、top-down attentional amplification により modular process が global workspace に利用可能になり、それゆえ consciousness に入ると述べている点である。つまり access には、多数の process から利用されるだけの**増幅と維持**が必要である。 citeturn11view0

### 対照理論としての RPT と IIT

限界と相互批判を考えるうえで、二つの対照が重要である。

**Recurrent Processing Theory（RPT）** は、**sensory cortex 内部の recurrent processing** こそ conscious vision の鍵だと強調し、高次の認知 / report 相関の多くは下流または任意だと論じる。したがって、GNW のような global ignition がなくても awareness を主張しうる。 citeturn27view3turn27view0

**Integrated Information Theory（IIT）** は、意識をシステムの intrinsic cause-effect power の性質と見なし、Φ などの数学的量によって定量化しようとする。これは GNW の access / broadcast の機能的役割とは構造的に異なる。 citeturn25view0turn3search7

ERIE にとっての工学的含意は、「どの metaphysics を選ぶか」ではない。むしろ、**GNW が conscious access の機能的 / アーキテクチャ的操作化に最も直接的な設計図を与える**という点である。一方で RPT と IIT は stress test として価値があり、GNW の指標の一部が report / working memory を追っているだけかもしれないこと、そして global availability だけでは consciousness と呼ぶには不十分かもしれないことを示す。 citeturn28view0turn27view0turn25view0turn20view0

### Conscious Access と Attention と Working Memory の違い

文献は、ERIE 設計に直接使える形でこれらを区別している。

- **Attention**: selective amplification / routing であり、conscious awareness なしに生じうるし、その逆もありうる。つまり top-down attention と consciousness は概念的に分ける必要がある。 citeturn21view0
- **Working memory**: 表象を時間にわたり維持 / 操作する機構。GNW は workspace content を globally broadcast されたものと見なしやすいが、P3b など一部の指標は awareness ではなく report / WM demand を反映している可能性がある。 citeturn28view1turn28view0turn17view0
- **Conscious access**: 情報が複数のシステムに global に利用可能になり、柔軟行動・report・cross-domain integration を導ける状態。GNW はその機構として ignition と long-range recurrent interaction を提案する。 citeturn28view1turn21view1turn21view2

ERIE にとって特に重要なのは、active inference 系の modeling が、conscious access を **precision と temporally deep policy selection** に依存するものとして扱っている点である。ここでは working-memory gating が cognitive action としてモデル化され、report / no-report 操作に応じて late signature は変わっても accessibility 自体は残りうる。 citeturn17view0turn7view5

### 人工システムにおける consciousness-like の最小定義と強い定義

実用的には段階的な定義がよい。

**弱い定義: GNW 型 access**  
ある表象状態が bottleneck / workspace を通じて複数の専門モジュールへ **globally available** になり、その状態遷移が局所処理と global availability を分ける **ignition 的非線形性** を示すなら、そのシステムは consciousness-like *access* を持つとみなす。これは GNW の機能的役割の直接翻訳である。 citeturn28view1turn21view2turn11view0turn5search1

**より強い定義: access + metacognitive control**  
どの情報が workspace に入り、どれだけ維持されるかを制御するために、システムが **明示的な confidence / uncertainty 表象** を形成し利用することを追加条件とする。これは active inference が access / report を temporally deep policy selection と precision control に結びつける立場と整合する。 citeturn17view0turn16view1turn15view0

**さらに強い定義: access + self/non-self boundary + self-maintenance constraints**  
Markov blanket 的 interface を伴う boundary model と、homeostatic variable や resource budget のような viability constraint を追加し、access state が self-maintaining な control loop に埋め込まれることを要求する。ただし Markov blanket は統計的ツールであり、追加前提なしに metaphysical な boundary へ拡張することには批判がある。 citeturn19view0turn2search0turn18view0turn2search9

重要な注意は、これらはどれも依然として**機能的**定義だということである。主観性そのものを解決するものではない。この限界は、ERIE の文書に明記しておくべきである。そうすることで、category error を避けつつ、反証可能な工学的主張を保てる。 citeturn28view1turn25view0turn20view0

### 理論間の比較表

| 用語 | FEP / Active Inference | GNW / Global Workspace | Recurrent Processing Theory | IIT | Enactivism / Autopoiesis（工学的読解） |
|---|---|---|---|---|---|
| **precision** | inverse variance/covariance、prediction error への gain、expected precision と policy precision を含む | 中核概念ではないが attentional amplification / gating を通じて間接的に関わる | 中核概念ではない | 中核概念ではない | 中核概念ではなく、autonomy と coupling が中心 |
| **conscious access** | 十分な precision のもとで temporally deep policy selection / higher-level inference に利用可能になること | 多数の processor への global broadcast によって情報が conscious になること | awareness に access/report は必須ではない | 中心概念ではない | 中心概念ではなく sense-making と autonomy が中心 |
| **consciousness-like property** | inferential depth, precision control, coherent global belief sharing などが候補 | global broadcast による functional consciousness | recurrent sensory processing による phenomenal / perceptual consciousness | integrated information / cause-effect power | 自己産出的組織と autonomy に結びつく |
| **global availability** | global belief-sharing / message passing として実装可能だが自動ではない | 定義の中核 | 必須ではない | 定義の中核ではない | coupling と機能組織から生じる |
| **ignition** | precision control 下の bifurcation / phase transition として形式化しうる | sudden, coherent, exclusive workspace activation が中核 | 非線形 recurrence はありうるが global ignition は必須ではない | 定義的ではない | 標準用語ではない |

## ERIE アーキテクチャへの写像

この節は、上記の定義を TRM-A（world model）、TRM-B（boundary）、GNW（integration / global broadcast）に写像する。焦点は「ERIE がその語を正当に使うために何が必要か」、すなわち necessary condition と falsifier である。 citeturn28view1turn9view0turn19view0turn18view0turn17view0

### TRM-A における precision の表現と検証

**本質**  
precision は prediction error の belief updating への影響を制御する重みであり、Gaussian / Laplace 仮定では inverse variance / inverse covariance として表される。policy selection にも関与しうる。 citeturn12view0turn9view0turn16view1

**必要条件**  
ERIE には、prediction error が **precision-weighted** される明示的な inference / update path が必要である。単に uncertainty を吐く forward predictor だけでは不十分で、precision は **gain として使われている**必要がある。 citeturn9view0turn7view2

**実装例**  
- TRM-A が `(pred_mean, pred_logvar)` を出し、ERIE が `Π̂ = exp(-pred_logvar)` を likelihood / transition precision proxy として計算する。これは heteroscedastic Gaussian NLL と整合する。 citeturn24view2
- ERIE が Π̂ を prediction error 項に乗じて state estimation を更新する。predictive coding 的 update rule や variational objective の重みとして使ってよい。 citeturn9view0turn12view0

**なぜこれだけでは Friston precision 全体ではないか**  
- `pred_logvar` 由来の Π̂ は自然には **aleatoric** であり、Friston 的 expected precision のような context-sensitive gating variable とは一致しない。 citeturn9view0turn15view3
- epistemic confidence が必要なら ensemble や posterior sampling を追加し、得られた dispersion を Π̂ と分けて扱うべきである。 citeturn13search5turn15view0turn24view2

**Proxy measure**  
- **Calibration + proper scoring**: negative log-likelihood, calibration error などで Π̂ が装飾ではなく意味のある量か確認する。 citeturn24view1turn13search4
- **Causal gain test**: precision weighting を ablation / clamp し、belief update が予測どおり過敏 / 鈍感になるかを見る。 citeturn9view0turn15view3

**反証条件**  
- Π̂ と realized residual の相関が弱く、標準的 calibration 手法を入れても calibration が悪ければ、`pred_logvar` を信頼できる precision proxy として扱うべきではない。 citeturn24view1turn13search4
- precision weighting を除去しても update magnitude がほぼ変わらないなら、precision は active inference の意味では実装されていない。 citeturn7view2turn9view0

### TRM-B における self/non-self boundary

**本質**  
Markov blanket は graphical model 上の conditional-independence boundary であり、internal, external, blanket states を分ける。active inference の用法では、internal と external は blanket を条件に独立になる。 citeturn2search0turn19view0

**重要な注意**  
Markov blanket は、Bayesian network / variational inference における認識論的ツールから、physical self の metaphysical boundary へ過剰拡張されがちだという批判がある。ERIE では、TRM-B はあくまで**検証可能な engineered conditional-independence interface** を実装するものとして扱うべきである。 citeturn18view0turn19view0

**必要条件**  
TRM-B は、「内側」と「外側」の情報流が明示的 interface variable を介してのみ生じるようにしなければならない。すなわち sensory in / action out を経ずに internal module が external state に直接依存してはならない。 citeturn2search0turn19view0

**Proxy measure**  
- **Graph / interface audit**: すべての cross-boundary dependency が指定された variable を介しているか確認する。  
- **Conditional independence residual**: interface variable を条件とした internal と external の conditional mutual information を測る。 citeturn19view0turn18view0
- **Viability maintenance**: energy budget, memory integrity, uptime constraint などの viability 変数を定義し、boundary policy が perturbation 下でもそれを維持できるかを見る。 citeturn19view0turn2search9

**反証条件**  
- hidden side-channel により boundary を迂回して性能を出しているなら、概念上 boundary を仮定していても、操作的には実現していない。

### GNW 層における global availability, broadcast, ignition

**本質**  
conscious access は、workspace / broadcast architecture を通じて情報が多数の processor に広く利用可能になることとして定義され、その際に ignition（非線形増幅、協調的安定化、競合排除）がしばしば伴う。 citeturn28view1turn11view0turn21view2

**必要条件**  
ERIE は以下を含まなければならない。

1. **workspace / bottleneck state**  
   capacity-limited であり、少なくとも「いま何が broadcast されるか」のレベルでは serializable である。 citeturn28view1turn4search9
2. **broadcast mechanism**  
   workspace content が多数の module に因果的に利用可能になる。 citeturn28view1turn10view3
3. **competitive selection + recurrent stabilization loop**  
   broadcast が単なる bus ではなく、ignition-like attractor に入れること。 citeturn21view2turn11view0

**人工システムにおける ignition の定義**  
ignition は PFC 活動である必要はなく、動力学的署名として定義できる。

- input strength / attention / prior を変化させたとき、global workspace activation が鋭い非線形転移を示す。 citeturn21view2turn21view1
- ある表象が access を勝ち取った瞬間に、cross-module coupling / coherence が急増し、代替候補が抑制される。 citeturn28view1turn11view0

**なぜ report/no-report の交絡が重要か**  
late GNW signature の多くは post-perceptual / report processing を追っている可能性がある。no-report paradigm では、P3b は報告がないと消えうる。したがって ERIE の global availability metric は、**report-based ではなく causal / capability-based** でなければならない。つまり「その状態で各 module が何をできるか」で測るべきである。 citeturn28view0turn17view0turn21view0

### Precision-weighted broadcast はどこで整合し、どこで飛躍するか

**整合する点**

- GNW は workspace への entry の gating に attention-like amplification を使い、active inference は attention-like effect を precision control と結びつける。したがって、**precision が workspace に何が入り支配するかを決める control signal になる**という橋渡しは妥当である。 citeturn11view0turn9view0turn7view2
- recent formal work は、GNW 的要素を持つ active inference model を構築しており、precision control によって working-memory gating や report / no-report pattern を説明している。 citeturn7view5turn17view0

**飛躍する点**

- broadcast を単なる weighted sum の全域ルーティングにすると、**非線形・排他的・metastable な workspace state** としての GNW ignition を失いかねない。 citeturn21view2turn28view1
- 「高 precision なら conscious」と同一視すると、attention / precision と consciousness を混同してしまう。これは文献上明示的に争点であり、両者は解離しうる。 citeturn21view0turn7view5
- GNW と IIT をめぐる大規模 adversarial test ですら、両理論に部分支持・部分反証が出ている。したがって、`precision-weighted broadcast = GNW` は定説ではなく、**明確な failure mode を伴う model hypothesis** として扱うべきである。 citeturn20view0turn20view1

## 採用すべき定義案

### 採用してよい定義

**Precision（ERIE 採用定義）**  
precision は、  
1. prediction error を belief updating の中で重みづける  
2. 必要なら policy selection confidence（policy precision γ）も制御する  
inverse variance / covariance 由来の control-relevant parameter である。これは predictive coding と active inference が precision を gain および inverse-temperature-like confidence と結びつける定式化に整合する。 citeturn9view0turn12view0turn16view1turn15view0

**Conscious access（ERIE 採用定義）**  
ある表象が workspace / broadcast architecture を通じて複数の専門モジュールから因果的に利用可能になり、通常は nonlinear ignition と recurrent stabilization を伴う状態。これは GNW の機能的コミットメントを直接移した定義である。 citeturn28view1turn11view0turn21view2turn10view3

**Consciousness-like property（最小運用定義）**  
強い条件が満たされるまでは、ERIE の consciousness-like property は **GNW-like access** に限定して主張する。すなわち measurable global availability + ignition dynamics + cross-module causal integration があればよい。ただし phenomenal consciousness は含意しない。 citeturn28view1turn21view2turn20view0

### 工学的 proxy として採用してよいもの

**`pred_logvar` 由来の precision proxy**  
calibration と causal gain test に通るなら、\( \exp(-\texttt{pred\_logvar}) \) を likelihood / transition precision proxy として使ってよい。これは heteroscedastic regression と整合し、precision-weighted inference / broadcast pipeline に接続できる。 citeturn24view2turn24view1

**Ignition proxy**  
workspace activation が、  
- control parameter に対して急峻 / 非線形  
- metastable / 持続的  
- 排他的  
になる regime change として定義する。これは attentional blink 研究や GNW の ignition 記述と整合する。 citeturn21view2turn28view1turn11view0

**Global availability proxy**  
ある late ERP 的署名ではなく、workspace content によって因果的に挙動が変わる module の**数と多様性**で global availability を測る。これは no-report 研究と GNW の「多数の processor による利用可能性」という定義に沿う。 citeturn28view0turn28view1turn21view0

## 保留すべき問題

### 現状では主張を保留すべきもの

**追加構造なしで `pred_logvar` を Friston precision と同一視すること**  
precision を最適化 / 制御する inferential loop がなく、aleatoric と epistemic uncertainty の分離もない段階で、`pred_logvar` を Friston precision と呼ぶのは誤解を招く。active inference における precision は modulatory / gain quantity であり、ML 的 log-variance output はその部分的一致にすぎない。 citeturn9view0turn15view3turn24view2turn13search5

**Markov blanket を self の metaphysical boundary と強く読むこと**  
blanket 形式は inference tool から metaphysical boundary claim へ過剰拡張されがちである。ERIE では、検証できる engineered boundary property だけを主張すべきである。 citeturn18view0turn19view0turn2search0

**consciousness-like property の十分条件を主張すること**  
神経科学でも competing theory は何が十分条件かで一致しておらず、GNW と IIT の近年の adversarial test も両者に課題を示している。ERIE では access に関する強い operational criterion は置けるが、subjective consciousness の十分条件を主張すべきではない。 citeturn20view0turn25view0turn27view0

### ERIE 設計に直接効く未解決課題

**Precision と conscious access の関係**  
precision が ignition / workspace entry の attentional gating として機能するという橋渡しは plausibility が高く、active inference 系のモデルもそれを使っている。しかし attention と consciousness の解離文献を踏まえると、precision は一部の ignition regime に必要でも、consciousness-like access そのものと同一ではない。 citeturn17view0turn7view5turn21view0turn11view0

**モジュール型人工システムで ignition をどこに置くか**  
GNW と RPT は global frontoparietal ignition の必要性で一致していない。ERIE では、ignition を解剖学的場所ではなく **system-level regime change** として定義すべきである。 citeturn28view0turn27view0turn20view0

**“global availability” の定量指標**  
単に broadcast bus があるだけでは弱すぎる。強い metric は causal で、capacity-limited で、behavioral flexibility に結びついていなければならない。そうでないと定義が単なる generic routing に崩れてしまう。 citeturn28view1turn4search9turn10view3

## 主要文献

以下の文献群が、上記定義と操作化の中核を支えている。

- Laplace / predictive coding における **inverse variance / covariance としての precision**、および precision-weighted prediction error。 citeturn12view0turn9view0
- **synaptic gain としての precision**、および attention / neuromodulation との接続。 citeturn9view0turn7view2turn15view3
- **policy precision (γ)** を inverse temperature / confidence と見る議論、および dopamine を expected precision とみなす議論。 citeturn16view1turn15view0
- **global workspace / GNW** の基礎、すなわち global availability, broadcast, ignition の機能的役割。 citeturn5search1turn5search2turn11view0turn28view1
- attentional blink 等における **all-or-none / bifurcation 的 ignition** の証拠。 citeturn21view2turn21view1
- **attention と consciousness の解離**。 citeturn21view0
- no-report paradigm と、late signature が report / WM demand を追う可能性の批判。 citeturn28view0turn17view0
- active inference による **GNW 的 access の形式化**。predictive global workspace, WM gating as cognitive action, precision control in access models など。 citeturn7view5turn17view0
- **RPT** としての contrast class。 citeturn27view3turn27view0
- **IIT** の原因結果構造と Φ を中心とする contrast。 citeturn25view0turn3search7
- **Markov blanket**, active inference, autopoiesis 的読解と、その過剰拡張への批判。 citeturn19view0turn2search0turn18view0turn2search9
- GNW vs IIT の大規模理論比較と、双方への challenge という結論。 citeturn20view0turn20view1
- `pred_logvar` のような uncertainty output と precision-like weighting の工学的接続。 citeturn24view2turn13search5
- probabilistic confidence / uncertainty output の評価としての calibration / proper scoring。 citeturn24view1turn13search4
