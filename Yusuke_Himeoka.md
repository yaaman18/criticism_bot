# Yusuke_Himeoka

- Source: Yusuke_Himeoka.pdf
- Pages: 67

## Page 1

APS/123-QED
Local stabilizability implies global controllability
in catalytically-controlled reaction systems
Yusuke Himeoka, 1, 2, ∗ Shuhei A. Horiguchi, 3, 4, 2 Naoto
Shiraishi,5 Fangzhou Xiao, 6 and Tetsuya J. Kobayashi 1, 4, 7, 2
1Universal Biology Institute, University of Tokyo,
7-3-1 Hongo, Bunkyo-ku, Tokyo, 113-0033, Japan
2Theoretical Sciences Visiting Program (TSVP),
Okinawa Institute of Science and Technology Graduate Unive rsity, Onna, 904-0495, Japan
3Nano Life Science Institute, Kanazawa University,
Kakumamachi, Kanazawa, 920-1192, Japan
4Institute of Industrial Science, The University of Tokyo,
4-6-1, Komaba, Meguro-ku, Tokyo 153-8505, Japan
5Faculty of arts and sciences, University of Tokyo,
3-8-1 Komaba, Meguro-ku, Tokyo, Japan
6Westlake University, School of Engineering,
600 Dunyu Road, Xihu District, Hangzhou, China
7Department of Mathematical Informatics,
Graduate School of Information Science and Technology,
The University of Tokyo, 7-3-1, Hongo,
Bunkyo-ku, Tokyo 113-8656, Japan
(Dated: February 26, 2026)
1
arXiv:2505.06834v3  [physics.bio-ph]  25 Feb 2026

## Page 2

Abstract
Controlling complex reaction networks is a fundamental cha llenge in the ﬁelds of physics, chem-
istry, biology, and systems engineering. Here, we prove a ge neral principle for catalytically-
controlled reaction systems with kinetics where the reacti on order and the stoichiometric coeﬃcient
match: the local stabilizability of a given state implies gl obal controllability within its stoichiometric
compatibility class. In other words, if a target state can be maintained against small perturba-
tions by a catalytic control, the system can be catalyticall y controlled from any initial condition to
that state. This result highlights a tight link between the l ocal and global dynamics of nonlinear
chemical reaction systems, and clear relationship between the controllability and thermodynamic
consistency of the reaction systems. The ﬁndings illuminat e the robustness of biochemical systems
and oﬀers a way to control catalytic reaction systems in a gene ric framework.
I. INTRODUCTION
Understanding the controllability of (bio)chemical reaction networ ks is crucial for both
theoretical insights and practical applications in systems biology an d chemical/metabolic
engineering. Controllability, in the control-theoretic sense, refer s to the ability to steer a
dynamical system from any given initial state to any desired ﬁnal st ate using suitable inputs.
In systems chemistry, networks of interacting molecules serve as programmable archi-
tectures whose emergent behaviors under out-of-equilibrium con ditions can be harnessed to
access speciﬁc functional states through precise chemical desig n [1–3]. For biological studies,
controllability translates to being able to drive a biochemical system, such as a metabolic or
gene regulatory network, to a desired state by adjusting certain control parameters such as
enzyme concentrations, amounts of the transcriptional/transla tional machineries, or external
conditions.
Historically, systems biology has been rooted in control theory [4]. T he early development
of the ﬁeld was initiated by ﬁnding feedback controls in the biochemica l systems for the
robust adaptation [5–7]. This approach has been extended to cons truct artiﬁcial biochemical
systems with a variety of functions [8–15], and for designing cell-cell interactions, including
artiﬁcial cellular diﬀerentiation, pattern formation in multicellular sys tems, and synthetic
∗ yhimeoka@g.ecc.u-tokyo.ac.jp
2

## Page 3

ecosystems [16–22]. In addition, the value of the controllability fram ework is well recognized
in application ﬁelds such as metabolic engineering [23, 24] and epidemiolo gy for policy
making [25, 26].
Among various (bio)chemical systems, we focus on catalytic react ion systems as they
provide the indispensable foundation for both biological metabolism a nd the bottom-up
construction of out-of-equilibrium synthetic systems. The import ance of catalysis stems
from its ability to exert kinetic control over otherwise dormant rea ctions; in biological con-
texts, most metabolic reactions are kinetically inhibited and would not proceed on relevant
timescales without enzymatic acceleration [27]. This constraint rath er enables living systems
to exert precise control over their internal states by modulating catalyst activities. Simi-
larly, in systems chemistry and bottom-up synthetic biology, cataly sis serves as an essential
“control knob” for driving systems away from equilibrium. Researc hers utilize catalytic
and autocatalytic feedback motifs to program functions—such as bistability, oscillations,
and dissipative self-assembly—into chemical reaction networks [1, 2 8]. By integrating these
metabolic-like pathways into microcompartments, it becomes possib le to assemble artiﬁcial
cells with controllable metabolic functions [29]. Thus, as the functiona l dynamics of both
natural and synthetic systems are fundamentally governed by th e modulation of reaction
rates through catalysis, catalytic reaction systems oﬀer a uniﬁed and necessary framework
for their control.
Establishing the theories on the control of (bio)chemical reaction s is crucial for deepening
our understanding of bottom-up constructions of artiﬁcial syst ems, cellular homeostasis,
and biological adaptation. While links between the passive responsiveness of biochemical
reactions to external perturbations and network topologies hav e been actively studied [30–
32], the global controllability of such catalytic reaction networks re mains largely unexplored.
Progress in the theory of controllability of cellular states is indispens able for deciphering
mechanisms that enable biological systems to adjust ﬂexibly to vario us environments [33–
39].
However, the application of the classical control theory to (bio)c hemical reaction net-
works presents unique challenges. Main diﬃculties arise from the larg e number of chemical
species, nonlinearity of the reaction rate functions, and non-neg ativity constraints on the
control parameters. The nonlinearity of reaction rate functions arises inherently from t he
multi-molecular reactions required to synthesize larger molecules fr om smaller components.
3

## Page 4

Because chemical reactions are mass-conserving, in unimolecular r eactions such as A ⇌ B,
the substrate and product must have identical mass. However, b uilding larger structures
necessitates multi-molecular reactions 1 such as A + B ⇌ C, where the mass of product C
exceeds that of A or B individually. For such reactions, the forward reaction rate depends
on the concentrations 2 of both A and B, thereby introducing nonlinearity into the system
dynamics [40, 41].
Classical control theory was originally developed for mechanical sy stems, such as vehicles
and aircraft. In these systems, typical control inputs, such as accelerators and brakes, allow
for both positive and negative acceleration. In contrast, contro l parameters in (bio)chemical
systems—typically the concentrations or activities of catalysts—a re inherently non-negative.
This non-negativity constraint poses a unique challenge for evaluat ing the controllability of
such systems [42].
In addition to nonlinearity and non-negativity, the evaluation of glob al controllability is
often desired, in addition to the local controllability. These three po ints lead to the sign-
constrained global nonlinear controllability problem. Controllability problems in this class are
highly nontrivial and unexplored by classical frameworks for the co ntrollability of chemical
reaction systems [42–46].
In our previous work, we developed a numerical method for eﬃcient computation of the
controllability of catalytic reaction systems [47]. This method allows us to convert the sign-
constrained global nonlinear controllability problem into a problem of ﬁ nding appropriate
conical combinations (non-negative linear combinations) of the vec tors. However, we were
unsuccessful in analytically identifying the controllability of the syst em.
In this study, we show that local stabilizability and global controllabilit y are tightly
coupled in catalytic reaction systems. In particular, we show for a w ide class of models that
if a state is locally stabilizable by feedback control, then all states ar e globally controllable
to the state by manipulating the activities of catalysts.
The main text outlines the central concepts of global controllability in catalytic reac-
tion systems, whereas the Supplementary Information provides t he rigorous mathematical
foundations in a deﬁnition-theorem-proof style.
1 Here, “multi” implies that the number of molecules on either the subst rate or product side is greater than
one.
2 This requirement is related to the consistency condition in chemical reaction network theory [40], which
guarantees that chemical concentrations do not become negativ e given initially non-negative concentra-
tions. 4

## Page 5

II. CONTROLLABILITY OF THE CA T AL YTIC REACTION SYSTEMS
In the present paper, we focus on a well-mixed, deterministic react ion rate equation model
with N chemical species and R reactions. Additionally, we deal with the case in which all
reactions are independently controllable to evaluate the maximum po ssible controllability
of the system. Then, the model equation is described by the followin g ordinary diﬀerential
equation with input-aﬃne, no-drift control:
dx
dt = Su(t) ⊙ [vf (x(t)) − vb(x(t))]
= Su(t) ⊙ v(x(t)), (1)
where x ∈ RN
≥ 0 is the vector of chemical species’ concentrations, and S is the N × R stoichio-
metric matrix. Each column of the stoichiometric matrix is called stoichiometric vectors .
The rth stoichiometric vector Sr represents the state transition of the system in the phase
space by the rth reaction. vf (x) and vb(x) are the forward- and backward reaction ﬂuxes,
respectively. The diﬀerence between the forward and backward r eaction ﬂuxes reads the net
ﬂuxes v(x). u(t) : R≥ 0 → RR
≥ 0 is the vector of time-dependent control. ⊙ is the Hadamard
(element-wise) product of vectors. In the present study, we mo del the net ﬂux (the diﬀerence
between the forward and backward reaction ﬂuxes) as a single rea ction ﬂux, and the control
is the modulation of catalysts’ activities. Thus, the control param eters increase and decrease
both the forward and backward reaction rates simultaneously, an d ui(t)′s are non-negative.
It is not possible to control the forward and backward reaction ra tes independently as cat-
alysts only change the activation energy of the chemical reactions , but do not change the
chemical equilibrium [41].
The ODE system (Eq. (1)) can be used as a model equation for a wide variety of the
biochemical systems. The well-mixed catalytic reaction system in the test tube regardless of
whether they are biological or purely chemical, they are modeled usin g the equations. The
metabolic reaction system in a single cell is usually modeled using the abo ve equation [48–
52]. In addition, metabolic models using Eq. (1) can be extended to mu lticellular systems
in a simple manner. For the extension, we add an additional subscript on the control u,
reaction ﬂux v, and the concentration of metabolites x for indexing diﬀerent cells. The
exchanges of metabolites between diﬀerent cells can be modeled as c ontrolled reaction with
transporters or channels as controllers.
5

## Page 6

As a simple example, we adopt the Sel’kov model [53]. The Sel’kov model is a minimalist
kinetic model for glycolytic oscillations, characterized by a Hopf bifu rcation and limit-cycle
dynamics driven by autocatalysis. The schematic of the reaction dia gram is provided in
Fig. 1(a). The model comprises three reactions: ∅ ⇌ A, A ⇌ B, and B ⇌ ∅.3 The ordinary
diﬀerential equations of the concentrations of A and B, xA and xB, with control is given by
d
dt

 xA
xB

 =

 1 − 1 0
0 1 − 1







u1
u2
u3




 ⊙





vmax
1 (a − k1xA)
vmax
2 (c + xγ
B)(xA − k2xB)
vmax
3 (xB − k3b)




 .
(2)
Here, ui, (i = 1 , 2, 3) are considered to be the concentrations of catalytic enzymes o f the
corresponding reactions. While the Sel’kov model is a coarse-graine d model, if we were
to assign a real-world equivalent, enzymes of reaction 1 , 2, and 3 correspond to glucose-6-
phosphate isomerase, phosphofructokinase, and fructose-bis phosphate aldolase, respectively.
vmax
i are the maximum reaction speed of the corresponding enzyme. The enzyme of the
second reaction (phosphofructokinase) is an allosteric enzyme an d positively regulated by
ADP which is represented by the chemical B in the Sel’kov model. The re action rate increases
as xB increases from the basal level vmax
2 c with nonlinearity parameter γ > 0. vmax
1 a and
vmax
3 k3b are the supply rate of A and B, respectively. k′
is are the reversibility parameters. By
taking ki → 0 limit for all i′s, all reactions become irreversible and we restore the original
Sel’kov model with control. The stoichiometric vectors are drawn as the red, blue, and green
arrows in Fig. 1(b).
The main question is how much controllability the system (Eq. (1)) has : What are the
conditions for the source state xsrc and the target state xtgt to have a control input u(t)
that realizes the transition from xsrc to xtgt following Eq. (1)? To answer this question, we
introduce a mathematical deﬁnition of controllability.
Deﬁnition 1 (Controllable). The system is said to be asymptotically controllable from xsrc
to xtgt if there exists a control input u(t) : R≥ 0 → RR
≥ 0 such that the solution of Eq. (1) with
the control u(t) and the initial state x(0) = xsrc reaches xtgt, i.e., limt→∞ x(t) = xtgt.
While this deﬁntion is for the asymptotic controllability in a rigorous man ner, we shall use
the term “controllable” as a shorthand for asymptotically controlla ble throughout this paper.
3 While reactions in the original model are irreversible, we treat them a s reversible here. This is because,
as discussed in the section IV, models consisting solely of irreversible reactions generically exhibit trivial
controllability.
6

## Page 7

We gather all the initial states xsrc that can be controlled to x, and call it the controllable
set of x, which is denoted by C(x). In the present study, we identify the controllable set
of a given state x in catalytic reaction systems, in particular, what type of state can have
maximum controllability, that is, the largest possible controllable set. In the following, we
restrict our attention to the positive orthant RN
>0 to avoid the boundary problem [54].
Controllability is determined by the constraints imposed on the syste m. Here, we focus
on the thermodynamic constraint that the catalysts cannot chan ge the chemical equilibrium.
Aside from this constraint, chemical reaction network systems, in general, have the network-
level constraints, which are conceptualized by the stoichiometric compatibility class (SCC)
[40]. SCC is given for each state x as the set of all states that can be reached from x by
controlling the reaction rates with the sign-free control input u(t) : R≥ 0 → RR in Eq. (1).
SCC is given by the parallel translocation of the full linear span of the stoichiometric vectors;
W(x) := {x +
R∑
i=1
aiSi |ai ∈ R} ∩ RN
>0. (3)
If a given pair of states xsrc and xtgt is not in the same SCC, any control is infeasible between
xsrc and xtgt. SCC of the Sel’kov model (Eq. (2)) is the whole positive orthant R2
>0, while
for instance, if one restrict the model to have only the second rea ction A ⇌ B, SCC becomes
the anti-diagonal line, W(x) = {y ∈ R2
>0 |yA + yB = xA + xB} (see Fig. 1(b)).
Lastly, we introduce the concept of local stabilizability for the catalytically controlled
chemical reaction systems for the later purpose [55].
Deﬁnition 2 (Local stabilizability). A state x∗ is said to be locally stabilizable if there exists
a state-dependent feedback control u(x) : RN
>0 → RR
≥ 0 such that x∗ is a locally asymptotically
stable ﬁxed point of the system Eq. (1) with u(x) as a control within its stoichiometric
compatibility class W(x∗).
Note that when we utilize a state-dependent control being only implic itly dependent
on t via state x, u(x(t)), as u(t), Eq. (1) becomes an autonomous dynamical system.
So, the stabilizability can be rephrased with the dynamical-systems la nguage; a system is
locally stabilizable if there exists a choice of state-dependent contr ol u(x) such that every
eigenvalues of the Jacobian matrix
J = S∂(u(x) ⊙ v(x))
∂x
7

## Page 8

FIG. 1. (a) A schematic illustration of the Sel’kov model. Th e black arrows represents the
substrate-product relationship of each reaction, whereas the red arrow is the feedback activa-
tion. (b) The stoichiometric vectors of the reactions (red, blue, and green arrows) in the phase
space. The stoichiometric compatibility class for x(i) (i = 1 , 2, 3) of the model consisting only of
the reaction R2 are depicted as the broken lines.
has the negative real part except the zero eigenvalues correspo nding to the left null-space of
S.
III. CA T AL YTIC CONTROL AND CONICAL COMBINA TIONS
In addition to the network-level constraint conceptualized by SCC , we have another
constraint on the control of catalytic reaction systems. This con straint originates from the
feature that the catalysts only change the activation energy of t he chemical reactions [41].
As a consequence, the catalysts cannot directly control the dire ctionality of the reactions,
but the directions are set by the concentrations of the metabolite s as the sign of the reaction
rate function vr(x) in Eq. (1). Therefore, the linear combination of stoichiometric vec tors
is no longer a way to evaluate the controllability of the system. Howev er, this approach
8

## Page 9

can be extended to evaluate the controllability of the catalytic reac tion system. The key
point is again that catalytic control does not change the chemical e quilibrium, and thus, the
directionality of the reaction is set by the concentrations of the me tabolites. This allows us
to partition the phase space into subregions where the directionalit y of the reaction is ﬁxed
and the conical combination of the stoichiometric vectors tells us th e controllability.
The purpose of the following paragraphs is to partition phase space into regions that we
call the cell where the directionalities of the reactions are ﬁxed. For this purpo se, we assume
that each reaction rate function vr(x) has the following decomposition:
vr(x) = fr(x)pr(x) (4)
fr(x) > 0 (5)
pr(x) =
N∏
i=1
x
n+
i,r
i − kr
N∏
i=1
x
n−
i,r
i , (6)
where n±
i,r represents the reaction order of the ith metabolite of the forward ( n+
i,r ), and the
backward ( n−
i,r ) reaction of the rth reaction. kr > 0 is the reversibility constant of the rth
reaction.4 We term p(x) and f (x) the thermodynamic part and the kinetic part, respectively.
Note that this assumption holds for most of the popular biochemical reaction rate kinetics
such as mass-action kinetics, (generalized) Michaelis-Menten kinet ics, ordered- and random
multi-molecular reaction kinetics, and ping-pong kinetics [56, 57].
An important feature of the reaction kinetics of this form is that th e direction of the
reaction is set by the thermodynamic part p(x) as it is related to the thermodynamic
force of the reaction. On the other hand, the remaining part f (x) is purely kinetic, and it
modulates only the absolute value of the reaction rate, but not the direction. Let us consider
the Michaelis-Menten kinetics, vmax(xS − kxP )/ (KM + xS + xP ) with vmax, KM , k as the
maximum reaction rate, Michaelis-Menten constant, and reversibilit y constant, respectively.
In this case, the thermodynamic part corresponds to its numerat or, xS − kxP , whereas the
remaining part vmax/ (KM + xS + xP ) is the kinetic part.
Here, we introduce the main concepts: the balance manifold Mr and the cell C(σ ).5
Deﬁnition 3 (Balance Manifold) . The balance manifold of the rth reaction, Mr is deﬁned
as zero locus of pr(x).
Mr = {x ∈ RN
>0 | pr(x) = 0 }, (7)
4 The results in the paper can be extended to the case with k ∈ RR
≥ 0. See SI text for the details.
5 The balance manifold and cell are termed “null-reaction manifold” and “direction subset” in [47].
9

## Page 10

The phase space is then partitioned by the balance manifolds {Mr}R
r=1 into regions that
we call cells. In a cell C(σ ), the reaction directions are ﬁxed and represented by a binary 6
vector σ ∈ {− 1, 1}R. The cell is formally given by the following
Deﬁnition 4 (Cell). A subset of RN
>0 deﬁned below is called a cell with reaction direction
σ , C(σ )
C(σ ) = {x ∈ RN
>0 | sgn p(x) = σ }. (8)
Hereafter, “cell” is used to denote a region in the phase space, not a biological cell. An
important note is that the geometries of Mr and C(σ ) remain invariant under the catalytic
control.
Because the directionality of the reactions in the cell C(σ ) is σ , the conical hull of the
directed stoichiometric vectors {σiSi}R
i=1 plays a central role in evaluating controllability
within the cell. In parallel to the deﬁnition of SCC, we deﬁne the stoichiometric cone [40]
of state x with the direction σ as
Vσ (x) := {x +
R∑
i=1
aiσiSi |ai ≥ 0} ∩ RN
>0. (9)
The set of all conical combinations of the set of vectors {wi}R
i=1 is often called the conical
hull and is denoted by cone( {wi}R
i=1) := {∑R
i=1 aiwi |ai ≥ 0}. The stoichiometric cone is a
parallel translation of the conical hull of the directed stoichiometr ic vectors. The geometry
of the balance manifolds and cells of the Sel’kov model (Eq. (2)) are s hown in Fig. 2.
Let us demonstrate how we can evaluate the controllability of the sy stem using a cell.
Assume that the trajectory of Eq. (1) is conﬁned within a single cell C(σ ) during the interval
t ∈ [t0, t 1]. Recalling that the balance manifold and cell are independent of the c ontrol u(t),
the trajectory is expressed as
x(t) = x(t0) + S
∫ t
t0
u(s) ⊙ f (x(s)) ⊙ p(x(s)) ds (10)
= x(t0) + Sσ ⊙
∫ t
t0
u(s) ⊙ f (x(s)) ⊙ |p(x(s))|ds (11)
= x(t0) + Sσ ⊙
∫ t
t0
˜u(s) ds (12)
6 For the readability, we work with the setup σ ∈ {− 1, 1}R of the problem in the main text, while the
rigorous setup should allow σi to be zero, i.e., σ ∈ {− 1, 0, 1}R. The proof in the SI text is provided with
this setup.
10

## Page 11

FIG. 2. The phase space of the Sel’kov model (Eq. (2)) are part itioned by the balance manifolds
Mr (red, blue, and green lines) into cells C(σ A), C (σ B), . . . , C (σ G). The directed stoichiometric
vectors {σiSi}3
i=1 are shown in each cell. The colors of the directed stoichiome tric vectors are the
same as the corresponding reactions. The yellow shaded regi on is the state reachable from xsrc by
the non-negative controls.
One target state xtgt, 1 is thus controllable from xsrc, while the other
target xtgt, 2 is not. In this model, only C(σ D) is a free cell.
where ˜u(t) = u(t) ⊙ f (x(t)) ⊙ | p(x(t))|. Since we have fr(x) > 0 and |pr(x)| > 0 for
x ∈ C(σ ) for 1 ≤ r ≤ R, there is a one-to-one correspondence between ˜u and u for ﬁxed x.
Therefore, the following statement holds: If there is a trajector y from x(0) := x(t0) to
x(1) := x(t1) following Eq. (1) and x(t) ∈ C(σ ) for t ∈ [t0, t 1], then there exists a non-
negative vector ˜u(t) ∈ RR
≥ 0 such that
x(t + ∆ t) = x(t) + Sσ ⊙ ˜u(t)∆ t. (13)
holds in the ∆ t → 0 limit for any t ∈ [t0, t 1]. Importantly, the converse of this statement
11

## Page 12

also holds. Suppose that there is a path x from x(0) to x(1) in the cell C(σ ) parameterized
by t ∈ [t0, t 1] so that x(t0) = x(0) and x(t1) = x(1) hold. If for any t ∈ [t0, t 1] there exists a
non-negative vector ˜u(t) ∈ RR
≥ 0 that satisﬁes Eq. (13) in the ∆ t → 0 limit, the path x(t) is
a solution of Eq. (1) with the control ui(t) = ˜ui(t)/ (pi(x(t))fi(x(t))).
Based on the above arguments, the global nonlinear controllability p roblem is reduced
to ﬁnding appropriate conical combinations in each cell. In Fig. 2, the balance manifolds,
cells, and directed stoichiometric vectors of the Sel’kov model (Eq. (2)) are presented. The
controllable region from the state xsrc (the gray point at the right middle of the ﬁgure) is
computed following the above argument, and is highlighted in yellow. If negative control is
allowed, the controllable region from xsrc is the entire space R2
≥ 0, whereas the controllable
region is restricted by the non-negativity of the control constra int. An intuitive restriction
is that the concentration of chemical A cannot increase further from the starting point xsrc
A
because xsrc
A is larger than a/k 1 and the equilibrium concentration of the reaction A ⇌ B. By
taking advantage of the mapping of the controllability problem to the conical combinations,
we can develop an eﬃcient method for numerically evaluating the cont rollable set. Further
details on the numerical approach are provided in [47].
Note that at the introduction of ˜u (Eq.(11) and (12)), the contribution of f (x) and
|p(x)|are cancelled by deﬁning ˜u by ˜u(t) = u(t) ⊙ f (x(t)) ⊙ |p(x(t))|. This means that the
nonlinearity in the kinetic part, e.g., feedback regulation of enzyme a ctivity, has no eﬀect
on the controllability because this activation can be counteracted b y modulating u. In the
speciﬁc case of the Sel’kov model, the control cancels the feedbac k term ( c + xγ
B) in Eq. (2),
which plays a critical role in the Hopf bifurcation. Consequently, the Hopf bifurcation does
not aﬀect the controllability of the system.
IV. UNIVERSAL CONTROLLABILITY OF FREE CELLS
Thus far, we have shown that the controllability of catalytic reactio n systems is well-
captured by the conical combination inside each cell. In the following, we describe that there
are special type of cells, which we call the free cells , where the controllability is maximized:
An arbitrary pair of states inside the same free cell is mutually contr ollable as long as the
two are in the same SCC. Additionally, a given state can be locally stabiliz ed by feedback
control if and only if the state is in a free cell. Furthermore, as the m ain claim of the present
12

## Page 13

study, an arbitrary state in the free cell is globally controllable from any initial state in the
same SCC. This means that if a state is locally stabilizable, then it is globa lly controllable.
In this section, we suppose that SCC of the model (Eq. (1)) match es the whole space RN
>0
for readability’s sake, whereas we provide proof for the general c ase in the SI text.
First, we formally introcude free cell ;
Deﬁnition 5 (Free Cell). A cell C(σ ) is termed a free cell if the set of conical combinations
of its directed stoichiometric vectors coincides with its f ull linear span:
cone{σiSi}R
i=1 = span{σiSi}R
i=1.
In practical terms, within a free cell C(σ ) the control is “unrestricted”: For any path
connecting two arbitrarily chosen states x, y ∈ C(σ ) there is always a choice of ˜u ∈ RR
≥ 0
satisfying Eq. (13) in ∆ t → 0 limit because the conical combination of the directed stoi-
chiometric vectors equals to their full linear span. C(σ D) in Fig. 2 is the free cell of the
Sel’kov model (Eq. (2)) and the others are non-free cells.
The reaction directions in this free
cell are all positive, and the reaction ﬂux ﬂows from the top to the b ottom of the network
(Fig. 1(a)): ∅ → A, A → B, and B → ∅ . Note that the freeness of the cell is set solely
by the function form of the thermodynamic part of the reaction ra te function p(x) and re-
versibility constant k. The structure of the cells is independent of the choice of, for inst ance,
the maximum catalytic speed vmax and the Michaelis-Menten constant Km in generalized
Michaelis-Menten kinetics.
The property of the free cell oﬀers resilience, or homeostasis in th e biological context, of
the system to external perturbations. Suppose that the syste m is perturbed within a free
cell from x ∈ C(σ ) to x′ ∈ C(σ ). If C(σ ) is a free cell, it is always possible to control the
system back to its original state x (see Fig. 3(a)). In contrast, if C(σ ) is a non-free cell,
there are perturbation directions that cannot be counteracted , regardless of the perturbation
strength (Fig. 3(b)). This intuition leads to two useful proposition s about the free cell.
Proposition 1 (Stabilizability in free cells) . For any state x∗ ∈ C(σ ), there exists a control
that locally stabilizes x∗ within C(σ ) if and only if C(σ ) is a free cell.
Proposition 2 (Steady-states and freeness). A cell C(σ ) is a free cell if and only if it admits
a ﬁxed point x∗ ∈ C(σ ), either stable or unstable, under a constant control u(t) = uc ̸= 0.
13

## Page 14

FIG. 3. A visual comparison of controllability in free versu s non-free cells. Yellow region is the
stoichiometric cone inside the cell. (a) A perturbation fro m x to x′ (dashed arrow) in a free cell
can be counteracted, restoring x (solid red arc), as the conical combinations span the cell. ( b) In
a non-free cell, the stoichiometric cone of x′ (yellow region) is insuﬃcient to return the system to
x.
Prop. S1 and S2 are proven in general setup in the SI text (Prop. S 1 and S2).
The signiﬁcance of free cells in terms of control is not only on local st abilizability, but
also on global controllability. For global controllability, we need to intr oduce a class of
reaction rate functions, which we call the stoichiometrically compatible kinetics (SCK). SCK
is a class of reaction rate functions in which the reaction order matc hes the corresponding
stoichiometric coeﬃcients.
Deﬁnition 6 (Stoichiometrically Compatibe Kinetics) . A reaction rate function vr(x) is
stoichiometrically compatible kinetics if it satisﬁes the decomposition in Eqs. (4)–(6) and
the relation Si,r = n−
i,r − n+
i,r . Here;
• Si,r is the stoichiometric coeﬃcient of the i-th species in the r-th reaction.
• n+
i,r and n−
i,r are the reaction orders for the forward and backward reactions, respectively.
The SCK is closely linked to the balancing condition of the chemical pote ntial between
substrates and products in the ideal-gas framework. Consider a r eaction nA ⇌ mB, where
n and m are the stoichiometric coeﬃcients of the substrate A and product B, respectively.
At chemical equilibrium, the energy balance is expressed as nµ A = mµ B, where µ ∗ denotes
14

## Page 15

the chemical potential of either A or B. In the ideal-gas approximat ion, the chemical po-
tential is given by µ ∗ = µ 0
∗ + ln x∗, where x∗ is the concentration of the chemical species 7.
Substituting this expression for the chemical potential, taking the exponential of both sides
of the equilibrium condition, and rearranging terms yields
xn
A − kxm
B = 0, (14)
where k = exp( mµ 0
B − nµ 0
A). Note that the left-hand side of Eq.(14) takes the form of the
thermodynamic part of the reaction rate function (Eq.(6)). By em ploying the left-hand side
of Eq.(14) as the reaction rate in energy-imbalanced states, we ob tain a rate function imple-
mented by SCK. While Eq.(14) determines the energy-balanced stat e, the overall reaction
rate retains a degree of freedom. This freedom is represented by the kinetic part f (x) > 0
(Eq.(5)), whose speciﬁc form depends on the details of the reactio n mechanism. In the fol-
lowing, we refer to models in which all reactions are implemented by the SCK as full SCK
models.
Although the SCK is intimately related to chemical equilibrium, a full SCK model can
attain non-equilibrium steady states. The Sel’kov model (Eq.(2) and Fig.1) is a full SCK
model while it can exhibit a nonzero steady reaction ﬂux, and even mo re, a limit cycle oscil-
lation. For the system to relax to a detailed-balanced chemical equilib rium, it is necessary to
turn oﬀ any one of the three reactions by setting the correspond ing ui to zero. The Sel’kov
model relaxes to the detailed-balanced states by turning oﬀ one of the three reactions.
The ability of the Sel’kov model to exhibit both non-equilibrium steady s tates and
detailed-balanced chemical equilibrium by adjusting u is not coincidental; rather, it is a
deﬁning feature of the full SCK model, as proven in the SI text. Mod els that relax to
detailed-balanced chemical equilibrium are in the class of thermodynamically consistent mod-
els in the chemical thermodynamics literature [58]. The full SCK model is o ne from which
thermodynamically consistent models can be derived by switching oﬀ t he reactions
so that
the number of active ( ui ̸= 0) reactions is minimized while the rank of the stoichiometric
matrix is the same with the original model (see section V in SI text). However, the origi-
nal model is not necessarily thermodynamically consistent and may e xhibit non-equilibrium
behaviors such as non-equilibrium steady states, oscillations, and c haos.
7 Here, we set RT = 1, with R and T representing the gas constant and temperature, respectively.
15

## Page 16

Our main claim is that if the reaction rate function v(x) of all reactions in the model
(Eq. (1)) is implemented by SCK (i.e., full SCK model), then any state in arbitrary free cell
can be controlled from any state (A simpliﬁed version of Corollary. S1 in the SI text. We
have Theorem. S1 for a general setup), when formally stated,
Theorem 1 (Main Theorem) . If all reactions in the model are implemented by SCK, then
every state in any free cell is controllable from an arbitrar y state in RN
>0.
In other words, the controllable set of any state in the free cell co incides with the whole
space, C(x) = RN
>0 for x ∈ C(σ ) with C(σ ) being a free cell. 8
In the following, we provide an intuitive description of the claim, while th e full proof
is provided in the SI text. Recall that we have N chemical species and R reactions, and
R is greater than N because otherwise non-zero steady-state ﬂux is impossible. Also, we
use the term “a vertex of a cell” in the following meaning: The balance m anifold of each
reaction is the zero locus of binomial (see Eq. (6) and (7)). Thus, t he balance manifolds are
hyperplanes in the logarithmic-transformed positive orthant RN
>0, and the cells are polytope
or polyhedra in the space. So, each cell has vertices at the interse ction of at least N balance
manifolds of the reactions.
Let us denote the target state xtgt ∈ C(σ ), the source state xsrc, and C(σ ) as the free
cell. We take one of the vertices of the free cell C(σ ) and denote it as xeq. By reordering
the reaction indices, we can assume that the intersection of the ba lance manifolds {Mr}N
r=1
is xeq. Since the balance manifolds are the zero-locus of each reaction ra te function, xeq
is the detailed-balanced, chemical equilibrium state of the N reactions. Analogous to the
uniqueness and global stability of the equilibrium state in thermodyna mics, it is known that
the detailed-balanced, chemical equilibrium state of a given system is unique and globally
asymptotically stable if the model is implemented by SCK [59–61]. There fore, the system
relaxes to xeq by the control to stop all reactions whose indices are greater tha n N.
Owing to the global stability of the chemical equilibrium, any state can reach the chosen
vertex of the free cell C(σ ), xeq. The next step is to enter the inside of the free cell
from the vertex xeq. This is achieved by controlling all reactions in the system. Since
xeq can be globally asymptotically stable, we can construct a locally asymp totically stable
8 Recall that we supposed SCC matches to the whole space RN for the readability. If SCC is not the whole
space, the controllable set of x coincides to SCC, C(x) = W(x).
16

## Page 17

state inﬁnitesimally close to xeq and inside the free cell, by perturbatively incorporating the
contribution of the reactions that were stopped in the previous st ep. In this step, the feature
of the free cell—the conical hull matching the full linear span—plays a crucial role.
Once the state enters the free cell, it can be controlled to any stat e in the free cell,
particularly to xtgt as any pair of states inside the same free cell is mutually controllable.
In summary, one can control any source xsrc to xtgt ∈ C(σ ) in three steps (see Fig. 4):
1. Control the state from xsrc to xeq by utilizing the global stability of the chemical
equilibrium.
2. Control the state from xeq to x∗ by utilizing all the reactions.
3. Control the state from x∗ to xtgt by taking advantage of the free cell property.
All reactions in the Sel’kov model (Eq. (2)) are implemented by SCK, a nd thus, any state
in the free cell is controllable from any state. The free cell of the mo del is the triangle at
center ﬁlled with yellow in Fig. 2 (a cell labeled C(σ D)). Readers may see by following the
arrows depicted in the ﬁgure that any state in the free cell is certa inly reachable from any
other state in R2
≥ 0.
This theorem and Prop. S2 are the reason why we needed to make th e Sel’kov model
reversible (Eq.(2)). If all reactions are irreversible, the model ha s only a single cell. Since
the irreversible Sel’kov model has a ﬁxed point regardless of parame ter values, from Prop.. S2
this cell is a free cell. Therefore, an arbitrarily chosen pair of state s are trivially, mutually
controllable.
V. A TOY MODEL DEMONSTRA TION
Finally, we demonstrate global controllability using a toy model of met abolism. The
model is a reduced model of glycolysis consisting of three metabolites ( S, ATP, and ADP)
17

## Page 18

FIG. 4. A graphical description of the control procedure. Th e state xsrc is driven to the target state
xtgt via three steps. (Step 1) Control is applied to reach the equi librium state xeq by activating
the relevant control parameters (the others are set to zero) . (Step 2) The control is adjusted to u∗,
controlling the state to x∗. (Step 3) Finally, control within the free cell brings the st ate to xtgt.
The free cell is highlighted in yellow. The states xsrc, xeq, x∗, and xtgt are depicted by an open
circle, green square, blue diamond, and orange cross, respe ctively. Balance manifolds intersecting
at xeq (e.g., Mi and Mk) are shown in red, while the other balance manifold ( Mj) is depicted in
blue.
and the following three reactions 9:
R1 : n ATP ⇌ S + n ADP
R2 : S + (n + 1) ADP ⇌ (n + 1) ATP
R3 : ATP ⇌ ADP.
A schematic diagram is presented in Fig. 5. Here, chemical S is supplied from the external
environment of the system by utilizing n molecules of ATP, and it is secreted to the external
with converting ( n + 1) ADPs into ( n + 1) ATPs. Thus, through the reaction R1 and R2,
the system obtains one ATP from one S. The reaction R3 is the reaction for balancing
9 As stated, implementing the reaction kinetics by non-SCK is a necess ary condition for the model to have
a state in a free cell whose controllable set does not coincide with the whole space. However, it seemingly
not easy to construct such model with a small number of chemical s pecies. We have evaluated the
controllability of 2-variable models in literature: the Sel’kov model, Sch nakenberg model, and Brusselator
model. Even if we make the reaction kinetics non-SCK, these three m odels do not exhibit uncontrollability
to a state in free cells (SI text section VI). 18

## Page 19

ATP and ADP. This “investment and payoﬀ” architecture mirrors the design of gly colysis.
Speciﬁcally, in the pathway from glucose-6-phosphate to pyruvat e, one ATP molecule is
initially consumed to subsequently yield four ATP molecules. As ATP and ADP are just
interconverted by the reactions, the total amount of ATP and AD P is conserved, and SCC
is the level set of the total amount of ATP and ADP. Thus, we denot e the total amount of
ATP and ADP as A and study the reduced model with two variables, given by
d
dt

 xS
xA

 =

 1 − 1 0
− n n + 1 − 1







u1
u2
u3




 ⊙





xn
A − k1xS(A − xA)n
xS(A − xA)m − k2xm
A
xA − k3(A − xA)




 ,
where xS and xA are the concentrations of S and ATP, respectively, and the kinetic part
fi(x) is set to unity for every i. The reaction rate function of the reaction R2 becomes
non-SCK by setting m ̸= n + 1, whereas the other reactions are implemented by SCK. We
study the case in which n = 2 in the following.
We demonstrate that the control procedure argued above work s for m = n + 1 case, i.e.,
the SCK case. Our claim does not guarantee that the control proc edure fails in the case of
non-SCK; however, in this toy model, the procedure can indeed fail to control in the case
of m ̸= n + 1. In the following, we show the result with the total concentratio n of ATP
and ADP, A, equal to unity. We denote the balance manifold on SCC with A = 1 as Mr
although this is an abuse of the notation.
First, we present a case with SCK. Here, we computed the controlla bility to the target
state xtgt = (1 . 0, 0. 6) in the free cell C(σ ) with σ = (1 , 1, 1) (blue cell in Fig. 6(a)) where
the system steadily generates ATP through reactions R1 and R2. The free cell has two
vertices M1 ∩ M 3 and M2 ∩ M 3. While we chose M1 ∩ M 3 as the equilibrium state to be
controlled to (see Fig. 6(a)), the choice does not aﬀect the result . The number and position
of the vertices of the cells depend on the reaction order and rever sibility parameters of the
reactions. To avoid confusion, we explicitly denote the chosen stat e as the equilibrium state
of SCK model, xeq, SCK. Following the control procedure, we ﬁrst set the control param eter to
ueq, SCK = (1, 0, 1) to relax to the equilibrium state xeq, SCK. The second step is to control the
state inside the free cell by setting the control parameter to u∗. Finally, we set the control
parameter to the feedback control ufb which locally stabilizes xtgt. As shown in Fig. 6(b),
the relaxation to xeq, entrance to C(σ ), and control to xtgt are successfully realized in each
step.
19

## Page 20

Next, we show that the control fails depending on the initial state in the non-SCK case.
In the non-SCK case, we chose m = 1 though n = 2, and used the same target xtgt =
(1. 0, 0. 6) ∈ C(σ ) with σ = (1 , 1, 1). In this case, the free cell has only a single vertex
M1 ∩ M 2 and we denote it as xeq, nSCK (see Fig. 6(c)). As the model is non-SCK, the
relaxation to xeq, nSCK with an inactivation of the reactions, except reactions R1 and R2
is not guaranteed. Indeed, the system relaxes toward origin 0 instead of xeq, nSCK with
ueq, nSCK = (1, 1, 0) as shown in the gray trajectories in Fig. 6(d). While all trajector ies relax
toward the origin with ueq, nSCK, some trajectories pass through the free cell C(σ ) depending
on the initial states. Once the state enters the free cell, it can be c ontrolled to the target
state xtgt regardless of whether the reaction kinetics are SCK or not. The cy an-colored
trajectories in Fig. 6(d) are the trajectories that enter the fre e cell and are controlled to
xtgt.
The control failure in the non-SCK case is not due to the control pr ocedure adopted here.
We recently developed Stoichiometric Rays that gives an overestimate of the controllable
set with a given number of reaction direction ﬂips (the allowed count t hat the controlled
trajectory crosses the balance manifold) [47]. Using Stoichiometric Rays, we can evaluate
the controllability of the catalytic reaction system given in Eq. (1), f rom a given source
state to a target state with an arbitrary, non-negative control input u(t). The black dots in
Fig. 6(c) are the states judged as uncontrollable to the target st ate xtgt with the maximum
reaction ﬂips being six by Stoichiometric Rays.
The diﬀerence in the controllability in the two cases is intuitively unders tood from a geo-
metric viewpoint: In the SCK case (Fig. 6(a)), the conical hull of th e directed stoichiometric
vectors 10 in the cells around xeq, SCK, C(σ ), C (σ B), C (σ C), and C(σ D) are directed to the
xeq, SCK. On the other hand, in the non-SCK case (Fig. 6(c)), the cell with t he reaction di-
rection σ D does not exist, and instead, C(σ F ) is a neighbor of xeq, nSCK. The direction of the
conical hull in C(σ F ) is not directed to xeq, nSCK. This makes the approach to xeq, nSCK from
C(σ F ) impossible and leads to the uncontrollablity of the states in the regio n represented
by the black points in Fig. 6(c) to xeq, nSCK.
10 For the visibility’s sake, we display the stoichiometric vectors in the log arithm-converted space. Note that
the stoichiometric vectors are not straight arrows in the logarithm -transformed space.
20

## Page 21

FIG. 5. A schematic diagram of the toy model of metabolism. Th e substrate chemical S is taken
up from the external environment via the reaction R1 with consumption of n ATPs. The reaction
R2 generates (n + 1) ATP by secreting the chemical S, respectively. The react ion R3 is the reaction
for balancing ATP and ADP.
VI. DISCUSSION
In the present study, we proved the global controllability to the st ates in free cells for
catalytic models in which all catalysts are independently controllable a nd the reaction rate
functions are implemented using the stoichiometrically compatible kine tics (SCK). For such
a model, if the state is locally stabilizable, then the state is guarantee d to be in a free cell;
thus, it is always controllable from any state in the same stoichiometr ic compatibility class
(SCC). Taken together with the theorem in the conventional cont rol theory that globally
controllable states are locally stabilizable [62], local stabilizability and glo bal controllability
are equivalent in catalytic reaction systems.
By evaluating controllability, we can estimate the fundamental limits o f the system. If
a given in silico model cannot attain a given task regardless of the control, it is rea sonable
not to expect the task to be achieved in the real system as long as t he in silico model is
carefully constructed. Such overestimation of the fundamental limits of the systems may be
useful for a deeper understanding of these systems. For instan ce, in the biological context,
the fundamental limits of survival, adaptation, diﬀerentiation etc. can be estimated by the
controllability of biological systems. The present results suggest t hat the metabolic system
itself is not a limiting factor for the controllability of biological systems to homeostatically
exhibit desired functions, as long as the reactions are implemented b y SCK.
The notion of chemical systems out of equilibrium emphasizes that op en, energy-driven
21

## Page 22

FIG. 6. (a) The geometry of the cells in the phase space of SCK m odel. The blue shaded region is
the free cells with the reaction directions shown in the inse t. The directed stoichiometric vectors
are depicted for the non-free cells with the red, blue, and gr een arrow for the reaction R1, R2, and
R3, respectively. As eye guides, angles of vectors that can be r epresented by a conical combination
of the directed stoichiometric vectors are highlighted in y ellow. (b) The controlled trajectories of
the toy model with SCK ( n = 2 and m = 3). The control input is sequentially switched from
ueq, SCK to u∗ and then to ufb at t = 100 and t = 150, respectively. The trajectories are colored in
red if the state is in the target free cell, and otherwise are c olored in gray. (c). The geometry of
the cells in the phase space of non-SCK model. The cells with c olor shade are the free cells. The
reaction directions of the red free cell is illustrated in th e inset while the blue one is the same as
shown in (a). The black dots are uncontrollable states to the target state regardless of the control.
(d) The controlled trajectories of non-SCK ( n = 2 and m = 1) model. The control input is initially
ueq, nSCK. The trajectories are colored in cyan if the state is in the fr ee cell C(σ ). Once the state
enters the free cell, the state is controlled to the target st ate xtgt. Control parameters are set as
follows; ueq, SCK = (1 , 0, 1), ueq, nSCK = (1 , 1, 0), and u∗ ≈ (4. 67, 0. 05, 1). The detailed description
on the feedback control ufb is given in the SI text.22

## Page 23

reaction networks can exhibit dynamics far from equilibrium. System s chemistry aims to har-
ness this principle by constructing reaction networks that operat e under out-of-equilibrium
conditions and use network motifs as programmable modules [1, 3]. In peptide-based sys-
tems, dynamic features emerge from integrating multiple componen ts: balancing strong
folding against adaptivity and ﬂexibility is critical for generating supr amolecular order and
disorder [2]. Our ﬁnding that local stabilizability implies global controllab ility suggests that
if a synthetic reaction network can maintain a target state against small perturbations, it
can be steered to that state from any initial condition by manipulatin g the concentrations
or activities of the catalytic molecules. This insight could contribute t o the design of pro-
grammable chemical networks and help balance stability and adaptab ility in peptide-based
systems and other out-of-equilibrium chemical reaction networks .
However, the lesson from universal controllability—all chemical sta tes exhibiting home-
ostasis are globally controllable by modulations of catalytic activity—s hould be taken with
careful consideration of the prerequisites of control. The metab olic system is an example
of catalytic reaction systems. The universal controllability, being t aken at the face value,
counterintuitively indicates that cellular metabolism can be controlled to any state showing
homeostasis, even from an apparently “dead” state. Here, we dis cuss two aspects that we
believe to be essential on the universal controllability.
First, SCK is crucial for controllability. As mentioned, SCK is the reac tion kinetics
with a minimalistic extension of the detailed balance condition to non-eq uilibrium states
with the ideal-gas formalism. While non-SCK reaction kinetics can be de rived from SCK
kinetics by coarse-graining multistep enzymatic reactions, the der ivation needs a speciﬁc
combination of the elementary steps of the reactions (see SI text Sec. IV). However, a few
mathematical models of cellular metabolism adopt non-SCK reactions for the better ﬁts with
the experimental results [49, 63]. The non-SCK reaction kinetics wo uld be better attributed
to the limitation of the ideal-gas based framework rather than the s peciﬁc construction of the
elementary reaction steps. The limitations of the ideal-gas formalism for understanding the
biological system have often been discussed, and much eﬀort has b een made for developing
the chemical thermodynamics and chemical reaction network theo ry based upon the non
ideal-gas formalism [58, 64–68]. In this sense, the deviation of the int racellular chemical
reaction kinetics from the ideal-gas kinetics led by the molecular crow ding, liquid-liquid
23

## Page 24

phase separation, etc. would be one of the key factors of cellular c ontrollability.11
Another aspect is the autonomy of living systems. In the current s etup for control, several
unrealistic conditions were implicitly assumed, namely, perfect obser vability, arbitrariness
of the control dynamics, and absence of the control cost. Amon g them, the absence of
the control cost relates to disregarding the autonomy of living sys tems: Living systems
must produce enzymes to control metabolic ﬂuxes by themselves. The production of these
enzymes requires energy and building blocks of proteins, such as AT P and amino acids, and
such resources should be produced by metabolic reaction systems .
This intrinsic requirement highlights a fundamental distinction betwe en engineered con-
trol systems and biological organisms: while the former typically rely on external agents
to set goals and manipulate parameters, the latter are character ized by their capacity
for self-determination and self-maintenance [69]. In the autonomy perspective, such self-
determination is grounded in organisational closure, i.e., the mutual dependence of the sys-
tem’s components and operations for their production and mainten ance, which collectively
determine the conditions under which the system can exist [70]. Regu lation then appears
as “control from within”: it is exerted by dedicated subsystems th at are produced by the
organism and materially supported by its ongoing self-maintenance, yet are dynamically
decoupled from the processes they modulate, enabling them to sele ctively adjust internal
dynamics in response to speciﬁc perturbations [71]. Accordingly, ou r control-theoretic re-
sults should be read as a ﬁrst step that characterises controllabilit y properties under idealised
external actuation; extending them toward genuine biological aut onomy will require endo-
genising the controller and its material/energetic constraints, and relating control objectives
to viability norms generated by the organisation itself.
Some of the authors recently proposed a mathematical framewor k of “death” [47]. In the
framework, the dead state is deﬁned as the state that is not reac hable to the representative
living states which are the reference states of “living”. Na ¨ ıvely, living states ha ve home-
ostasis, and otherwise we cannot observe “living” states because organisms are under the
continuous exposure of the ﬂuctuations. Therefore, the repre sentative living states should
have local stabilizability by control, and thus, they should be chosen from free cells. Ac-
cording to the present results, all states can be controlled to the representative living states
11 The global asymptotic stability of the detailed-balanced state is cru cial for the whole proof as seen in the
SI text. The universal controllability may hold even for the cases wit h non-SCK kinetics as long as the
uniqueness and global asymptotic stability of the detailed-balanced state is guaranteed.
24

## Page 25

indicating that death is impossible in metabolic models with the current c ontrollability
setup.
For the mathematical understanding of cell deaths, the above tw o points would be es-
sential, namely, the non ideal-gas based framework of metabolism an d theories focusing on
the autonomous nature of living systems: The machineries for cont rolling the system are
produced by the system to be controlled.
ACKNOWLEDGMENTS
This work was supported by JSPS KAKENHI (Grant Numbers JP22H05403 and JP25H01390
to Y.H.; JP24K00542 and JP25H01365 to T.J.K.; JP24KJ0090 to S.A.H), JST (Grant Num-
ber JPMJCR25Q2 to T.J.K.), and Joint Research of the Exploratory R esearch Center on
Life and Living Systems (ExCELLS) (ExCELLS program No 25EXC603 -2 to YH). This
research was partially conducted while visiting the Okinawa Institute of Science and Tech-
nology (OIST) through the Theoretical Sciences Visiting Program ( TSVP).
[1] Albert S Y Wong and Wilhelm T S Huck. Grip on complexity in c hemical reaction networks.
Beilstein J. Org. Chem. , 13(1):1486–1497, July 2017.
[2] Fahmeed Sheehan, Deborah Sementa, Ankit Jain, Mohit Kum ar, Mona Tayarani-Najjaran,
Daniela Kroiss, and Rein V Ulijn. Peptide-based supramolec ular systems chemistry. Chem.
Rev., 121(22):13869–13914, November 2021.
[3] Jan H van Esch, Rafal Klajn, and Sijbren Otto. Chemical sy stems out of equilibrium. Chem.
Soc. Rev., 46(18):5474–5475, September 2017.
[4] Norbert Wiener. Cybernetics. Scientiﬁc American , 179(5):14–19, 1948.
[5] N Barkai and S Leibler. Robustness in simple biochemical networks. Nature, 387(6636):913–
917, June 1997.
[6] T. M. Yi, Y. Huang, M. I. Simon, and J. Doyle. Robust perfec t adaptation in bacterial
chemotaxis through integral feedback control. Proceedings of the National Academy of Sci-
ences, 97(9):4649–4653, 2000.
[7] Hiroaki Kitano. Systems biology: a brief overview. Science, 295(5560):1662–1664, 2002.
25

## Page 26

[8] Michael B. Elowitz and Stanislas Leibler. A synthetic os cillatory network of transcriptional
regulators. Nature, 403(6767):335–338, 2000.
[9] Timothy S. Gardner, Charles R. Cantor, and James J. Colli ns. Construction of a genetic
toggle switch in escherichia coli. Nature, 403(6767):339–342, 2000.
[10] Attila Becskei and Luis Serrano. Engineering stabilit y in gene networks by autoregulation.
Nature, 405(6786):590–593, 2000.
[11] Eileen Fung, Wilson W. Wong, Jason K. Suen, Thomas Bulte r, Sun-Gu Lee, and James C.
Liao. A synthetic gene-metabolic oscillator. Nature, 435(7038):118–122, 2005.
[12] Corentin Briat, Ankit Gupta, and Mustafa Khammash. Ant ithetic integral feedback ensures
robust perfect adaptation in noisy biomolecular networks. Cell Systems , 2(1):15–26, 2016.
[13] Hsin-Ho Huang, Yili Qian, and Domitilla Del Vecchio. A q uasi-integral controller for adap-
tation of genetic modules to variable ribosome demand. Nature communications, 9(1):5415,
2018.
[14] Timothy Frei, Chia-Hung Chang, Milos Filo, Georgios Ar ampatzis, and Mustafa Khammash.
A genetic mammalian proportional-integral feedback contr ol circuit for robust and precise
gene regulation. Proceedings of the National Academy of Sciences USA , 119(24):e2122132119,
2022.
[15] Javier Santos-Moreno, Eve Tasiudi, J¨ org Stelling, an d Yolanda Schaerli. Multistable and dy-
namic crispri-based synthetic circuits for reliable biolo gical pattern formation. Nature Com-
munications, 11(1):2746, 2020.
[16] Lingchong You, Robert S. Cox, Ron Weiss, and Frances H. A rnold. Programmed population
control by cell-cell communication and regulated killing. Nature, 428(6985):868–871, 2004.
[17] Hideki Kobayashi, Mads Kaern, Michihiro Araki, Kristy Chung, Timothy S. Gardner,
Charles R. Cantor, and James J. Collins. Programmable cells : interfacing natural and engi-
neered gene networks. Proc. Natl. Acad. Sci. USA , 101(22):8414–8419, 2004.
[18] Subhayu Basu, Yaron Gerchman, Cynthia H. Collins, Fran ces H. Arnold, and Ron Weiss. A
synthetic multicellular system for programmed pattern for mation. Nature, 434(7037):1130–
1134, 2005.
[19] Frederick K. Balagadd´ e, Hao Song, Jun Ozaki, Cynthia H . Collins, Matthew Barnet,
Frances H. Arnold, Stephen R. Quake, and Lingchong You. A syn thetic escherichia coli
predator–prey ecosystem. Molecular Systems Biology , 4(1):187, 2008.
26

## Page 27

[20] Yang-Yu Liu, Jean-Jacques Slotine, and Albert-L´ aszl ´ o Barab´ asi. Controllability of complex
networks. Nature, 473(7346):167–173, 2011.
[21] Leonardo Morsut, Kole T. Roybal, Xin Xiong, Russell M. G ordley, Scott M. Coyle, Matthew
Thomson, and Wendell A. Lim. Engineering customized cell se nsing and response behaviors
using synthetic notch receptors. Cell, 164(4):780–791, 2016.
[22] Satoshi Toda, Lucas R. Blauch, Sindy K. Y. Tang, Leonard o Morsut, and Wendell A. Lim.
Programming self-organizing multicellular structures wi th synthetic cell–cell signaling. Sci-
ence, 361(6398):156–162, 2018.
[23] H. Kacser and J. A. Burns. The control of ﬂux. In Symp. Soc. Exp. Biol. , volume 27, pages
65–104, 1973.
[24] Reinhart Heinrich and Tom A. Rapoport. A linear steady- state treatment of enzymatic chains.
general properties, control and eﬀector strength. Eur. J. Biochem. , 42(1):89–95, 1974.
[25] Oluwaseun Sharomi and Tufail Malik. Optimal control in epidemiology. Ann. Oper. Res. ,
251(1-2):55–71, 2017.
[26] Simon K Schnyder, John J Molina, Ryoichi Yamamoto, and M atthew S Turner. Understanding
nash epidemics. Proc. Natl. Acad. Sci. U. S. A. , 122(9):e2409362122, March 2025.
[27] R Wolfenden and M J Snider. The depth of chemical time and the power of enzymes as
catalysts. Acc. Chem. Res. , 34(12):938–945, December 2001.
[28] Gonen Ashkenasy, Thomas M Hermans, Sijbren Otto, and An nette F Taylor. Systems chem-
istry. Chem. Soc. Rev. , 46(9):2543–2554, May 2017.
[29] Thomas Beneyton, Dorothee Kraﬀt, Claudia Bednarz, Chri stin Kleineberg, Christian Woelfer,
Ivan Ivanov, Tanja Vidakovi´ c-Koch, Kai Sundmacher, and Je an-Christophe Baret. Out-of-
equilibrium microcompartments for the bottom-up integrat ion of metabolic functions. Nat.
Commun., 9(1):2391, June 2018.
[30] Takashi Okada and Atsushi Mochizuki. Law of localizati on in chemical reaction networks.
Physical review letters , 117(4):048101, 2016.
[31] Yuji Hirono, Takashi Okada, Hiroyasu Miyazaki, and Yos himasa Hidaka. Structural reduction
of chemical reaction networks based on topology. Physical Review Research, 3(4):043123, 2021.
[32] Yuji Hirono, Ankit Gupta, and Mustafa Khammash. Rethin king robust adaptation: Charac-
terization of structural mechanisms for biochemical netwo rk robustness through topological
invariants. PRX Life , 3(1):013017, March 2025.
27

## Page 28

[33] Rachel M Walker, Valeria C Sanabria, and Hyun Youk. Micr obial life in slow and stopped
lanes. Trends Microbiol., December 2023.
[34] Diederik S Laman Trip, Th´ eo Maire, and Hyun Youk. Slowe st possible replicative life at frigid
temperatures for yeast. Nat. Commun. , 13(1):7518, December 2022.
[35] Tori M Hoehler and Bo Barker Jørgensen. Microbial life u nder extreme energy limitation.
Nat. Rev. Microbiol. , 11(2):83–94, February 2013.
[36] Declan A Gray, Gaurav Dugar, Pamela Gamba, Henrik Strah l, Martijs J Jonker, and Leen-
dert W Hamoen. Extreme slow growth as alternative strategy t o survive deep starvation in
bacteria. Nat. Commun. , 10(1):1–12, February 2019.
[37] Ksenija Zahradka, Dea Slade, Adriana Bailone, Suzanne Sommer, Dietrich Averbeck, Mirjana
Petranovic, Ariel B Lindner, and Miroslav Radman. Reassemb ly of shattered chromosomes
in deinococcus radiodurans. Nature, 443(7111):569–573, 2006.
[38] Takuma Hashimoto, Daiki D Horikawa, Yuki Saito, Hiroka zu Kuwahara, Hiroko Kozuka-
Hata, Tadasu Shin-i, Yohei Minakuchi, Kazuko Ohishi, Ayuko Motoyama, Tomoyuki Aizu,
et al. Extremotolerant tardigrade genome and improved radi otolerance of human cultured
cells by tardigrade-unique protein. Nature communications, 7(1):12808, 2016.
[39] Kaito Kikuchi, Leticia Galera-Laporta, Colleen Weath erwax, Jamie Y Lam, Eun Chae Moon,
Emmanuel A Theodorakis, Jordi Garcia-Ojalvo, and G¨ urol M S¨ uel. Electrochemical potential
enables dormant spores to integrate environmental signals . Science, 378(6615):43–49, October
2022.
[40] Martin Feinberg. Foundations of Chemical Reaction Network Theory , volume 202 of Applied
Mathematical Sciences. Springer, Cham, Switzerland, 2019.
[41] Peter William Atkins, Julio De Paula, and James Keeler. Atkins’ physical chemistry. Oxford
university press, 2023.
[42] Wassim M Haddad, VijaySekhar Chellaboina, and Qing Hui . Nonnegative and compartmental
dynamical systems . Princeton University Press, 2010.
[43] Stephen H Saperstone. Global controllability of linea r systems with positive controls. SIAM
J. Control Optim. , 11(3):417–423, August 1973.
[44] Gyula Farkas. Local controllability of reactions. J. Math. Chem. , 24(1):1–14, August 1998.
[45] D Dochain and L Chen. Local observability and controlla bility of stirred tank reactors. J.
Process Control, 2(3):139–144, January 1992.
28

## Page 29

[46] D´ aniel Andr´ as Drexler and J´ anos T´ oth. Global controllability of chemical reactions. J. Math.
Chem., 54(6):1327–1350, June 2016.
[47] Yusuke Himeoka, Shuhei A Horiguchi, and Tetsuya J Kobay ashi. Theoretical basis for cell
deaths. Phys. Rev. Res. , 6(4):043217, November 2024.
[48] Ali Khodayari, Ali R Zomorrodi, James C Liao, and Costas D Maranas. A kinetic model of
escherichia coli core metabolism satisfying multiple sets of mutant ﬂux data. Metab. Eng. ,
25:50–62, September 2014.
[49] Simon Boecker, Giulia Slaviero, Thorben Schramm, Wito ld Szymanski, Ralf Steuer, Hannes
Link, and Steﬀen Klamt. Deciphering the physiological respo nse of escherichia coli under high
ATP demand. Mol. Syst. Biol. , 17(12):e10504, December 2021.
[50] Christophe Chassagnole, Naruemol Noisommit-Rizzi, J oachim W Schmid, Klaus Mauch, and
Matthias Reuss. Dynamic modeling of the central carbon meta bolism of escherichia coli.
Biotechnol. Bioeng., 79(1):53–73, July 2002.
[51] Y Himeoka and N Mitarai. Emergence of growth and dormanc y from a kinetic model of the
escherichia coli central carbon metabolism. Physical Review Research, 2022.
[52] Yusuke Himeoka and Chikara Furusawa. Perturbation-re sponse analysis of in silico metabolic
dynamics in nonlinear regime: Hard-coded responsiveness i n the cofactors and network spar-
sity. eLife, June 2024.
[53] Evgeny Evgenievich SEL’KOV. Self-oscillations in gly colysis 1. a simple kinetic model. Eu-
ropean Journal of Biochemistry , 4(1):79–86, 1968.
[54] D F Anderson. A proof of the global attractor conjecture in the single linkage class case.
SIAM Journal on Applied Mathematics , 2011.
[55] Henk Nijmeijer and Arjan Van der Schaft. Nonlinear dynamical control systems , volume 175.
Springer, 1990.
[56] M Schauer and R Heinrich. Quasi-steady-state approxim ation in the mathematical modeling
of biochemical reaction networks. Math. Biosci. , 65(2):155–170, August 1983.
[57] A. Cornish-Bowden. Fundamentals of Enzyme Kinetics . Wiley, 2013.
[58] Francesco Avanzini, Emanuele Penocchio, Gianmaria Fa lasco, and Massimiliano Esposito.
Nonequilibrium thermodynamics of non-ideal chemical reac tion networks. J. Chem. Phys. ,
154(9):094114, March 2021.
29

## Page 30

[59] Riccardo Rao and Massimiliano Esposito. Nonequilibri um thermodynamics of chemical reac-
tion networks: Wisdom from stochastic thermodynamics. Phys. Rev. X , 6(4):041064, Decem-
ber 2016.
[60] Stefan Schuster and Ronny Schuster. A generalization o f wegscheider’s condition. implications
for properties of steady states and for quasi-steady-state approximation. J. Math. Chem. ,
3(1):25–42, January 1989.
[61] Gheorghe Craciun. Toric diﬀerential inclusions and a pr oof of the global attractor conjecture.
arXiv [math.DS] , January 2015.
[62] F H Clarke, Y S Ledyaev, E D Sontag, and A I Subbotin. Asymp totic controllability implies
feedback stabilization. IEEE Trans. Automat. Contr. , 42(10):1394–1407, 1997.
[63] Zane R Thornburg, David M Bianchi, Troy A Brier, Benjami n R Gilbert, Tyler M Earnest,
Marcelo C R Melo, Nataliya Safronova, James P S´ aenz, Andr´ as T Cook, Kim S Wise, Clyde A
Hutchison, 3rd, Hamilton O Smith, John I Glass, and Zaida Lut hey-Schulten. Fundamental
behaviors emerge from simulations of a living minimal cell. Cell, 185(2):345–360.e28, January
2022.
[64] Stefan M¨ uller and Georg Regensburger. Generalized ma ss-action systems and positive solu-
tions of polynomial equations with real and symbolic expone nts (invited talk). In Computer
Algebra in Scientiﬁc Computing , Lecture notes in computer science, pages 302–323. Springe r
International Publishing, Cham, 2014.
[65] Artur Wachtel, Riccardo Rao, and Massimiliano Esposit o. Thermodynamically consistent
coarse graining of biocatalysts beyond Michaelis–Menten. New J. Phys. , 20(4):042002, April
2018.
[66] F Horn and R Jackson. General mass action kinetics. Arch. Ration. Mech. Anal., 47(2):81–116,
January 1972.
[67] Yuki Sughiyama, Dimitri Loutchko, Atsushi Kamimura, a nd Tetsuya J Kobayashi. Hessian
geometric structure of chemical thermodynamic systems wit h stoichiometric constraints. Phys.
Rev. Res., 4(3):033065, July 2022.
[68] Tetsuya J Kobayashi, Dimitri Loutchko, Atsushi Kamimu ra, Shuhei A Horiguchi, and Yuki
Sughiyama. Information geometry of dynamics on graphs and h ypergraphs. Information
Geometry, 7(1):97–166, June 2024.
30

## Page 31

[69] Kepa Ruiz-Mirazo, Juli Peret´ o, and Alvaro Moreno. A un iversal deﬁnition of life: autonomy
and open-ended evolution. Orig. Life Evol. Biosph. , 34(3):323–346, June 2004.
[70] Alvaro Moreno and Matteo Mossio. Biological autonomy: A philosophical and theoretical en-
quiry. History, Philosophy and Theory of the Life Sciences. Sprin ger, Dordrecht, Netherlands,
2015 edition, May 2015.
[71] Leonardo Bich, Matteo Mossio, Kepa Ruiz-Mirazo, and Al varo Moreno. Biological regulation:
controlling the system from within. Biol. Philos. , 31(2):237–265, March 2016.
[72] King-Wah Eric Chu. Generalization of the bauer-ﬁke the orem. Numer. Math. (Heidelb.) ,
49(6):685–691, November 1986.
[73] Friedrich L Bauer and Charles T Fike. Norms and exclusio n theorems. Numerische Mathe-
matik, 2(1):137–141, 1960.
[74] Alexander Schrijver. Theory of linear and integer programming . John Wiley & Sons, 1998.
31

## Page 32

SUPPLEMENT AR Y INFORMA TION
]yhimeoka@g.ecc.u-tokyo.ac.jp
I. PREP ARA TION
In the SI text, we consider the system given by
dx
dt = Su(t) ⊙ f (x(t)) ⊙ p(x(t)), (S1)
where u is the control input and f and p are the kinetic- and thermodynamic parts of the
reaction rate function, respectively (see main text). In the follow ing, if a given vector a has
all positive (non-negative) elements, we denote a ≻ 0, (a ⪰ 0). We allocate I to the index
set of reactions.
We often work on the following system
dx
dt = Su(t) ⊙ p(x(t)), (S2)
where the contribution of the kinetic part f (x) is canceled by the control parameter u(t).
Note that we assumed f (x) ≻ 0 for x ∈ RN
>0; therefore the cancellation does not violate the
non-negativity constraint of the control parameter.
For the readers’ convenience, we recapitulate the deﬁnition of se veral concepts. The
stoichiometric compatibility class is the parallel translation of Im S,
W(x) := {x +
R∑
i=1
aiSi |ai ∈ R} ∩ RN
>0. (S3)
In the present text, we assume dim W(x) > 0 since no control is allowed in dim W(x) = 0
case.
The balance manifold is deﬁned as the zero locus of pr(x), and the phase space is parti-
tioned by the balance manifolds {Mr}r∈ I into cells C(σ ) where the reaction directions are
ﬁxed and represented by σ .
Mr := {x ∈ RN
>0 | pr(x) = 0 }, (S4)
C(σ ) := {x ∈ RN
>0 | sgn p(x) = σ }. (S5)
32

## Page 33

The controllable set C(x) is a set of states controllable to x: If x0 is in C(x), there is a
solution of Eq. (S1) with the initial condition x0 reaching x.
In the SI text, we use steady-state generically to refer to states satisfying dx/dt = 0, while
the equilibrium state is for the steady-state with vanishing reaction ﬂux for the reactio ns
with non-zero control, that is, the steady-state with vr(x) = 0 for ur > 0. Note that the
terms do not contain stability information. The stability of the state s will be explicitly
stated if necessary.
In the present text, we provide proofs for the following claims.
Proposition S1 (Stabilizability of states in free cells) . For any state x∗ ∈ C(σ ), there
exists a control that locally asymptotically stabilizes x∗ within C(σ ) ∩ W (x∗) if and only if
C(σ ) is a free cell.
Proposition S2 (Steady-states and freeness) . A cell C(σ ) is a free cell if and only if it
admits a steady-state x∗ ∈ C(σ ) under a constant control u supported on the active reactions
(i.e., ∃u ⪰ 0 such that S(u ⊙ v(x∗)) = 0 with ui > 0 for all i where σi ̸= 0).
Here, we exclude steady-states made by setting all the control in puts to zero u = 0. Such
marginally stable steady-states can exist in any cell, but they do not admit non-zero ﬂux.
The formal deﬁnition of the stoichiometrically compatible kinetics (SC K) is as follows.
Deﬁnition S1 (Stoichiometrically compatible kinetics) . The reaction rate function vr(x) is
stoichiometrically compatible kinetics if it can be decomp osed as
vr(x) = fr(x)pr(x), (S6)
fr(x) > 0, (S7)
pr(x) =
N∏
i=1
x
n+
i,r
i − kr
N∏
i=1
x
n−
i,r
i , (k ≥ 0) (S8)
and in addition, the n±
i,r satisfy
Si,r = n−
i,r − n+
i,r
where Si,r is the stoichiometric coeﬃcient of the ith chemical in the rth reaction and kr > 0
is the reversibility parameter of the rth reaction.
For the statement of the main theorem, boundaries of cells play a cr ucial role. We will
show in Sec. III that for any cell there is a set of balance manifolds t hat are the boundaries
33

## Page 34

of the cell. Having established the existence of the boundary, we de ﬁne the full SCK subset
as follows:
Deﬁnition S2 (Full SCK subset). Let J ′ be the index set of the balance manifolds {Mj}j∈ J ′
which are the boundaries of the cell C(σ ). If there is an index set J ⊆ J ′ such that |J| =
rank S and all reactions in J are implemented by SCK, the cell C(σ ) is said to have a full
SCK subset within its boundaries.
Here, we state the main theorem in the case with σ ∈ {− 1, 1}R because the generic claim
for σ ∈ {− 1, 0, 1}R is technical and may obscure the main argument. The extension of t he
main theorem to the case with σ ∈ {− 1, 0, 1}R is straightforward, and the statement and
proof are provided after the proof of the main theorem.
Theorem S1 (Main theorem) . If a free cell C(σ ) has a full SCK subset within its bound-
aries, then the controllable set of any state in the free cell x ∈ C(σ ) coincides with the entire
stoichiometric compatibility class, C(x) = W(x).
Corollary S1. If all reactions in the model are implemented by SCK, then eve ry state in
any free cell is controllable from an arbitrary state in the s toichiometric compatibility class
of the corresponding state.
Corollary S2. For a system with reactions implemented only by SCK, if state x is locally
stabilizable, then the controllable set of x is the entire stoichiometric compatibility class,
C(x) = W(x).
The Corollary. S1 is a direct consequence of Theorem. S1, and Coro llary. S2 is derived
from the Prop. S1 and Cor. S1. We provide proofs for propositions S1, S2, and theorem S1.
Note that we have argued Cor. S1 in the main text for readability. Co r. S1 requires all
the reactions in the model to be implemented by the stoichiometrically compatible kinetics
(SCK), while the main theorem (Theorem S1) states a generic argum ent. For Theorem S1
to hold, the reactions to be required to be implemented by SCK are on ly those that the
balance manifolds are chosen as a set of boundaries of the free cell. SCK is not required for
the other reactions.
We introduce a useful concept and lemma to prove the claims.
34

## Page 35

Deﬁnition S3 (Positive dependence [40]) . The set of vectors {v}i∈ I is positively dependent
if there exist constants a ≻ 0 such that
∑
i∈ I
aivi = 0.
Recall that the free cell is deﬁned as the cell in which the conical com binations of directed
stoichiometric vectors span the entire cell.
Deﬁnition S4 (Free cells). A cell C(σ ) is termed a free cell if the set of conical combinations
of its directed stoichiometric vectors equals their full li near span:
cone{σiSi}i∈ I = span{σiSi}i∈ I.
Lemma S1. The following two statements are equivalent;
• The vectors {vi}i∈ I are positively dependent.
• cone{vi}i∈ I = span{vi}i∈ I.
Proof. First, we show that cone{vi}i∈ I = span{vi}i∈ I holds if {vi}i∈ I is positively dependent.
As cone {vi}i∈ I ⊆ span{vi}i∈ I is trivial, we show that cone {vi}i∈ I ⊇ span{vi}i∈ I.
For any vector v ∈ span{vi}i∈ I, we have bi ∈ R, (i ∈ I) such that
v =
∑
i∈ I
bivi, b i ∈ R (S9)
holds. From the positive dependence of the vectors, we can const ruct the zero vector using
the positive linear combination as 0 = ∑
i∈ I aivi, (a ≻ 0). By adding γ0 to Eq. (S9), we
obtain the following
v + γ0 = v =
∑
i∈ I
(bi + γai)vi. (S10)
Because a ≻ 0 holds, by setting γ suﬃciently large, we have b + γa ≻ 0. Thus, v ∈
cone{vi}i∈ I holds.
Next, we show that {vi}i∈ I is positively dependent if cone {vi}i∈ I = span{vi}i∈ I holds.
Since for any ρ ∈ I, − vρ ∈ span{vi}i∈ I = cone{vi}i∈ I holds, we have that
− vρ =
∑
i∈ I
a(ρ)
i vi, (a(ρ)
i ≥ 0), (S11)
35

## Page 36

and thus,
0 =
∑
i∈ I
¯a(ρ)
i vi, (S12)
where ¯a(ρ)
i = a(ρ)
i + 1 for ρ = i and otherwise ¯a(ρ)
i = a(ρ)
i . The sum of Eq. (S12) over every
ρ ∈ I leads to
0 =
∑
ρ∈ I
∑
i∈ I
¯a(ρ)
i vi
=
∑
i∈ I
(
1 +
∑
ρ∈ I
a(ρ)
i
)
vi.
As 1 + ∑
ρ∈ I a(ρ)
i are positive for any ρ ∈ I, the vectors {vi}i∈ I are positively dependent.
II. THE PROOF OF PROP . S1 AND PROP . S2
Proposition S1 (Stabilizability of states in free cells) . For any state x∗ ∈ C(σ ), there
exists a control that locally asymptotically stabilizes x∗ within C(σ ) ∩ W (x∗) if and only if
C(σ ) is a free cell.
To show this, we use the following lemma.
Lemma S2. If the system is in a free cell, for any control without non-ne gativity constraint
w(t) : R → RR, there exists a constrained control u(t) : R → RR
≥ 0 that realizes the same
control as w(t).
Proof. Let us consider the system
dx
dt = Sw(t) ⊙ p(x),
where w(t) : R → RR is a non-constrained control. Because the state is in the free cell, f or
any stoichiometric vector Si, there exists a positive linear combination of the form
− σρSρ =
∑
i̸=ρ
a(ρ)
i σiSi, (a(ρ)
i > 0), (S13)
Since the combinations are generally not unique, we choose one arbit rarily and adopt its
expansion coeﬃcients; without loss of generality, we may set a(ρ)
ρ = 0.
Because x ∈ C(σ ), we have |p(x)| ≻ 0. Thus, we absorb the absolute value of pi into wi
and consider
˙x = S σ ⊙ w(t) (S14)
36

## Page 37

(For the equation that uses u, we similarly absorb the corresponding factor). Using the step
function
Θ( w) =





1 ( w > 0)
0 ( w < 0),
we can rewrite the right-hand side of Eq. (S14) as follows:
R∑
i=1
wi σiSi =
R∑
i=1
Θ( wi) wi σi Si +
R∑
i=1
Θ( − wi) wi σi Si, (S15)
=
R∑
i=1
Θ( wi) wi σi Si +
R∑
i=1
Θ( − wi) |wi|(− σi Si), (S16)
=
R∑
i=1
Θ( wi) wi σi Si +
R∑
i=1
Θ( − wi) |wi|
( R∑
j=1
a(i)
j σj Sj
)
, (S17)
=
R∑
i=1
[
Θ( wi)wi +
R∑
j=1
Θ( − wj)|wj|a(j)
i
]
σi Si. (S18)
Since the expression inside the brackets is non-negative, we can ch oose this to be our u.
The mapping from w to u is smooth for wi except wi = 0.
We now prove Proposition S1.
Proof. Consider the diﬀerential equation inside the free cell
dx
dt = S σ ⊙ w(t), (S19)
where the absolute value of p(x(t)) is lumped into w(t) as x ∈ C(σ ) and |p(x)| ≻ 0 holds.
Since we are considering the dynamics inside the free cell, w may have either positive or
negative values (Lem. S2). Now, we wish to design dynamics that sta bilize a state x∗ ∈ C(σ ).
Without a loss of generality, we set σi = 1 for σi ̸= 0 by utilizing the arbitrariness of the
reaction direction.
To this end, we consider a function given by
Φ( x, x∗) = 1
2
R∑
r=1
σr
[ N∑
i=1
Sir(xi − x∗
i ) − w(0)
r
]2
,
where w(0)
r is selected such that S σ ⊙ w(0) = 0. As shown below, w(0) corresponds to the
steady-state reaction ﬂux of the system. If the zero vector is c hosen as w(0), all reactions in
37

## Page 38

the system are halted at x∗, whereas the system has a non-zero steady ﬂux at x∗ if we have
w(0) ̸= 0.
For this Φ, we have
∂Φ
∂xi
=
∑
j,r
σrSirSjr(xj − x∗
j ) −
∑
r
σrSirw(0)
r (S20)
=
∑
j,r
σrSirSjr(xj − x∗
j ). (S21)
The rank of matrix S S⊤ is reduced by the dimension of coker S. Therefore, while the states
that minimize Φ are arbitrary in the cokernel directions, once the SC C is speciﬁed, the
intersection of the SCC with ∇ Φ = 0 is unique. In particular, if we choose the SCC to
which x∗ belongs, x∗ is the unique state that minimizes Φ in SCC.
On the other hand, we can rewrite Eq. (S21) as
∂Φ
∂xi
=
∑
r
σrSir
[∑
j
Sjr (xj − x∗
j ) − w(0)
r
]
(S22)
Thus, by designing the feedback control as
wr(t) = wr(x(t)) = −
(
Sr ·(x − x∗) − w(0)
r
)
, (S23)
we obtain Eq. (S19). Note that the feedback w(t) given by Eq.(S23) can be negative-valued.
However, since we are considering only the trajectories in the free cell, the control w(t) is
always realized by a non-negative control u(t) (Lem.S2).
Hence, Φ is a potential function satisfying
dx
dt = −∇ Φ( x, x∗). (S24)
Thus, the dynamics are potential dynamics and relax to the potent ial minima x∗. Note that
this feedback control is feasible inside C(σ ), and the gradient dynamics (Eq. (S24)) does
not exclude the possibility that the trajectory in Eq. (S24) reache s the boundary of the free
cell C(σ ). Thus, the basin of attraction for x∗ is not necessarily as large as C(σ ) ∩ W (x∗).
However, the basin of attraction has a non-zero volume because t he distance between x∗
and the balance manifolds of the reaction with non-zero σi is larger than zero as x∗ ∈ C(σ ).
The impossibility of stabilizing the system at a state in a non-free cell is straightforward.
For this purpose, we deﬁne the stoichiometric cone with strictly pos itive coeﬃcients as
V +
σ (x) := {x + ∑
i∈ I aiσiSi | ai > 0}. Suppose that C(σ ) is a non-free cell. We take
38

## Page 39

an arbitrary state x∗ ∈ C(σ ), and consider the ǫ-neighborhood of x∗ constrained on the
stoichiometric compatibility class,
Nǫ(x∗) := {x ∈ W (x∗) | ∥x − x∗∥ < ǫ}
so that Nǫ(x∗) ⊂ C(σ ). Since the cell is not free, there exists x′ ∈ Nǫ(x∗) such that
x∗ /∈ V σ (x′).
Concretely, for an arbitrary state x′ ∈ V +
σ (x∗) ∩ C(σ ), x∗ /∈ V σ (x′) holds, i.e., if the state
changes by occasional activations of the all reactions, the syste m is no longer able to return
to the original state. To show this, we prove if any perturbation wit hin a given cell can be
counteracted by a control, the cell should be the free cell. The con traposition corresponds
to the statement that we want to show.
Suppose that the system is originally at the state x∗ and transits to x′ = x∗ +∑
i∈ I aiσiSi
by a perturbation. Since we suppose that any perturbation is coun teracted by a control,
there is a control ∑
i∈ I biσiSi (b ⪰ 0) which drag the system back to the original state x∗.
Then, the following holds:
0 =
∑
i∈ I
aiσiSi +
∑
i∈ I
biσiSi =
∑
i∈ I
(ai + bi)σiSi.
Here, we consider the perturbation with a ≻ 0. a ≻ 0 and b ⪰ 0 lead to a + b ≻ 0, and
thus, the vectors {σiSi}i∈ I are positively dependent. From Lem. S1 and the deﬁnition of
the free cell (Def. S4), cell C(σ ) becomes the free cell.
It is noteworthy that the only way to realize ˙x = 0 in non-free cells is to set the control ur
to zero for r with pr(x) ̸= 0 because the stoichiometric vectors are not positively dependen t.
Thus, the steady-state is not a dynamic equilibrium state in non-fre e cells, but rather a state
in which the reactions are forced to stop by setting the control pa rameters to zero. This
marginality is related to the system’s lack of resilience to perturbatio ns.
Next, we prove the proposition. S2.
Proposition S2 (Steady-states and freeness) . A cell C(σ ) is a free cell if and only if it
admits a steady-state x∗ ∈ C(σ ) under a constant control u supported on the active reactions
(i.e., ∃u ⪰ 0 such that S(u ⊙ v(x∗)) = 0 with ui > 0 for all i where σi ̸= 0).
39

## Page 40

Proof. Let J be the index set with non-zero σ, that is, J := {j ∈ I | σj ̸= 0 }. The
steady-state condition is given by
0 =
∑
j∈ J
σjujvj(x∗)Sj =
∑
j∈ J
α jσjSj,
where α j = ujvj(x∗). Since uj > 0 and vj(x∗) > 0 hold, α j is positive for j ∈ J, and thus,
the vectors {σjSj}j∈ J are positively dependent. From Lem. S1, the cell C(σ ) is a free cell.
Conversely, suppose that C(σ ) is a free cell. Let J := {j ∈ I | σj ̸= 0 }. According
to Lem. S1, the set of directed stoichiometric vectors {σjSj}j∈ J is positively dependent.
Therefore, there exist positive coeﬃcients α j > 0 for all j ∈ J such that
∑
j∈ J
α jσjSj = 0. (S25)
We choose an arbitrary state x∗ ∈ C(σ ). Since x∗ is in the cell C(σ ), the reaction rate
satisﬁes sgn ( vj(x∗)) = σj, and thus vj(x∗) = σj|vj(x∗)| with |vj(x∗)| > 0 for j ∈ J. We
deﬁne the control input uj for j ∈ J as
uj := α j
|vj(x∗)|. (S26)
Since α j > 0 and |vj(x∗)|> 0, we have uj > 0. Substituting this into the reaction equation
yields
∑
j∈ J
ujvj(x∗)Sj =
∑
j∈ J
α j
|vj(x∗)|(σj|vj(x∗)|)Sj =
∑
j∈ J
α jσjSj = 0. (S27)
For j / ∈ J, we have vj(x∗) = 0, so the term ujvj(x∗)Sj vanishes regardless of the value of uj.
Thus, x∗ is a steady-state (either stable or unstable) under the construc ted positive control
u.
III. PROOF OF THEOREM. S1
In the following section, we provide detailed proofs of the main theor em.
Theorem S1 (Main theorem) . If a free cell C(σ ) has a full SCK subset within its bound-
aries, then the controllable set of any state in the free cell x ∈ C(σ ) coincides with the entire
stoichiometric compatibility class, C(x) = W(x).
40

## Page 41

Let C(σ ) be a free cell, and suppose that it has a full SCK subset within its bou ndaries.
We denote the index set of the reactions by I, and the set diﬀerence of I and J by K := I\J.
Recall that the number of chemicals and reactions are N and R, respectively.
The source and target states are denoted by xsrc and xtgt, respectively. As no controlled
dynamics is possible if xsrc and xtgt are not in the same SCC, we suppose that xsrc ∈ W (xtgt)
holds.
The steps of the controls are divided into three steps; (step 1) co ntrol from xsrc to an
equilibrium state xeq, (step 2) control from xeq to a state inside the free cell x∗ ∈ C(σ ),
and (step 3) control from x∗ to xtgt. The feasibility of step 3 results from the unrestricted
controllability of free cells (see the main text for details). Thus, we p rove only step 1 and 2.
Note that our controls are asymptotic control, and thus, the sys tem does not reach exactly
the target state of each control step in ﬁnite time.
A. Step 1: control from xsrc to xeq
First, we show that the unique equilibrium state xeq exists as the intersection of the
balance manifolds {Mj}j∈ J and stoichiometric compatibility class W(xtgt). For that, it is
useful to introduce the reaction order matrix T, given by
Ti,r = n−
i,r − n+
i,r , (S28)
where n±
i,r are the reaction orders of the forward (+) and backward ( − ) reactions of the
chemical i in reaction r, respectively. The rth row vector of the reaction order matrix T
is denoted by Tr and is called the reaction order vector. Note that if the rth reaction is
implemented by SCK, Sr = Tr follows.
The deﬁnitions of the cells and balance manifolds are rewritten using t he reaction order
matrix T as follows:
Mr = {x ∈ RN
>0 | T ⊤
r ln x + ln kr = 0}, (S29)
C(σ ) = {x ∈ RN
>0 | sgn(T⊤ ln x + ln k) = σ }. (S30)
We denote the reaction order matrix and reversibility parameter ve ctor consisting of the
reactions in J by TJ and kJ , respectively. The intersection of the balance manifolds in J is
41

## Page 42

given by
⋂
j∈ J
Mj = {x ∈ RN
>0 |T⊤
J ln x = − ln kJ }, (S31)
and denoted as MJ. At any state in MJ , all the reaction ﬂuxes of reactions in J are zero,
and thus, the system is in a detailed-balanced state of the system o nly with the reactions in
J. For MJ to be non-empty, the equation
T⊤
J ln x = − ln kJ (S32)
should have a solution. A suﬃcient condition for Eq. (S32) to have a s olution is that the rank
of the reaction order matrix TJ equals the number of reactions in J, that is, rank TJ = |J|.
For that, we have a following useful proposition:
Proposition S3. Suppose C(σ ) is non-empty. There is an index set J ⊂ I, |J| = rank T
such that:
1. The matrix TJ satisﬁes rank TJ = rank T.
2. The balance manifolds {Mj}j∈ J are boundary of the cell C(σ ).
The proof of the proposition is provided at the end of this section.
Following the assumption of theorem S1, we assume that the free ce ll C(σ ) has a full
SCK subset within its boundaries {M}j∈ J. Thus, TJ can be replaced by SJ , and we have
rank SJ = rank S. The intersection MJ is then given by
MJ = {x ∈ RN
>0 |S⊤
J ln x = − ln kJ }, (S33)
and rank SJ = |J| holds, where kJ is the vector consisting of kj (j ∈ J). Because S⊤
J is
column full-ranked and rank S ≤ N holds, the equation
S⊤
J ln x = − ln kJ , (S34)
has solutions, and thus, MJ is not empty. Eq. (S34) provides rank SJ = rank S independent
conditions, leading to dim[ MJ] = N − rank S.
Let WJ (x) be the SCC of an arbitrary state x ∈ RN
>0 with reactions in J, i.e.,
WJ (x) = {x +
∑
j∈ J
ajSj |ai ∈ R} ∩ RN
>0. (S35)
42

## Page 43

Note that WJ (x) does not necessarily have the same dimension as the original SCC,
W(x), because the reactions in K are not included. However, owing to Prop. S3, we
have dim[ WJ (x)] = dim Im SJ = rank SJ = rank S, and which equals to dim[ W(x)]. By
combining this result with the dimensionality argument of MJ , we have
dim[WJ (x)] + dim[MJ] = N. (S36)
In addition, for any states x1, x2 ∈ W J (x) and y1, y2 ∈ M J, the following holds:
⟨x1 − x2, ln y1 − ln y2⟩ = ⟨SJ a, ln y1 − ln y2⟩ = ⟨a, S⊤
J (ln y1 − ln y2)⟩ = 0, (S37)
where ⟨·, ·⟩ represents the dual product between the concentration space and logarithm-
transformed concentration space. In the ﬁrst equality, we have used the fact that x1 and x2
are in the same SCC, and thus, there is a vector a ∈ R|J| satisfying SJ a = x1 − x2. The
second equality is due to the fact that y1 and y2 are in MJ, and thus, satisfy Eq. (S34).
From Eq. (S36) and Eq. (S37), WJ (x) and MJ are dually orthogonal complements of each
other. The intersection of dually orthogonal complements with Eq ( S36) always exists and
is unique [67, 68]. We denote the intersection of WJ (xtgt) and MJ by xeq.
The reactions in J are exactly balanced, and thus, the thermodynamic part of the re ac-
tions in J is zero at xeq. Also, xeq is an adherent point of the free cell. Thus, an arbitrarily
small displacement from xeq can lead to the entrance of the free cell, that is, there is a vector
a with an inﬁnitesimally small norm satisfying sgn p(xeq + Sa) = σ . As this balanced state
can be seen as the detailed-balanced state for the reaction syste m (Eq. (S1)) with uj > 0
for j ∈ J while uk = 0 for k ∈ K, we term it the equilibrium state.
Let us consider the reduced model with only reactions in J
dx
dt = SJ uJ ⊙ pJ (x), (S38)
where uJ ∈ R|J|
>0 and pJ(x) are the control inputs for the reaction in J, and thermodynamic
part of the reactions in J, respectively. Here, we have
WJ (x) = W(x) (S39)
for any x ∈ RN
>0 because Im SJ ⊆ Im S and rank SJ = rank S hold.
Since uJ is a constant vector, Eq. (S38) can be considered as an autonomo us dynamical
system with the mass–action kinetics. Also, because xeq satisﬁes Eq. (S34), xeq is the
43

## Page 44

detailed-balanced state of Eq. (S38). Thus, xeq is the globally asymptotically stable steady-
state, particularly, the equilibrium state of the system within WJ (xtgt) = W(xtgt) [59–61].
The original model (Eq. (S2)) is reduced to Eq. (S38) by setting ui(t) > 0 for i ∈ J and
ui(t) = 0 for i ∈ K. Let us denote one such control as ueq.
Overall, any state in W(xtgt) can be asymptotically controlled to xeq by setting u(t) =
ueq.
B. Step 2: control from a neighboring state of xeq to x∗
1. Overview of the proof of Step 2
First, we divide Eq. (S2) into the two parts, namely, the term gener ated by the reaction
with indices in J and K, respectively. We denote the stoichiometric matrices, controls, a nd
thermodynamic part of the reactions as SJ , SK, uJ (t), uK(t), pJ (x), and pK(x), respectively.
The diﬀerential equation is given by
dx
dt = SJ uJ (t) ⊙ pJ (x) + SKuK(t) ⊙ pK(x) (S40)
In the following, we show the existence of control to x∗ ∈ C(σ ) from neighbors of xeq by
utilizing the stability of an equilibrium state xeq. For this purpose, we consider the linearized
dynamics of Eq. (S40) around x∗ which is suﬃciently close to xeq.
Let x∗ be a neighbor of xeq and be in the free cell C(σ ). The relationship between x∗
and xeq is given by
x∗ := xeq + ǫc (∥c∥ = 1, ǫ ≪ 1), (S41)
where c is selected such that x∗ ∈ C(σ ). By expanding Eq. (S40) around x∗ and retaining
the terms up to the ﬁrst order in ∆ x := x − x∗, we obtain
d∆ x
dt =
[
SJ uJ (t) ⊙ pJ (x∗) + SKuK(t) ⊙ pK(x∗)
]
+
[
SJ diag{uJ (t)} ∂pJ
∂x (x∗) + SK diag{uK(t)} ∂pK
∂x (x∗)
]
∆ x (S42)
Hereafter, we denote
JX(uX, x∗) := SX diag{uX(t)} ∂pX
∂x (x∗), (X = J, K ).
44

## Page 45

We say JX(uX, x∗) is stable if the real parts of all eigenvalues of JX(uX, x∗) are negative
except for the zero eigenvalue corresponding to the left null spac e of SX . For the sum of the
matrices, JJ (uJ, x∗) + JK(uK, x∗), we say that it is stable if the eigenvalue of the sum of
the matrix satisﬁes the above condition. In this case, we allow the eig envalues corresponding
to the left null space of S to be zero.
The condition for ∆ x = 0 being stable is the vanishment of the ﬁrst term on the right-
hand side of Eq. (S42) and the stability of the second term. We will sh ow the following
two claims: First, there are constants uJ (t) = u∗
J and uK(t) = u∗
K
12 which leads to the
vanishment of the ﬁrst term on the right-hand side in Eq. (S42). Se cond, the matrix in the
second term JJ(uJ , x∗) + JK(uK, x∗) is stable with such u∗
J and u∗
K. The local asymptotic
stability of ∆ x = 0 means that x = x∗ is locally asymptotically stable in Eq. (S40). The
basin of attraction to ∆ x = 0 is at least as large as the range of linearization of the dynamics
(Eq. (S42)) being validated. Thus, when the state reaches suﬃcie ntly close to xeq by the
control in the step 1, we can switch the control from ueq to u∗ to asymptotically control the
state to x∗.
2. Vanishment of the ﬁrst term
First, we show the vanishment of the ﬁrst term. According to the d eﬁnition of the free
cell, we have cone {σiSi}i∈ I = span {σiSi}i∈ I. From Lemma. S1, {σiSi}i∈ I is positively
dependent, and thus, there is a positive linear combination of {σiSi}i∈ I leading to the zero
vector, that is ∑
i∈ I
wiσiSi = 0, (wi > 0). (S43)
By comparing Eq. (S43) with the ﬁrst term in Eq. (S40) being zero,
SJ u∗
J ⊙ pJ (x∗) + SKu∗
K ⊙ pK(x∗) = 0 (S44)
⇔
∑
i∈ I
u∗
i pi(x∗)Si = 0 (S45)
we have u∗
i = wi/ |pi(x∗)|> 0 for i ∈ I.
For the later proof, we show that the leading term of u∗
K is of the order of ǫ if we take
u∗
J so that its leading term is the 0th order of ǫ, i.e, u∗
J = u(0)
J + O(ǫ). Intuitively, this
12 As an abbreviation, we also denote the concatenated vector of uJ (t) and uK(t) by u(t) = u∗
45

## Page 46

is because pJ (x∗) = O(ǫ) and pK(x∗) = O(1) hold, and thus, u∗
K should be O(ǫ) if u∗
J is
chosen to be O(1).
To verify this claim, we expand pJ(x∗) = pJ(xeq + ǫc) in Eq. (S44) on ǫ. Then, we have
0 = SJ (u(0)
J + O(ǫ)) ⊙ pJ(xeq + ǫc) + SKu∗
K ⊙ pK(x∗) (S46)
=
(
SJ (u(0)
J + O(ǫ)) ⊙ pJ (xeq) + ǫSJ diag{u(0)
J } ∂pJ
∂x (xeq)c + O(ǫ2)
)
+ SKu∗
K ⊙ pK(x∗)
=
(
ǫSJ diag{u(0)
J } ∂pJ
∂x (xeq)c + O(ǫ2)
)
+ SKu∗
K ⊙ pK(x∗) (S47)
= ǫq + SKu∗
K ⊙ pK(x∗) (S48)
where q = SJ diag{u(0)
J } ∂ pJ
∂ x (xeq)c + O(ǫ) and note that pJ (xeq) = 0 holds.
From Eq. (S48), we have
− ǫq ∈ cone{Sipi(x∗)}i∈ K = cone{Siσi}i∈ K,
in particular, − q ∈ cone{Siσi}i∈ K. Thus, there is a non-negative constant ai such that
− q =
∑
i∈ K
aiSiσi. (S49)
Note that ai remains O(1) as ǫ → 0 since the leading term of q is independent of ǫ. The
choice of a is not necessarily unique. We choose one arbitrarily and multiply both s ides by
ǫ. By comparing this with the coeﬃcient of the second term in Eq. (S48 ), we obtain
ǫai = u∗
i pi(x∗), (i ∈ K), (S50)
and u∗
i is given by
u∗
i = ǫ ai
pi(x∗) , (i ∈ K).
Because pi(x∗) → pi(xeq) ̸= 0 holds as ǫ → 0, we have u∗
i = O(ǫ). Henceforth, to emphasize
that u∗
K is of the order ǫ, we write
u∗
K = ǫu◦
K.
3. Stability of the second term
Next, we show that the second term is stable. We evaluate the chan ge in the eigenvalues of
the matrix JJ(u∗
J , x∗)+JK(u∗
K, x∗) by perturbatively incorporating the eﬀect of JK(u∗
K, x∗)
into JJ (u∗
J, xeq). Since xeq is asymptotically stable, the eigenvalues of JJ (u∗
J, xeq) are all
46

## Page 47

negative except for the zero eigenvalue corresponding to the left null space of SJ . The changes
in the eigenvalues with the non-zero real part are evaluated by utiliz ing the generalized
Bauer–Fike theorem [72].
Let ∥ · ∥p be the p-norm of the matrix and κp(V ) be the condition number of a matrix V
induced by the p-norm. Suppose that A ∈ CN,N is a matrix with eigenvalues {λ i}N
i=1 and is
converted to
˜A = V − 1AV = diag(Ai),
where Ai’s are triangular Schur form. Suppose that the matrix A is perturbed by matrix
B ∈ CN,N as A + δB, ( δ > 0), and the eigenvalues of A + δB are denoted by {µ i}N
i=1.
The generalized Bauer–Fike theorem states that for any i ∈ { 1, 2 . . . , N }, there exists j ∈
{1, 2 . . . , N } such that
|µ i − λ j| ≤ max{θ, θ 1/q } (S51)
holds with
θ = δCκ p(V )∥B∥p, (S52)
where C and q are positive, ﬁnite-valued constants computed from the Schur blo cks of the
matrix A (see [72] for details). Note that if A is diagonalizable, C and q become unity, and
the classical Bauer–Fike theorem [73] is recovered.
Here, we apply the generalized Bauer–Fike theorem to the matrix JJ(u∗
J , x∗)+JK(u∗
K, x∗).
As x∗ := xeq + ǫc and ǫ ≪ 1, we have
JJ (u∗
J, x∗) = JJ(u∗
J , xeq + c ǫ)
= JJ(u∗
J , xeq) + ǫ
N∑
i=1
∂
∂xi
JJ(u∗
J , x)
⏐
⏐
⏐
x=xeq
ci + O(ǫ2).
Also, for JK(u∗
K, x∗) we have
JK(u∗
K, x∗) = JK(ǫu◦
K, xeq + ǫc)
= ǫJK(u◦
K, xeq + ǫc)
= ǫJK(u◦
K, xeq) + O(ǫ2).
Retaining terms up to the ﬁrst order in ǫ, we have
˜Jǫ(u∗
J, u∗
K, xeq) := JJ (u∗
J, xeq) + ǫ
N∑
i=1
∂
∂xi
JJ(u∗
J , x)
⏐
⏐
⏐
x=xeq
ci + ǫJK(u◦
K, xeq) (S53)
47

## Page 48

We denote the ﬁrst order of ǫ term as
∆( u∗
J , u∗
K, xeq) =
N∑
i=1
∂
∂xi
JJ(u∗
J , x)
⏐
⏐
⏐
x=xeq
ci + JK(u◦
K, xeq)
in the following.
Now we apply the generalized Bauer–Fike theorem. Here, we denote δ-independent part in
Eq. (S52) as θ0 which is determined by the original matrix JJ(u∗
J , xeq) and the perturbation
matrix ∆( u∗
J , u∗
K, xeq). Let the eigenvalues of ˜Jǫ(u∗
J , u∗
K, xeq) be denoted by {µ i}N
i=1 and
those of JJ (u∗
J, xeq) by {λ i}N
i=1. Because the inﬁnitesimal parameter corresponding to δ in
Eq. (S52) is ǫ, from the generalized Bauer–Fike theorem, for any µ i there exists λ j such that
|µ i − λ j| ≤ max{ǫθ0, (ǫθ0)1/q } (S54)
holds. Therefore, we can take ǫ small enough so that if we have Re λ j < 0 then Re µ i < 0
holds.
Although Eq. (S54) also holds for zero eigenvalues, tighter bounds on the zero eigenvalues
are necessary for the stability analysis because there is a possibility that some of the zero
eigenvalues in JJ(u∗
J , x∗) become non-zero by perturbations. 13 We show that the zero
eigenvalues of JJ (u∗
J , xeq) remain zero after inﬁnitesimally small perturbations, that is, the
zero eigenvalues of JJ(u∗
J , xeq) remain zero in JJ (u∗
J, x∗) + JK(u∗
K, x∗).
Recall that we have Im S = Im SJ (Eq. (S39)). We rewrite Im
[
JJ(u∗
J , x∗) + JK(u∗
K, x∗)
]
as follows
Im
[
JJ(u∗
J , x∗) + JK(u∗
K, x∗)
]
= Im
[
SJ diag{u∗
J } ∂pJ
∂x (x∗) + SK diag{u∗
K} ∂pK
∂x (x∗)
]
= Im
[(
SJ SK
)
diag{u∗
J , u∗
K}


∂ pJ
∂ x
∂ pK
∂ x

 (x∗)
]
(S55)
= Im
[
Sdiag{u∗} ∂p
∂x (x∗)
]
(S56)
=: ImJ (u∗, x∗) (S57)
where ( SJ SK) and ( ∂pJ /∂ x ∂pK/∂ x)⊤ are the horizontal and vertical concatenation of the
matrices, respectively. diag {u∗
J, u∗
K} is the diagonal matrix of the concatenated vector of u∗
J
and u∗
K. S, u, and p are the stoichiometric matrix, control input vector, and thermod ynamic
13 Since xeq is globally asymptotically stable in SCC, JJ (u∗
J , xeq) has no pure imaginary eigenvalue.
48

## Page 49

part of the reaction function vector consisting of all reactions, r espectively. As JJ(u∗
J , x∗) +
JK(u∗
K, x∗) and J (u∗, x∗) are equivalent as a linear map, we deal with J (u∗, x∗) in this
section.
In the following, we denote diag {u∗
X} ∂ pX
∂ x (x) as AX(x). A(x) without subscript represents
diag{u∗} ∂ p
∂ x (x). The goal of this section is to show the existence of the state x∗ ∈ C(σ ) ∩
W(xeq) being suﬃciently close to xeq such that the following equation holds
ImJ (u∗, x∗) = Im JJ(u∗
J , xeq). (S58)
If this holds, we have ker J (u∗, x∗) = ker JJ (u∗
J, xeq), and thus, for any eigenvectors of
JJ(u∗
J , xeq) corresponding to the zero eigenvalue, we have J (u∗, x∗)v = 0. Therefore, the
eigenvalues of J (u∗, x∗) that correspond to these vectors remain zero.
To this end, we show
ImJ (u∗, xeq) = Im JJ(u∗
J , xeq) (S59)
holds (note that the input of J here is xeq, not x∗). Since xeq is asymptotically stable
within the SCC, rank SJ = rank JJ(u∗
J , xeq) holds; otherwise, xeq cannot attract states from
the entire SCC. Thus, the matrix rank of AJ (xeq) and SJ satisfy rank AJ (xeq) ≥ rank SJ .
Further,
rank AJ (xeq) = rank SJ
holds because AJ is |J| × N matrix, |J| is less than or equal to N, and rank SJ is |J|.
The matrix AJ (xeq) is surjective as a linear map RN → RM , and thus, we have
ImSJ AJ (xeq) = Im SJ . (S60)
As Im S = Im SJ , the condition to be proven (Eq. (S59)) is equivalent to the following
conditions
ImJ (u∗, xeq) = Im JJ (u∗
J , xeq) (S61)
⇔ ImSA(xeq) = Im SJ AJ (xeq) (S62)
⇔ ImSA(xeq) = Im SJ (S63)
⇔ ImSA(xeq) = Im S. (S64)
Thus, it is suﬃcient to show
ImSA(xeq) = Im S. (S65)
49

## Page 50

As Eq. (S65) is equivalent with
rank A(xeq) ≥ rank S, (S66)
we show Eq. (S66) as follows. Note that the ( n, r ) element of the AX (x) matrix is given
by ur∂pr/∂x n(x) where r is the rth index of X which is either nothing or J. Since J is a
subset of I, all the column in AJ (x) is in A(x). So, Im A(x) ⊇ Im AJ (x) holds.
By using Im A(x) ⊇ Im AJ (x) and rank AJ (xeq) = |J|, we have
rank A(xeq) = dim Im A(xeq) ≥ dim Im AJ (xeq) = rank AJ (xeq) = |J|= rank S.
Therefore, we have rank A(xeq) ≥ rank S, and so, Im JI(u∗
I, xeq) = Im JJ (u∗
J , xeq).
Since S and SJ are constant matrices and A(x) and AJ (x) are continuous matrices of x,
we have x∗ in Nǫ(xeq) ∩ C(σ ) such that Im JI(u∗
I, x∗) = Im JJ(u∗
J , xeq) holds, where Nǫ(xeq)
is the ǫ neighborhood of xeq restricted to SCC, that is,
Nǫ(xeq) := {x ∈ W (xeq) | ∥x − xeq∥ < ǫ}.
C. Summary of the proof
We have shown the existence and uniqueness of the equilibrium state xeq on the boundary
of the free cell C(σ ) and the local asymptotic stability of x∗ ∈ C(σ ) in Eq. (S42) with the
control u∗. Thus, ﬁrst, we can control the system from xsrc ∈ W (xtgt) to xeq with the
constant control satisfying
u =





> 0 ( i ∈ J)
= 0 ( i ∈ K)
. (S67)
Next, we switch the control to uJ = u∗
J and uK = u∗
K. The state converges to a state being
arbitrarily close to x∗, in particular, inside the free cell. Once the state enters the free c ell,
we can take advantage of the unrestricted controllability inside the free cell to control the
state to xtgt. The proof of theorem S1 is completed.
It is noteworthy that the asymptotic stability of the equilibrium stat e is utilized twice;
ﬁrst, to show the existence of the control to xeq and second, to show the stability of the
equilibrium state for checking the stability of ˜Jǫ(u∗
J , u∗
K, xeq). Also note that the coincidence
of the controllable set and the SCC holds as long as the global asympt otic stability of
50

## Page 51

the detailed-balancing, adherent point of the target free cell C(σ ) holds. Therefore, our
argument never exclude the possibility that models with non-SCK also have the universal
controllability.
In the proof, we supposed that all reactions are reversible, i.e, k ∈ RR
>0. However, this
is not necessary. If reactions in the index set H are irreversible, the balance manifolds of
reactions in H do not exist. However, the arguments above hold by replacing the in dex
set of the reactions I by I\H. If there is a full SCK subset that indices of corresponding
reactions are chosen from I\H, we have an equilibrium state, xeq, as an adherent point of
the free cell C(σ ), and it is globally asymptotically stable. The following part of the proo f
does not depend on the reversibility of the reactions.
D. Cases with σ /∈ {− 1, 1}R
Here, we consider cases where the target state xtgt is in the free cell C(σ ) with σi = 0 for
some indices. Note that there is the upper limit of the count of the ze ro elements in σ . The
rth element of σ being zero means that the cell C(σ ) is a subset of the balance manifold
of the rth reaction. The maximum number of intersecting balance manifold is g iven by
rank S ≤ N.
For a clear statement of the condition, we introduce the parental cell
Deﬁnition S5 (Parental cell). A cell C(η ) is the parental cell of the cell C(σ ) if the sign
vector η is given by
ηi =





1 or − 1 ( σi = 0)
σi (σi ̸= 0).
(S68)
The parental cell of C(σ ) is generally not unique while the parental cell of C(σ ) with
σ ∈ {− 1, 1}R is C(σ ) itself.
Let Lσ be the index set of zero elements in σ , that is, Lσ := {i ∈ I |σi = 0}. For proving
the controllability of a free cell C(σ ) with σ ∈ {− 1, 0, 1}R, we assume that there is at least
one parental cell of the free cell C(σ ), C(η ), with a full SCK subset {Mj}j∈ J where Lη ⊂ J
holds. Also, we denote the set diﬀerence of I and J by K, i.e., K := I\J.
Under the condition described above, we can show the existence an d uniqueness of the
equilibrium state xeq := W(xtgt) ∩ M J in the identical manner as in the previous proof.
51

## Page 52

Therefore, the equilibrium state xeq exists uniquely and is globally asymptotically stable in
the SCC.
Next, we deﬁne x∗ so that
x∗ = xeq + ǫc , x∗ ∈ C(σ ) ⊂ M Lσ , (∥c∥ = 1, ǫ ≪ 1)
holds. Note that such c exists for arbitrarily small ǫ because xeq ∈ M J ⊂ M Lσ (Recall that
J is chosen so that Lσ ⊂ J holds.). By expanding the right-hand side of Eq. (S40) on ǫ in
the same manner as in the previous proof,
d∆ x
dt =
[
SJ\ Lσ uJ\ Lσ (t) ⊙ pJ\ Lσ (x∗) + SKuK(t) ⊙ pK(x∗)
]
+
[
SJ diag{uJ(t)} ∂pJ
∂x (x∗) + SK diag{uK(t)} ∂pK
∂x (x∗)
]
∆ x. (S69)
Note that pl(x∗) = 0 holds for l ∈ Lσ because x∗ ∈ M Lσ while the derivative ∂pl/∂ x (l ∈
Lσ ) is a non-zero vector. Since C(σ ) is the free cell, the stoichiometric vectors {σiS}i∈ I\ Lσ
are positively dependent, and thus, there exists a control vecto r u such that the ﬁrst term
of Eq. (S69) vanishes. Also, such control vector can be chosen s o that uJ\ Lσ and uK be
O(1) and O(ǫ) for ǫ → 0, respectively. The second term of Eq. (S69) is identical to that in
Eq. (S40). Thus, we can apply the same argument as that in the pre vious proof.
As C(σ ) is a free cell, we can utilize the unrestricted controllability inside the f ree cell
to control the state to xtgt.
The statement of the main theorem is summarized as follows.
Theorem S1’ (Main theorem with ternary σ ). For a given free cell C(σ ), if there is
a parental cell C(η ) having a full SCK subset within its boundaries {Mj}j∈ J satisfying
Lη ⊂ J, then the controllable set of any state in the free cell x ∈ C(σ ) coincides with the
entire stoichiometric compatibility class, C(x) = W(x).
If all reactions are implemented by SCK, every cell has at least rank S balance manifolds
as its boundary and full SCK subsets, Cor. S1 and S2 are unmodiﬁed .
E. Proof of the Prop. S3
Proposition S3. Suppose C(σ ) is non-empty. Then one can construct an index set J ⊂ I
with |J|= rank T such that:
52

## Page 53

1. The matrix TJ satisﬁes rank TJ = rank T.
2. The balance manifolds {Mj}j∈ J are boundary of the cell C(σ ).
Proof. Without loss of generality we assume σi = 1 for all i.
As the closure of the cell is the polyhedron in the logarithm-transfo rmed space, we rewrite
the variables and parameters as y = − ln x and b = ln k. Then, the closure of the cell is
given by
C(σ ) =
{
y ∈ RN |T⊤ y ≤ b
}
.
Since C(σ ) ̸= ∅, this system has an interior point and is full-dimensional. 14
The ( N − 1) dimensional boundaries of N dimensional polyhedra are termed facet and
there is a useful theorem for identifying them:
Theorem S2’ (Theorem 8.1 in [74]) . Let Ax ≤ b be a system of non-redundant inequalities
in RN . Then there is a one-to-one correspondence between the face ts of the polyhedron
P = {x ∈ RN |Ax ≤ b}
and the individual inequalities in Ax ≤ b. Explicitly, the ith facet is
Fi = {x ∈ P |a⊤
i x = bi},
where ai is the ith row of A.
Here the inequalities Ax ≤ b are redundant if removing one inequality from Ax ≤ b
does not change the polyhedron. In general, Ty ≤ b may contain redundancies, so we apply
Farkas’ Lemma to eliminate them:
Proposition S4 (Farkas’ Lemma). Let A be an Rm× d matrix and let α ∈ Rd, β ∈ R. Then,
α ⊤ x ≤ β holds for all x satisfying Ax ≤ b if and only if there exists λ ≥ 0 such that
λ ⊤ A = α ⊤ and λ ⊤ b ≤ β .
By iteratively removing the redundant inequalities, one obtains the n on-redundant in-
equalities deﬁning the polyhedron
T∗ y ≤ b∗. (S70)
14 In general, it is possible that the combination of inequalities leads to th e equality constraint (e.g. a⊤ x ≤ b
and − a⊤ x ≤ − b together enforce a⊤ x = b.). Such resulting equalities are termed implicit equalities. Full-
dimensional polyhedra are the polyhedra without implicit equalities. Fu ll-dimensional polyhedra have the
same dimension as the space that the polyhedra are embedded.
53

## Page 54

From Farkas’ lemma, the rank of the matrix T does not change by the removal of redundant
inequalities. Thus, rank T∗ = rank T holds. From Thm. S2’, there are at least rank T facets
of C(σ ), each corresponding to a balance manifold. One can construct th e index set J by
selecting the reaction index such that the corresponding column ve ctors of rank T∗ become
all linearly independent. Then, |J|= rank TJ = rank T and the balance manifolds {M}j∈ J
correspond to the boundary of the cell C(σ ).
IV. DERIV A TION OF THE NON-SCK FROM SCK REACTIONS
Let us consider the reaction
2X ⇌ Y,
catalyzed by enzyme E. An ordinary reaction diagram for this react ion is shown in Fig. S1(a),
which leads to the following coarse-grained reaction kinetics when th e quasi-equilibrium
method is applied:
v ∝ [X]2 − k[Y ]
KM + [X]/K X, 1 + [X]2/K X, 2 + [Y ]/K Y
,
where [ ·] is the concentration of the corresponding chemical. K∗s are the Michaelis-Menten
constant set by the forward and backward reaction rate consta nts of each elementary step.
k is the reversibility parameter. Since we focus only on the function fo rm, we do not go
into the details of the parameters. The point here is that the therm odynamic part of this
kinetics is given by [ X]2 − k[Y ], and the reaction order matches the stoichiometry of each
chemical, i.e., the reaction rate function is stoichiometrically compatib le kinetics (SCK).
Another reaction diagram is shown in Fig. S1(b), where the irrevers ible maturation pro-
cess EX → E∗X is introduced. Owing to the irreversibility of the maturation process , the
process E + X ⇌ EX → E∗X ⇌ E∗XX → E + Y becomes irreversible 15. Thus, we must
introduce another diagram for the backward reaction Y → 2X to make the coarse-grained
reaction reversible. The diagram for the backward reaction is the r ight side of the diagram.
The dynamics of the concentrations is given by the following equation
15 Even if the ﬁnal step is made reversible, the following argument does not change qualitatively. We set the
ﬁnal step irreversible for the ease of calculation.
54

## Page 55

d[E]
dt = − ka
+[E][X] + ka
− [EX ] + v[E∗XX ] − la
+[E][Y ] + (la
− + w)[EY ] (S71)
d[EX ]
dt = ka
+[E][X] − ka
− [EX ] − kb
+[EX ] (S72)
d[E∗X]
dt = kb
+[EX ] − kc
+[E∗X][X] + kc
− [E∗XX ] (S73)
d[E∗XX ]
dt = kc
+[E∗X][X] − kc
− [E∗XX ] − v[E∗XX ] (S74)
d[EY ]
dt = la
+[E][Y ] − la
− [EY ] − w[EY ]. (S75)
The de-novo production of the enzyme is not considered within the timescale that we focus
on, and the total concentration of the enzyme is conserved:
[E] + [EX ] + [E∗X] + [E∗XX ] + [EY ] = [ E]T = const.
The steady-state solutions of the ﬁnal complexes E∗XX and EY are given by
[E∗XX ] = ka
+kb
+
v(ka
− + kb
+) [E][X] (S76)
[EY ] = la
+
la
− + w [E][Y ] (S77)
Thus, the net reaction ﬂux is given by
˜v = v[E∗XX ] − w[EY ] = ˜vmax[E]
(
[X] − ˜k[Y ]
)
,
where ˜vmax =
ka
+kb
+
ka
− +kb
+
and ˜k =
wla
+
la
− +w
ka
− +kb
+
ka
+kb
+
. Note that [ E] is the concentration of the free en-
zyme and ˜vmax[E] leads to the kinetic part of the reaction rate function. The therm odynamic
part is given by [ X] − ˜k[Y ], and thus, it is non-SCK.
In this derivation, the enzyme must recognize the metabolite with wh ich the complex is
formed, and the subsequent reaction is determined. In addition, f or non-SCK kinetics, the
reactions EX → E∗X and EY → E + 2X must be irreversible. Thus, this reaction 2 X ⇌ Y
needs to be externally driven.
V. CONSTRUCTION OF THE SUB-NETWORK AND ITS LINK TO THERMO-
DYNAMIC CONSISTENCY
We have discussed the relationship between Stoichiometrically Compa tible Kinetics
(SCK) and thermodynamic consistency in the main text. In this sect ion, we clarify how the
55

## Page 56

FIG. S1. (a) A reaction diagram for SCK. (b) A reaction diagra m for non-SCK. The coarse-grained
reaction scheme is the same with (a), 2 X ⇌ Y . The black arrows and red arrows represent the
reversible reactions and irreversible reactions, respect ively. “2” on the red arrow on the right
bottom represents that two molecules of X are produced by the reaction. The corresponding
symbols of rate constants of the reactions are put on the arro ws in (b).
active sub-network used in our proof is constructed and, more imp ortantly, why our results
do not rely on global parameter constraints often associated with thermodynamic consis-
tency. In this section, we term the model with some reactions turn ed-oﬀ for the relaxation
to the equilibrium state the reduced model.
Speciﬁcally, while thermodynamic consistency in a cyclic network gene rally requires the
kinetic parameters to satisfy the detailed-balance condition at equ ilibrium, our theorem
ensures global controllability even when such conditions are violated . We illustrate this
point using the Onsager model as an example.
The Onsager model is given by the following reactions:
R1 : A ext ⇌ A
R2 : A ⇌ B
R3 : B ⇌ C
R4 : C ⇌ A
R5 : C ext ⇌ C
the schematic of the reaction network is given by Fig. S2(a). The OD E with control is given
56

## Page 57

by
d
dt





a
b
c




 =





1 − 1 0 0 0
0 1 − 1 0 0
0 0 1 − 1 1















u1
u2
u3
u4
u5










⊙










v1
v2
v3
v4
v5










(S78)
Here, we consider a simple case that the reaction rate function follo ws the mass-action
kinetics:
v1 = k+
1 aext − k−
1 a, (S79)
v2 = k+
2 a − k−
2 b, (S80)
v3 = k+
3 b − k−
3 c, (S81)
v4 = k+
4 c − k−
4 a, (S82)
v5 = k+
5 cext − k−
5 c. (S83)
The rank of the stoichiometric matrix is three, and thus, we need to turn oﬀ two reactions
to obtain a reduced model that relaxes to the equilibrium state. The choice of the reactions
turned oﬀ depends on the target of the control.
Note that the model has a reaction cycle A → B → C → A and its reverse. Let us consider
the model with these three reactions, R 2, R3 and R 4. The detailed-balance condition for the
reaction cycle is given by
k+
2 k+
3 k+
4
k−
2 k−
3 k−
4
= 1. (S84)
If this condition is not satisﬁed, the relaxation to equilibrium is prohibit ed, but the cycle
current persists, i.e., the system relaxes to the non-equilibrium ste ady state.
The remark of this section is that our theorem (Theorem. S1 and S1 ’) holds even if a
model does not satisfy the condition on the parameters like Eq.(S84 ). It is because for
the construction of the reduced model, at least a single reaction in e ach cycle is turned
oﬀ. According to Proposition.S3, for any non-empty cell C(σ ) there is an index set J such
that the balance manifolds {Mj}j∈ J are the boundaries of the cell and the stoichiometric
matrix consisting of the reactions in J, SJ , satisﬁes rank S = rank SJ = |J|. Therefore, the
57

## Page 58

stoichiometric vectors of the reactions in J must be linearly independent, and no cycle can
be in the reaction network SJ .
In the Onsager model, the rank of the full stoichiometric matrix is th ree, and thus, we
need to select three reactions to keep activated and the others a re turned oﬀ. If we construct
a reduced model with reaction R 2, R3 and R4, the stoichiometric matrix SJ is not full-ranked,
but rank SJ = 2. All allowed choices of the reactions so that rank S = rank SJ = |J| are
illustrated in Fig. S2(b). Readers can see that every reduced mode l has the detailed-balanced
equilibrium as its steady-state.
Two choices {R2, R3, R4} and {R1, R4, R5} do not satisfy the rank condition. Consistently,
the steady-state of the reduced model with either {R2, R3, R4} or {R1, R4, R5} is, in general,
the non-equilibrium steady-state (Fig.S2(c)).
While the detailed balance condition is not required for global controlla bility to states
within free cells, it is nevertheless noteworthy that the detailed bala nce condition determines
the number of free cells. Figure S3 illustrates the free cells of the On sager model for diﬀerent
choices of the reversibility parameters. In Fig. S3(a), the detailed balance condition is
violated, and there exists a free cell in which a net reaction ﬂow circu lates along the cycle
A → B → C → A. By contrast, when the detailed balance condition is satisﬁed, only a
single free cell remains, in which no net ﬂow exists along the cycle as sh own in Fig. S3(b).
In this section we used the Onsager model as a simple model for highlig hting the link
between the control and thermodynamic consistency. However, note that in linear models
such as the Onsager model, the controllability to the free cell is guar anteed from the chemical
reaction network theory [40]. We consider the model given by
dx
dt = Su(t) ⊙ v(x),
where v(x) is linear on x (monomolecular), and all reactions are reversible. According to th e
deﬁciency zero theorem, every reversible monomolecular reaction system has a unique ﬁxed
point in the positive stoichiometrically compatible class which is globally as ymptotically
stable for a given parameter set. Given the uniqueness of the ﬁxed point and Prop. S2, the
model has only a single free cell for a given parameters other than u. Thus, for an arbitrarily
selected constant control u(t) = uc ≻ 0, all states converge to the attractor inside the free
cell, and there is no other free cell. Since the control between any t wo states in the same free
cell is possible, the controllable set of a given state in the free cell is a whole stoichiometric
58

## Page 59

FIG. S2. (a) A reaction network of the Onsager model. (b) The allowed c ombinations of reactions to
be kept active for constructing the reduced model that relax es to the equilibrium state (rank SJ = 3),
(c) and does not relax to the equilibrium state but to the non- equilibrium steady state (rank SJ = 2).
The external A and C are not depicted for the ease of illustrat ion.
59

## Page 60

FIG. S3. The union of free cells in the Onsager model for diﬀerent choic es of the reversibility
constants k. (a) A case in which detailed balance is violated,
(
k+
2 k+
3 k+
4 /k −
2 k−
3 k−
4
)
̸= 1. The orange
and cyan regions correspond to free cells. In the orange cell , the reaction direction is given by
(1, 1, 1, 1, − 1), indicating a net reaction ﬂow along the cycle A → B → C → A. In contrast, in the
cyan cell, the reaction direction is (1 , 1, 1, − 1, − 1), for which no net ﬂow exists along the cycle. (b)
A case in which detailed balance is satisﬁed,
(
k+
2 k+
3 k+
4 /k −
2 k−
3 k−
4
)
= 1. In this case, there is only a
single free cell, whose reaction direction is given by (1 , 1, 1, − 1, − 1). The red, blue, green, yellow,
and purple planes are the balance manifolds of the reaction R 1, R 2, R 3, R 4, and R 5, respectively.
The following parameters are identical in panels (a) and (b) : aext = 1, cext = 0 . 1, k+
1 = 1 . 0,
k−
1 = 1 . 0, k+
5 = 5 . 0, and k−
5 = 1 . 0. In panel (a), the parameters are set to k+
2 = 2 . 0, k−
2 = 1 . 5,
k+
3 = 3 . 0, k−
3 = 1 . 0, k+
4 = 0 . 7, and k−
4 = 1 . 0, whereas in panel (b) all of k+
2 , k−
2 , k+
3 , k−
3 , k+
4 , and
k−
4 are set to unity.
compatibility class.
VI. ABSENCE OF UNCONTROLLABLE ST A TES IN THREE BIOCHEMICAL
MODELS
We have shown that the non-SCK kinetics is a key factor that the mo del has uncontrol-
lable states to a free cell, and indeed, there were regions from which no control is possible
to one of the two free cells in the toy model introduced in the section V in the main text.
60

## Page 61

Here we show, however, implementing reactions with non-SCK kinetic s does not necessarily
result in the existence of uncontrollable states to a free cell. In this section we show the
absence of such states in the popular biochemical models: Sel’kov mo del, Brusselator model,
and Schnakenberg model.
The original Sel’kov model is a minimal model of the glycolitic oscillation, a nd is given
by the following ODE
dx
dt = b − (a + yγ )x, (S85)
dy
dt = ( a + yγ )x − y. (S86)
The Sel’kov model is a model focusing on the positive feedback regula tion of the phospho-
fructokinase (PFK) whose reaction stoichiometry is
fructose-6-phosphate + ATP ⇌ fructose-1,6-bisphosphate + ADP.
In the Sel’kov model, the substrates and products of the reaction are lumped up to a single
variable X and Y , respectively.
There are three reactions
R1 : ∅ → X
R2 : X → Y
R3 : Y → ∅ ,
where ∅ represents the external environment. The second reaction is th e PFK reaction and
PFK enzyme activity is positively regulated by Y (ADP) as ( a + yγ ). a is the basal PFK
activity and the activity increases as y increase. For γ > 1, there is a parameter region of a
and b in which the Hopf bifurcation occurs and limit cycle attractor emerge s.
To study the controllability of the Sel’kov model, ﬁrst we need to make the reactions
reversible. In the original Sel’kov model, all reactions are irreversib le, meaning that there
is no balance manifold in R2
>0 and there is a single cell. Indeed, this cell is a free cell, and
thus, control between arbitrarily chosen pair of states is always p ossible. The freeness of
the cell is guaranteed by Proposition S2. The proposition states th at a ﬁxed point can exist
only in free cells and if there is a ﬁxed point in a cell, the cell is a free cell. T he original
Sel’kov model has the ﬁxed point
x = b
a + bγ , y = b
61

## Page 62

regardless of the parameter choice. Thus, the whole positive orth ant R2
>0 is a free cell.
To avoid this trivial controllability consequence, we need to introduc e the reversibility to
the model. We modify the reactions as follows;
R1 : X ext ⇌ X
(S87)
R2 : X ⇌ Y (S88)
R3 : Y ⇌ Yext. (S89)
We updated the model equation accordingly. By additing the contro l parameter u, now the
model equation is given by
dx
dt = u1(t)(b − x) − u2(t)(a + yγ )(x − k2y), (S90)
dy
dt = u2(t)(a + yγ )(x − k2y) + u3(t)(c − y), (S91)
where we have newly introduded parameters c and k2 representing the external concentration
of Y, and the reversibility parameter of the reaction R2, respectively. The reversibility
parameter should also be introduced for the reaction R1 and R3, though the reversibility
parameters of those reactions are considered to be absorbed to the external concentration
and maximum reaction rate. The maximum reaction rate of each reac tion is ﬁnally set to
unity by absorbing it to the control parameter ui(t).
Now the modiﬁed model is fully implemented by SCK. Here, we allow the re actions to
be implemented by non-SCK
dx
dt = u1(t)(b − x) − u2(t)(a + yγ )(xα − k2yβ ), (S92)
dy
dt = u2(t)(a + yγ)(xα − k2yβ ) + u3(t)(c − y). (S93)
The introduction of α and β (α, β ≥ 0) allows us to make the reaction rate kinetics to deviate
from SCK. Here, we did not make the reaction rate function for R1 and R3 be non-linear
on x and y, respectively. This is because even if we introduce nonlinearity, the balance
manifold is inherently unchanged, that is, the balance manifold of the reaction R1 with its
rate function as b − x is given by
M1 = {(x, y ) ∈ R2
>0 |x = b},
while with rate function as b − xδ is given by
M′
1 = {(x, y ) ∈ R2
>0 |x = b1/δ }.
62

## Page 63

Thus, the diﬀerence of M1 and M′
1 is recovered by changing b value, and introduction of δ
does not qualitatively change the structure of the balance manifold .
In this extension we allowed the reaction order to be an arbitrary po sitive real number.
However, we do not allow the chemical species which are not in the rea ction equation to
have non-zero value; for example, in the reaction R2 (Eq.(S88)), we do not allow the reaction
rate function to have a form like ( xα yµ − k2yβ ) with µ ̸= 0 because Y is not a reactant of
the forward reaction of R2. Note that the substrate-level regulations are possible within
this framework; for example, the reaction rate function of R2 can have a form like xλ (a +
yγ )(xα − k2yβ ), where the regulation term xλ (a + yγ ) modulates the enzyme activity but
this part does not aﬀect the controllability because it is the kinetic pa rt of the reaction rate
function and is absorbed to the control parameter u2(t).
The balance manifolds of Eq.(S92) and (S93) are given by
M1 = {(x, y ) ∈ R2
>0 |x = b}
(S94)
M2 = {(x, y ) ∈ R2
>0 |y = (xα /k 2)1/β }
(S95)
M3 = {(x, y ) ∈ R2
>0 |y = c}
(S96)
Note that the regulation term of the reaction R2, ( a + yγ), does not appear in the balance
manifold M∈ . This is because the change in the overall rate (enzyme activity) is a part of the
kinetic part in the decomposition of the reaction rate function into t he thermodynamic and
kinetic parts, v(x) = p(x)f (x). The kinetic part is absorbed into the control parameter u
and does not aﬀect the controllability; only the thermodynamic part determines the structure
of the cells and controllability.
In Eq.(S94)-(S96), only M2 can change its functional shape while M1 and M3 are the
parallel line to the Y-axis and X-axis, respectively. By chaning α and β , M2 can be convex,
linear, and concave function, while it is the monotonically increasing fu nction of x regardless
of the values of α, β , and k2 in α, β, k 2 > 0. As shown in Fig. S4, the possible conﬁguration
of the cells is fully determined by the order that M2 has the intersection with M1 and M3
(e.g., ﬁrst M2 intersects with M1 and then M3) regardless of whether M2 is convex, linear,
or concave function because M2 is a monotonically increasing function of x.
As shown in Fig. S4, the cell at the center is the only free cell in all cas es. The case where
M2 has an intersection simultaneously with M1 and M3 is an exception. In this case, only
63

## Page 64

FIG. S4. All conﬁgurations of the cells by changing the parameters a, b, c, α, β , and k2 which are
not control inputs but ﬁxed constants. By changing α and β , the balance manifold M2 becomes
the convex, linear, or concave function. The modulation of t he parameters changes the order that
M2 intersects with M1 and M3. When we follow M2 from the origin, the possible intersection
patterns are as follows; (1) ﬁrst intersects with M1 and then intersects with M3, (2) intersects
with M1 and M3 at an indentical point, and (3) ﬁrst intersects with M3 and then intersects with
M1. In the ﬁgure, we showed the order of intersection for three d iﬀerent functional forms of M2,
while the conﬁguration of cells are the same once the order of intersection is determined, regardless
of the functional form. The red, green, and blue lines or curv e are the balance manifold M1,
M2, and M3, respectively. The directed stoichiometric vectors of the reaction R1, R2, and R3 are
represented by the red, green, and blue arrows in each cell. T he cell ﬁlled with yellow is the free
cell.
64

## Page 65

the intersection point is the free cell 16.
In the both cases of order of intersection, there is always a contr ol to the free cell at
the center. First, reaching a balance manifold M3 using the reaction R3. If it is on the
boundary of the free cell, by using the conical combination of the re action R1 and R2, the
state is controlled to the free cell. If it is not on the boundary of the free cell, by using
only the reaction R1, the state is controlled to the intersection of M3 and either M1 or
M2, i.e., on a vertex of the free cell. As shown in the proof of the main the orem, the state
is controlled into the free cell from a vertex of the free cell. Thus, t he arbitrary state is
controllable to the free cell.
The same argument is applied for the Schnakenberg model. The mode l is extended to
have the reversible reactions, non-SCK rate functions, and the c ontrol as follows;
dx
dt = u1(t)(a − x) − u2(t)(b − x) + u3(t)x2(yβ − k3xα ), (S97)
dy
dt = u3(t)x2(yβ − k3xα ) + u4(t)(c − y). (S98)
In the Schnakenberg model, there are two exchange reactions of chemical X. As the function
form of the third reaction is nonlinear x2(yβ − k3xα ), the balance manifold can be convex,
linear, or concave depending on α and β values. But with the same argument that we have
presented for the Sel’kov model, the conﬁguration of the cell does not depend on whether the
M3 is convex, linear, or concave. The all conﬁgurations of the cells are shown in Fig. S5.
The congulations of the cells are symmetric for the exchange of pos ition of the balance
manifold M1 and M2. Thus, we depict only the cases where M2 is located at higher value
of X-coordinate than that of M1. An inherent diﬀerence in the Schnakenberg model is that
it has two connected free cells. But in any cases, every state is con trollable to any state in
the free cells by utilizing the reaction R4 to reach a suﬃciently close state to M4, and then
reach the free cells along M4.
Finally, we check the Brusselator model. The reversible Brusselator model with control
and non-SCK kinetics is given by
dx
dt = u1(t)(a − x) − u2(t)(xα − k1yβ ) + u3(t)x2(yγ − k3xδ) (S99)
dy
dt = u2(t)(xα − k1yβ ) − u3(t)x2(yγ − k3xδ). (S100)
16 By deﬁnition, a cell with a single point is a free cell because span {0} = cone {0}. Also, since such a cell
consists of only a single point, any points from the cell are mutually co ntrollable.
65

## Page 66

FIG. S5. The three conﬁgurations of the cells in the Schnakenberg mod el. The ﬁgure is made for
the case that M3 is linear while the conﬁguration does not change even if M3 is either convex
or concave. Colored lines and arrows represent the correspo nding balance manifolds and directed
stoichiometric vectors, respectively. The regions ﬁlled w ith yellow are the free cells.
An important feature of the Brusselator is that the stoichiometric vector of the reaction R2
and R3 is identical except for the sign. S2 = − S3. Because of this redundancy, there is no
2-dimensional free cell in the Brusselator model. All possible conﬁgu rations of the cells are
presented in Fig. S6. The conﬁgurations are classiﬁed into three ty pes based on where M2
and M3 have intersection in R2
>0: Fig. S6(a) no intersection, (b) at the greater x than a
(external concentration of X), and (c) at the lower x than a.
While there is no 2-dimensional free cell, the interval highlighted in yello w in the panels
are “free” in terms that any two points are mutually controllable; in F ig. S6(a) case for
instance, for controlling a state to another state with lower y value (note that both states
are on the same interval), one can utilize the stoichiometric vector o f the reaction R3 to
decrease y. Since this operation drags the state out from the interval, the st oichiometric
vector of the reaction R1 can recover the system’s state onto the interval. Iteration of th is
operation allows us to control the system from ( a, y 1) to ( a, y 2) with y1 > y 2. Feasibility of
the control to the opposite direction is conﬁrmed by the same argu ment while we utilize the
stoichimoetric vector of the reaction R2 for this purpose. In this sense, the interval is a free
cell, and indeed, the ﬁxed points are only on this interval.
By checking the conﬁgurations and cells and directed stoichiometric vectors, we can see
66

## Page 67

that an arbitrary state is controllable to the interval.
We have checked the controllability of the three popular 2-dimension al biochemical mod-
els. All of them have no state which is uncontrollable to a state in a fre e cell. In the
low-dimensional models, models with uncontrollable state to states in free cells might be
rare. Our toy model in the main text is dealt as a two-dimensional mod el, while the model
consists of three chemical species and a conserved quantity allowe d us to reduce the model
into two-dimensional. As far as we have explored we have never foun d any models consisting
of one or two chemical species and exhibit uncontrollability to a state in a free cell.
FIG. S6. The three conﬁgurations of the cells in the Brusselator mode l. The ﬁgure is made for
the case that M3 is linear while the conﬁguration does not change even if M3 is either convex
or concave. Coloed lines and arrows represent the correspon ding balance manifolds and directed
stoichiometric vectors, respectively. The regions ﬁlled w ith yellow are the free cells.
67
