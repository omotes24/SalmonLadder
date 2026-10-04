# Validity audit: assumptions, claims and implementation gaps

These are mathematical statements under explicit assumptions, not a claim that the paper implementation satisfies those assumptions. Numerical calibration is reported separately. The final product of scores, including the TINS factor, is an ID-ranking score, **not a valid p-value**.

## Lemma 1: rank validity

Let (Z1,...,Zn,Z*) have an exchangeable joint distribution conditional on a background B. Let their real scores (a1,...,an,a*) be produced by a measurable transformation equivariant to permutations of these n+1 objects, with all algorithmic randomness included in B or transformed equivariantly. Then

`p* = (1 + sum_i 1{ai <= a*})/(n+1)`

satisfies `P(p* <= alpha | B) <= floor(alpha*(n+1))/(n+1) <= alpha`. Reverse the inequality for an OOD-high score. Proof: conditional on the unordered score multiset, the queried object is uniformly distributed over its n+1 positions. At most floor(alpha*(n+1)) positions have conservative upper-tie rank at most alpha*(n+1). Ties can only reduce this count. Integrating over B yields marginal validity. This is not a conditional-on-the-realized-calibration guarantee.

For independent calibration and query from the same ID population and an independently frozen static scoring function, the hypotheses hold. See [Bates et al., Testing for Outliers with Conformal p-values](https://doi.org/10.1214/22-AOS2244). That paper distinguishes marginal and calibration-conditional guarantees; its stronger constructions are not implemented here.

## Lemma 2: finite symmetric propagation is not intrinsically invalid

Fix labeled support nodes and any background nodes. Suppose calibration and the queried ID node satisfy Lemma 1 conditional on this background. Suppose the graph builder is equivariant under permutations of those unlabeled nodes. Define `Wn=D^(-1/2) W D^(-1/2)`, unlabeled entries of y equal to zero, and start **all** nodes at u0=0. Every finite iterate

`u_(j+1)=lambda Wn u_j+(1-lambda)y`

is equivariant: permutation conjugates Wn, permutes y and u0, and induction applies. Thus the ranks of the calibration and query coordinates satisfy Lemma 1 even for exactly 15 iterations. The same is true for the unique exact solution, or a stopping rule invariant to permutation in exact arithmetic. Convergence is not necessary for this argument. Adaptive score construction can be admissible when the entire construction preserves the required exchangeability; see [Gazin, Blanchard and Roquain, 2024](https://proceedings.mlr.press/v238/gazin24a.html).

Implementation mapping:

- `core.solve(..., mode='L1')`: all-zero, 15 shared sweeps.
- `mode='L2'`: all-zero, explicit equation residual; converged flag is required.
- `mode='L0'`: first support values start at one; subsequent old nodes retain a state depending on their age. Swapping an old calibration node and a new ID node changes that history. This is not covered by the above induction. A node-array permutation with history transported is a weaker test than exchanging arrival roles and replaying history.
- `PrefixGraph`: all *currently available* nodes are candidates, and graph updates are exact incremental top-k. New future features are never queried. However, legacy `argpartition` tie selection is array dependent. Duplicate-feature tests expose a failure of graph equivariance. Absence of exact boundary ties, or a genuinely equivariant tie rule, is an additional assumption.
- Floating-point multiplication and reductions need numerical diagnostics; approximately symmetric outputs are not an exact mathematical proof.
- A fixed recent-history window preserves calibration while deleting old stream nodes. This age-based sampling also needs a separate argument, even with a symmetric solver on the retained graph.

[Conformal Inductive Graph Neural Networks](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3024ea88cec9f45a411cf4d51ab649c-Abstract-Conference.html) studies node/edge exchangeable graph sequences and updated calibration scores. Its assumptions and graph process are not automatically satisfied by this image-stream kNN graph, balanced shots, age-based retention or a warm solver. We do not transplant its theorem to REPRISE.

## Lemma 3: equation residual bounds propagation error and possible rank changes

For a symmetric normalized graph in exact arithmetic, `||Wn||2 <= 1`. For `0 <= lambda < 1`, let `A=I-lambda Wn`, `b=(1-lambda)y`, `u*=A^(-1)b`, and `r=b-A uhat`. Then

`||uhat-u*||2 <= ||r||2/(1-lambda)`.

Proof: the eigenvalues of A lie in [1-lambda,1+lambda], so its inverse has operator norm at most 1/(1-lambda). Every coordinate error is at most this bound delta. Therefore a pairwise ordering with exact margin greater than 2 delta cannot reverse. A p-rank can change only through calibration values within 2 delta of the query, and its absolute change is at most their count divided by n+1. This does **not** guarantee all ranks agree when margins are small.

`core.solve` records the equation-relative residual `||A u-b||2 / ||b||2`, not merely successive-iterate difference. FP64 direct solves validate the small graphs. The recorded error bound pertains to the frozen numerical matrix; for a rounded Wn with a certified norm bound `1+eta`, replace the denominator by `1-lambda*(1+eta)`. We have not certified eta for every large graph, so the large-run `l2_error_bound` field is a theoretical diagnostic, not a machine-verified certificate. The original convergence routine used successive-iterate tolerance 1e-6, a different condition.

The exact solution is the classical local/global consistency propagation solution; [Zhou et al.](https://proceedings.neurips.cc/paper_files/paper/2003/hash/87682805257e619d49b8e0dfdc14affa-Abstract.html). That method supplies a graph-based smoothness objective, not automatic OOD error control.

## Lemma 4: what calibration-role separation can establish

Assume S, the hyperparameters theta, and the arrival process are independent of four newly sampled i.i.d. ID calibration sets C1..C4; the stream does not react to detector outputs. Also assume the next ID query is independent of the past and identically distributed with calibration. In M1:

- A1(t) depends on S,theta,C1 and arrivals before t.
- A2(t) depends additionally on C2, not C3 or C4.
- Therefore the stage-3 admission score against A2(t), calibrated by C3, has conditional exchangeable ranks and marginal ID admission probability at most epsilon3.
- M(t) additionally depends on C3 but not C4. Hence its final pt readout, calibrated by C4, is marginally valid for a fresh ID query.

Conditioning here excludes the calibration set used for the stated rank. It is not validity conditional on all four realized sets. This induction also relies on scores/admissions not feeding back into the external stream, fixed hyperparameters, and role-independent preprocessing.

M0 reuses C throughout. Then A1/A2/M depend on C, and conditioning only on the resulting memory does not restore independent/exchangeable C and query scores. Recomputing calibration under the same memory is insufficient by itself. Disjoint C roles remove this particular path, but do not prove the actual protocol valid.

## Gaps for the paper's actual sampling and development history

1. Four calibration examples per class are stratified, not n i.i.d. mixture draws. A pooled query rank is not automatically exchangeable with that array, even if nominal class frequencies match. Our iid simulation and balanced-calibration simulation are distinct. We provide no exact pooled-rank theorem for the paper's sampling law.
2. Selecting theta with the same development populations and reusing examples later for calibration can invalidate independence from theta. The existing 80-shot pool has development history. Fifty redraws of that pool are an empirical robustness diagnostic, not fifty pristine holdouts.
3. ID shift, sorted class bursts, reactive arrivals and arbitrary temporal dependence are outside the iid-query statement. Good measured rates on some of these conditions do not extend the theorem.
4. CLIP/DINOv2 pretraining optimizes representation objectives, not kNN OOD separability or conformal validity. Frozen features do not guarantee the class-cluster assumption. TINS adapts a negative bank using the arriving batch and is not a p-value.
5. Per-arrival ID admission, ID fraction inside memory, final ID false-alarm rate, FDR among selected images, and simultaneous control over time are different probabilities. No FDR or anytime theorem for the full system is established here.

Even perfectly valid component p-values do not make their uncorrected product valid. For independent Uniform(0,1) p-values P1,P2, `P(P1*P2<=alpha)=alpha*(1-log(alpha))>alpha` for 0<alpha<1. Under perfect dependence P1=P2, the probability is sqrt(alpha)>alpha. Thus the product issue is distinct from reuse, solver symmetry and calibration sampling. REPRISE's product is evaluated only as a ranking score.

## Threshold/rank equivalence and resolution

### Counterexample: balanced calibration does not imply distribution-free pooled rank validity

This counterexample concerns the sampling rule alone, with a frozen scalar score, not a fitted REPRISE result. Use 900 equally likely ID classes and four independent calibration images per class. Let scores in 90 classes be Uniform(0,1), and scores in the other 810 classes be Uniform(2,3). All samples are independent within their class, with no ties. Calibration thus contains exactly 360 low and 3,240 high scores. A fresh equal-mixture ID query is low with probability .1.

At `alpha=361/3601`, every low query has pooled lower-tail p at most alpha. A high query meets that threshold exactly when it is smaller than all 3,240 high calibration scores, with probability `1/3241`. Hence

`P(p<=alpha)=.1+.9/3241 = .1002776921 > 361/3601 = .1002499306`.

The violation is small but exact. Thus fixed four-per-class calibration and correct ID class proportions alone are insufficient for a finite-sample distribution-free pooled-rank theorem. This does **not** show that REPRISE violates validity at .01, .05 or .10; those levels are separately measured. It rules out silently treating the balanced design as iid calibration for all alpha.

For OOD-high scores, `p(x)<=eps` iff `#{cal >= score(x)} <= floor(eps*(n+1))-1`. This is a conservative empirical-quantile rule with the stated strictness at ties. If floor(eps*(n+1))=0, no sample can be admitted. The p-value name does not create a new decision rule.

Primary 900-class calibration has n=3600, minimum p=1/3601. M1 uses n=900 per role, minimum 1/901, with no additional labels. The exact serialized epsilon3 is 0.10191613435745239; the displayed nominal value is 0.102. Simulations use n=100 (M0) or n=25 (M1), so M1 cannot issue p<=.01. Zero rejection at such a level is a resolution constraint, not evidence of useful detection.

For the primary M0 implementation, the serialized epsilon3 is very slightly **below** 367/3601. Therefore its exact `<=` rule admits ranks up to 366, giving grid level 366/3601, whereas literal .102 would admit rank 367. M1's exact entrance grid level is 91/901. Both paper-display and implementation levels are retained in `reports/calibration_resolution.csv`; the experiment reproduces the serialized value. This quantization is distinct from whether a stochastic guarantee holds.
