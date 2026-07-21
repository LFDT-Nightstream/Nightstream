# F′ / NIFS constraint-minimization plan

Status: two strict-PiDEC source-schedule reductions are implemented and proved;
their exact active source-row image is artifact-checked for the bounded
`kappa = 4` fixture. No final selective-row count or broader NIFS removal is
authorized by this document.

## Objective

Prove an independently specified SuperNeo/HyperNova F′/NIFS verifier sound
and complete, prove every retained semantic obligation necessary, refine that
verifier to the exact Rust/R1CS implementation, and remove only physical row
families whose absence is justified by those proofs.

“Minimum” has two mechanically checkable levels:

1. inclusion-minimal semantic obligations for a fixed protocol target; and
2. minimum retained row families for the selected concrete encoding.

A globally minimum algebraic circuit is a stronger claim and requires a lower
bound; exact counts or local countermodels alone do not establish it.

## Paper baseline

The production-selected edge is `K = 1`, `k = 14`, `b = 2`:

```text
1 CCS(b) + 14 CE(b)
  -- PiCCS --> 15 CE(b)
  -- PiRLC --> 1 CE(B)
  -- PiDEC --> 14 CE(b)
```

SuperNeo Section 7.3 owns fresh CCS truth, every source norm, and the fourteen
carried evaluation claims. Section 7.4 owns the fifteen strong-set challenges
and the common linear combination of commitment, public input, witness, and
evaluations. Section 7.5 owns decomposition back to fourteen low-norm claims.

HyperNova Construction 2 places the selected NIFS verifier inside `F′_j`; it
also owns program dispatch, the prior hash/link, the running-instance update,
the base/default branch, and the next public hash. Those lifecycle obligations
must not be smuggled into the inner SuperNeo relation.

## PiDEC specification correction

The existing Lean `PiDEC.Accepted` is a recomposition core, not the exact
Section-7.5 verifier. The paper verifier computes each child public input as
`split_b(parent.x)`. `PiDEC.Accepted` instead accepts any child public-input
family whose radix-weighted recomposition equals `parent.x`.

The correction is now implemented at model level:

- `PiDEC.PaperVerifier.Attempt` contains only the public parent and the prover's
  child commitment/evaluation messages;
- `PiDEC.PaperVerifier.children` computes child public inputs, copies
  structure/point, and fixes the fresh stage;
- `PiDEC.PaperVerifier.Accepted` enforces the paper's fixed `t`-tuple shape for
  the parent and every child message, then the two Section-7.5
  commitment/evaluation equations (plus the combined parent stage);
- the concrete Phi81 coordinatewise public split is proved to commute with
  public projection and to recompose exactly;
- the concrete Phi81 evaluation tuple is proved to contain exactly one value
  per structure matrix, and a trailing-value counterexample is rejected;
- exact paper acceptance projects theoremically to the old recomposition core,
  so its existing reduction-of-knowledge theorem remains reusable; and
- recomposition-only public inputs are a separate optimization candidate. They
  may be cheaper and sound, but row removal requires an independent
  completeness and knowledge-soundness theorem for that relaxed verifier.

The existing production child-substitution fixture changes a public child
coordinate. It witnesses the difference between the relaxed Lean predicate
and the paper verifier; it does not prove that the paper accepts that family.

The generic and concrete fixed-active paper profiles now use the operational
paper predicate. This correction is model-level only: it authorizes no Rust or
R1CS row removal by itself.

## Current proof frontier

Completed model-level foundations:

- independent CCS/CE relations and concrete Phi81 algebras;
- paper-shaped PiCCS, PiRLC, PiDEC, and composition theorems;
- fixed production parameters and the `1 + 14` source profile;
- finite SumCheck, sampler, transcript, commitment, encoding, and many exact
  generated-row correspondence slices;
- a fixed-candidate six-leaf paper plan exactly equivalent to its indexed
  realization, with one concrete countermodel for removing each leaf;
- an exact reduced strict-PiDEC compiler model whose soundness reaches the
  operational paper verifier and whose same-assignment completeness retains
  explicit deterministic auxiliary definitions;
- two proved source-schedule reductions: common-sign binary public digits and
  semantic-prefix evaluation recomposition, saving exactly 3,500 source-R1CS
  rows in the active profile; and
- a generated bounded active strict-PiDEC artifact with 11,845 exact source
  rows whose sparse coefficients match the independent reduced compiler and
  whose satisfaction implies operational paper PiDEC acceptance; and
- a kernel-checked active-result seam proving that the decoded strict carrier
  is exactly `FixedActive.resultOf` once the parent-point and ordered
  child-payload column bindings are supplied; the outgoing state theorem keeps
  delayed block-by-lane `y_zcol` as a separate value computed from that same
  accepted certificate; and
- generated row-family manifests and drift checks.

Not yet sufficient for minimization:

- the six-leaf result is macro-level inclusion minimality; it does not prove
  necessity of every internal PiDEC equation, source-binding coordinate, or
  physical row;
- PiCCS still has an unresolved paper `Q`/`T` exponent convention and the
  production two-SumCheck FE/NC flow lacks a complete equivalence/security
  bridge to the paper obligations;
- executable acceptance has not yet been shown to imply the independent NIFS
  relation or named bad events end-to-end;
- the flat-column `y_zcol` handoff is known not to commute with the PiRLC ring
  action, so the remaining bridge must use the expanded block-by-lane witness
  image used by the implementation; and
- the generated strict-PiDEC source refinement is only the bounded `kappa = 4`
  fixture. Production `kappa = 18`, the final selective-row projection, and
  the two decoder bindings that identify its parent point and ordered child
  payloads with the lifecycle certificate remain open. The model seam proves
  these two facts are sufficient; it does not assume or manufacture them from
  source-row satisfaction. The next artifact must therefore cross phase
  boundaries: it must carry the PiCCS `r_prime` columns and point-binding rows,
  ordered child commitment/public-input/evaluation provenance, and (for a
  final-R1CS theorem) source-to-selective decoder provenance. PiDEC
  recomposition and outgoing digests cannot substitute for these bindings.

## Work plan and exit gates

### 1. Exact paper relations — model-level complete

The paper PiDEC public-input splitter, fixed evaluation arity, and operational
acceptance predicate are implemented, with honest completeness, reduction of
knowledge, projection to the existing recomposition core, and generic/concrete
fixed-active profile integration. HyperNova sidecars remain isolated from this
relation.

Exit gate met: kernel-checked exact paper acceptance has no Rust/R1CS import
and no compiler-trusted decision procedure in its semantic theorems.

### 2. Fixed-candidate obligation plan — model-level complete

Use fixed source data, target, point, and challenges so existential witness
substitution cannot rescue a forgery. Start with six macro obligations:

1. fresh CCS truth;
2. strict norm for every source;
3. all carried evaluations are true;
4. the public source product binds the complete authoritative source family;
5. every PiRLC challenge belongs to the strong set; and
6. exact paper PiDEC acceptance for the computed PiRLC parent.

Common structure/point facts derived from source binding are not leaves. PiCCS
outputs and the PiRLC parent are computed intermediates, not independent
authorities. Split the aggregate source-binding and PiDEC leaves further when
mapping physical rows.

Exit gate met: `ObligationPlan.accepts_iff_target` proves the six-leaf
conjunction equivalent to the fully indexed paper realization.

### 3. Necessity and derived-field elimination — macro level complete

For each retained leaf, remove only that leaf and exhibit an invalid candidate
accepted by the remainder. Separately prove every eliminated field from
retained data or construct it directly. Necessity of a semantic equality does
not imply that a separate R1CS equality row is necessary.

Macro exit gate met: every one of the six leaves has a concrete weakened-plan
countermodel and `PaperSemanticMinimality.inclusionMinimalSound` closes the
ledger. Per-equation and physical-family refinement remains part of step 6.

### 4. Prove candidate protocol relaxations

Define relaxed PiDEC public recomposition separately from paper acceptance.
Prove or reject completeness and reduction-of-knowledge for the relaxed
verifier, including the production full-public projection profile. Apply the
same discipline to the FE/NC split, cached constant terms, padding, delayed
points, digests, and any aggregated check.

Exit gate: every deviation from the paper has its own theorem showing the same
target relation/security conclusion, or it remains retained.

### 5. Executable and cryptographic closure

Prove finite SumCheck acceptance implies the three PiCCS truths or a named
bad-root/round event; bind Fiat-Shamir replay and sampler shortfall; prove the
PiRLC weak-extraction/uniqueness bridge; obtain valid PiDEC child openings or
a binding event; and compose the selected NIFS call with HyperNova lifecycle
and compatibility.

Exit gate: executable success implies the exact or justified-relaxed
fixed-active transition, or an explicit security event.

### 6. Rust/R1CS refinement and cost ownership

Decode exact generated rows into the reduced verifier, never the reverse.
Assign every emitted row to one retained obligation, derived field, encoding
requirement, or justified lifecycle check. Prove soundness and honest witness
construction for each family and record its exact cost.

Current manifests are prioritization data only. In the full-history fixture,
recursive NIFS is 827,866 rows (`PiCCS` 320,528; `PiRLC` 496,739; `PiDEC`
10,597; point binding 2), while terminal NIFS is 2,278,831 rows. The largest
families should be audited first after their semantic bridges close; these
counts do not define what is correct.

Exit gate: exact Rust success/R1CS satisfaction iff the reduced verifier (or
named event), with a complete protocol-to-row ownership tree.

The bounded active strict-PiDEC source slice now meets the forward direction
of this gate at `kappa = 4`: exact generated source rows imply the independent
reduced compiler and operational paper acceptance. This is artifact-checked
with `Lean.trustCompiler` at the sparse-data equality only. It does not cover
production `kappa = 18`, final selective rows, or certificate/FoldResult
decoder identity. The separate kernel-only active-result seam reduces that
last identity to exact parent-point and ordered child-payload decoder facts;
neither fact is present in the current generated artifact. A focused
cross-phase decoder artifact is sufficient; a second giant whole-circuit
artifact is not required.

### 7. Removal batches

Delete one proved-redundant family at a time, regenerate artifacts, rerun
soundness/completeness/necessity and drift gates, and record the before/after
count plus theorem that authorized the deletion.

Completed strict-PiDEC source batch: replace independent binary child-alphabet
checks by the sound and honestly complete shared-sign schedule (3,240 rows
saved), and omit padded evaluation recomposition already implied by retained
padding-zero rows (260 rows saved). The active source count is therefore
`54*kappa + 11,629`; at the bounded `kappa = 4` artifact it is 11,845. These
are source-R1CS counts, not final selective constraints.

Exit gate: no retained row lacks a semantic owner or necessity argument, and
no removed row was justified by matching old code or old counts.
