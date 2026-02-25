# SuperNeo Lean Cross-Check (Standalone)

This folder is intentionally outside `crates/` and independent from the Rust workspace.

It provides a Lean implementation of core SuperNeo/Neo math checks and verifies them
against vectors generated directly from Rust (`neo-math`).

## What is checked

- SuperNeo inner-product transform identity:
  - `ct(cf_inv(bar(a)) * cf_inv(b)) == dot(a, b)`
- Ring multiplication in `R_q = F_q[X]/(X^54 + X^27 + 1)`
- Centered `l_inf` norm on field/ring elements
- Balanced `split_b` decomposition and recomposition
- `eq(x,y)` polynomial behavior (Boolean and non-Boolean inputs)
- MLE identity via inner-product form vs. folding form
- Definition 7 coefficient embedding round-trips (element/vector/matrix)
- Definition 8 bar-lift transform checks (vector and matrix forms)
- Theorem 4 computational identity checks: `Mz = ct(bar(M)z)`
- Remark 2 evaluation/`ct` linkage checks
- Theorem 5 linear-combination evaluation homomorphism checks
- Module-homomorphism sanity checks used by evaluation homomorphism
- Theorem 8 assumption boundary + concrete precondition checks
- Definition 17/Theorem 9 strong-sampling + expansion-factor checks
- Appendix C Lemma 6-style eq-lifting table checks (+ SZ bound interface sanity)
- Neo-style polynomial interpolation/evaluation helpers (base field)

## Layout

- `SuperNeo/Field.lean`: Goldilocks modular arithmetic implementation
- `SuperNeo/Dimensions.lean`: concrete `eta`, `d`, and shape helpers
- `SuperNeo/Parameters.lean`: Appendix B.2 concrete parameter constants/sanity checks
- `SuperNeo/Ring.lean`: ring multiplication/reduction, `ct`, bar-block mat-vec
- `SuperNeo/CoeffMaps.lean`: `cf` / `cf^-1` map definitions and round-trips
- `SuperNeo/Norm.lean`: centered-representative norms
- `SuperNeo/Decomp.lean`: balanced base-`b` decomposition (`split_b`) helpers
- `SuperNeo/EqPoly.lean`: `eq` polynomial helpers
- `SuperNeo/MLE.lean`: multilinear-extension identities (`r_hat`, folding)
- `SuperNeo/Embedding.lean`: Definition 7 element/vector/matrix embeddings
- `SuperNeo/BarLift.lean`: Definition 8 blockwise lifting transform
- `SuperNeo/MatrixTransform.lean`: Theorem 4 computational transform identity
- `SuperNeo/EvalLink.lean`: Remark 2 coefficientwise evaluation linkage
- `SuperNeo/EvalHom.lean`: Theorem 5 computational evaluation homomorphism
- `SuperNeo/ModuleHom.lean`: module-hom interfaces + linearity sanity checks
- `SuperNeo/Thm3Core.lean`: P10/Theorem 3 core proposition + dimensional preconditions
- `SuperNeo/InvertibilityAxioms.lean`: Theorem 8 assumption boundary and concrete checks
- `SuperNeo/SamplingSet.lean`: Definition 17/Theorem 9 sampling-set and expansion checks
- `SuperNeo/PolyLemmas.lean`: reusable polynomial helpers for Lemma 5/6 style checks
- `SuperNeo/Interp.lean`: polynomial eval + interpolation
- `SuperNeo/P20.lean`: first P20 arithmetic-obligation composition skeleton theorem
- `SuperNeo/P21.lean`: first protocol-target theorem shell derived from P20
- `SuperNeo/ProtocolRelations.lean`: protocol context/claim/witness relation predicates
- `SuperNeo/ProtocolReduction.lean`: final medium-term theorem skeletons (`..._of_props`, `..._of_checks`)
- `SuperNeo/Checks.lean`: cross-check predicates against generated vectors
- `SuperNeo/Generated/Vectors.lean`: Rust-generated constants (bar matrix + cases)
- `rust-vectors/`: standalone Rust generator crate

## Regenerate vectors from Rust

```bash
cargo run --manifest-path formal/superneo-lean/rust-vectors/Cargo.toml
```

## Run Lean checks

```bash
cd formal/superneo-lean
lake build
lake exe check
```

Expected output ends with:

```text
all_checks=true
```

## Check Output Breakdown (`lake exe check`)

`lake exe check` reports 21 atomic checks plus one aggregate gate.
`true` means the corresponding executable predicate passed on all configured inputs.
This is stronger than unit smoke tests, but weaker than full universal theorem proofs.

| Output flag | `true` means (exactly what passed) | Evidence type | Remaining gap to a full SuperNeo proof |
|---|---|---|---|
| `superneo_cases` | For every generated `(a,b)`: `ct(mulRq(superneoBarBlock(bar,a), b))`, `dot(a,b)`, and expected values all agree. | Rust-generated vectors + Lean recomputation | Prove identity for all valid inputs, not only sampled/generated cases. |
| `ring_mul_cases` | `mulRq a b` matches expected coefficient vectors for all generated multiplication cases. | Rust-generated vectors | Prove quotient-ring multiplication semantics universally. |
| `norm_cases` | `normInfCoeffs` equals expected norms on all generated norm cases. | Rust-generated vectors | Prove general norm properties/bounds used in later theorems. |
| `split_cases` | `splitBalancedVec` digits match expected, recomposition equals expected and original input, and per-digit bounds hold. | Rust-generated vectors + invariant check | Prove reconstruction and bound theorems for all inputs. |
| `eq_cases` | `eqPoly x y` matches expected; Boolean points also satisfy indicator behavior check. | Rust-generated vectors + Boolean invariant | Prove full hypercube-indicator theorem. |
| `mle_cases` | Inner-product MLE and folding MLE both match expected and match each other (`mleIdentity`). | Rust-generated vectors + identity check | Lift to quantified theorem over all vectors/points. |
| `embedding_vec_cases` | `embedVec` matches expected blocks and `unembedVec (embedVec v) = v`. | Rust-generated vectors + round-trip invariant | Prove embedding bijection/linearity generally. |
| `embedding_matrix_cases` | `embedMatrix` matches expected blocks and `unembedMatrix (embedMatrix M) = M`. | Rust-generated vectors + round-trip invariant | Prove matrix-level embedding theorems generally. |
| `bar_lift_vec_cases` | Bar-lift outputs for `v`, `w`, `v+w`, `s*v` match expected and satisfy add/scale linearity checks. | Rust-generated vectors + linearity invariant | Prove Definition 8 properties for all vectors/scalars. |
| `bar_lift_matrix_cases` | `barLiftMatrix` matches expected lifted matrices on all generated cases. | Rust-generated vectors | Prove matrix lift correctness and algebraic properties generally. |
| `matrix_transform_cases` | `matrixVecDirect M z`, `matrixVecCtBar bar M z`, and expected vectors all agree; identity predicate also holds. | Rust-generated vectors + identity check | Formalize Theorem 4 universally from lower lemmas. |
| `eval_link_cases` | Evaluation-link computations (`evalRingVec`, `ct`, expected outputs) agree and `evalLinkIdentity`/`evalLinkForMatrix` checks pass. | Rust-generated vectors + identity checks | Replace computational checks with quantified Remark 2 proof. |
| `eval_hom_cases` | Evaluation homomorphism outputs (`Y1`, `Y2`, linear combo, direct combo) all match expected and each other; `evalHom2` holds. | Rust-generated vectors + homomorphism invariant | Prove full Theorem 5 algebraically. |
| `module_hom_cases` | `moduleHomSanity` passes add/scale preservation for representative concrete homomorphisms. | Fixed sanity witnesses (not generated) | Prove abstract module-hom lemmas, not only witness instances. |
| `invertibility_cases` | Concrete parameter preconditions for low-norm invertibility interface are satisfied. | Deterministic constant checks | Replace axiom boundary with proved invertibility theorem (Theorem 8 core). |
| `sampling_cases` | Strong-sampling predicate, max norm, bound, empirical expansion, and `empirical <= bound` all match expected/hold. | Rust-generated vectors + bound check | Prove Theorem 9 bound universally over required set class. |
| `eq_lift_cases` | `eqLiftFromTable` matches expected sums; Boolean-point behavior matches expected values when applicable. | Rust-generated vectors + Boolean-point check | Prove Appendix C eq-lifting lemmas for all tables/points. |
| `poly_lemma_cases` | `polyLemmaSanity` passes (`eqLiftAllBoolean` on a sample table + SZ interface condition). | Fixed sanity witnesses | Prove Schwartz-Zippel and related lemmas in general form. |
| `coeff_map_cases` | Coefficient-map round-trip checks pass on superneo/ring-generated data; additional sanity predicates pass. | Mixed: generated data + sanity predicates | Complete formal inverse/linearity proofs for `cf`, `cf^-1`, `ct`. |
| `parameter_cases` | Shape sanity, concrete parameter sanity, and norm sanity predicates all hold. | Deterministic constant/invariant checks | Prove all downstream inequalities that depend on these constants. |
| `interp_cases` | Interpolation coefficients and evaluation at a test point match expected values for all generated interpolation cases. | Rust-generated vectors | Prove interpolation correctness/uniqueness generally. |
| `all_checks` | Logical conjunction of every check above is `true`. | Aggregate gate | No new math content; only reports that all current executable checks passed. |

## Proof Dependency Map (`P1..P21`)

`P1..P21` are planning milestones (not paper numbering) to make dependency flow explicit.

```text
                    ALGEBRA TRACK                NORM TRACK             POLY TRACK
               ------------------           -----------------       ------------------

               P1 --> P3  Coeff maps        P1 --> P5  Norm        P1 --> P7  Eq poly
               P1 --> P4  Ring arith                |                      |
               P3 --> P9  Embedding                 v                      v
                       |                   P5 --> P6  split_b       P7 --> P8  MLE
                       v                                                    |
               P10 Theorem 3 core <-- P2      P2,P5 -> P16  Theorem 8      v
                       |                              |             P8 --> P18  SZ
                       v                              v                      |
                  P11  Def 8 lift             P16 --> P17  Theorem 9         v
                       |                                             P18 --> P19  Interp
                       v
                  P12  Theorem 4 identity
                       |
                       v                     +---- P3 --> P15  Module hom
                  P13  Remark 2 <--- P3      |
                       |                     |
                       +-------- + P15 --> P14  Theorem 5 eval hom


               =====================================================================
                P20  Arithmetic obligations  <-- P6, P12, P14, P16, P17, P18, P19
                 |
                 v
                P21  Protocol theorem target <-- P10, P12, P14, P16, P17, P18, P19, P20
               =====================================================================
```

| Proof ID | Lean modules | Core claim target | Depends on | Enables | Current evidence |
|---|---|---|---|---|---|
| `P1` | `Field.lean`, `Dimensions.lean` | Concrete base field/ring dimensions are correctly instantiated. | - | `P3`, `P4`, `P5`, `P7` | Implemented + used by all checks. |
| `P2` | `Parameters.lean` | Appendix B.2 constants and inequalities are fixed concretely. | - | `P10`, `P16` | Parameter sanity checks pass. |
| `P3` | `CoeffMaps.lean`, `Ring.lean` (`ct`) | Coefficient maps and constant-term bridge are algebraically sound. | `P1` | `P9`, `P10`, `P13`, `P15` | Round-trip/sanity checks pass; theorem API now includes explicit shape/`ct`/`mulRq` compatibility under `cf`/`cfInv`; full linearity layer still pending. |
| `P4` | `Ring.lean` | Quotient-ring multiplication semantics are correct. | `P1` | `P10` | Rust parity checks pass; size/shape and extraction semantics strengthened; universal reduction proof pending. |
| `P5` | `Norm.lean` | Centered `l_inf` norm definition and bounds behave as required. | `P1` | `P6`, `P16`, `P17` | Rust parity + sanity pass; theorem layer now includes reusable entry/row bound constructors plus challenge-coefficient subtraction bounds (`coeffSub`, `normInfCoeffs_le_four_of_allChallenge_sub`). |
| `P6` | `Decomp.lean` | `split_b` recomposes exactly with per-digit bounds. | `P5` | `P20` | Rust parity + invariants pass; quantified theorem pending. |
| `P7` | `EqPoly.lean` | `eq` behaves as Boolean selector on hypercube points. | `P1` | `P8`, `P18`, `P19` | Rust parity + indicator checks pass; full theorem pending. |
| `P8` | `MLE.lean` | `\tilde v(r)=<v,\hat r>` and folding form are equivalent. | `P7` | `P18` | Rust parity + identity checks pass; quantified theorem pending. |
| `P9` | `Embedding.lean` | Definition 7 embedding is a vector/matrix bridge with round-trip properties. | `P3` | `P10`, `P11` | Rust parity + round-trip checks pass; proof layer pending. |
| `P10` | `Thm3Core.lean`, `Ring.lean`, `Checks.lean` (`superneo_cases`) | Theorem 3 core inner-product transform identity. | `P1`, `P2`, `P3`, `P4`, `P9` | `P11`, `P12`, `P21` | In progress (check/prop equivalence plus theorem-native `thm3CoreAssumption` interface added; full universal proof pending). |
| `P11` | `BarLift.lean` | Definition 8 lifted transform is correct and linear. | `P9`, `P10` | `P12` | Rust parity + linearity checks pass; proposition-level linearity wrappers and explicit theorem-native additivity/homogeneity assumption interfaces are now added (`p11AdditivityAssumption`, `p11HomogeneityAssumption`), plus structural lemmas (`chunkBlocks_size`, `barLiftVec_singleBlock`). |
| `P12` | `MatrixTransform.lean` | Theorem 4 matrix-vector transform identity. | `P10`, `P11` | `P13`, `P14`, `P20`, `P21` | Rust parity + identity checks pass; theorem proof pending. |
| `P13` | `EvalLink.lean` | Remark 2 coefficientwise evaluation/`ct` linkage holds. | `P3`, `P12` | `P14` | Rust parity + identity checks pass; quantified proof pending. |
| `P14` | `EvalHom.lean` | Theorem 5 evaluation homomorphism for linear combinations. | `P12`, `P13`, `P15` | `P20`, `P21` | Rust parity + homomorphism checks pass; theorem proof pending. |
| `P15` | `ModuleHom.lean` | Abstract module-hom linearity obligations are available. | `P3` | `P14` | Witness sanity checks pass; abstract theorem layer pending. |
| `P16` | `InvertibilityAxioms.lean` | Theorem 8 invertibility preconditions and interface boundary. | `P2`, `P5` | `P17`, `P20`, `P21` | Preconditions proven computationally; theorem-native challenge-subtraction window/invertibility bridges added; core invertibility still axiomized. |
| `P17` | `SamplingSet.lean` | Definition 17 + Theorem 9 expansion-factor interface. | `P5`, `P16` | `P20`, `P21` | Rust parity + bound checks pass; universal theorem pending. |
| `P18` | `PolyLemmas.lean` | Eq-lifting and Schwartz-Zippel helper lemmas. | `P7`, `P8` | `P19`, `P20`, `P21` | Sanity and table checks pass; full general lemmas pending. |
| `P19` | `Interp.lean` | Interpolation/evaluation correctness and consistency. | `P7`, `P18` | `P20`, `P21` | Rust parity checks pass; uniqueness/proof lemmas pending. |
| `P20` | `P20.lean` + `Checks.lean` | Arithmetic side-conditions needed by protocol-level reduction compose cleanly. | `P6`, `P12`, `P14`, `P16`, `P17`, `P18`, `P19` | `P21` | In progress (both proposition-native and check-driven constructors are implemented; theorem-native matrix row-compatibility (`MatrixRowsCompatible`) is now included and bridged to/from checks). |
| `P21` | `P21.lean`, `ProtocolRelations.lean`, `ProtocolReduction.lean` | End-to-end SuperNeo protocol theorem from completed math stack. | `P10`, `P12`, `P14`, `P16`, `P17`, `P18`, `P19`, `P20` | Final claim | In progress (hardened protocol relation layer completed: shape/arithmetic split now uses theorem-native row compatibility, CE witness/norm obligations, and `..._of_props`/`..._of_checks` skeleton composition; full reduction theorem still pending). |

### Tracked Status and Exit Criteria

| Proof ID | Status now | Missing now | Exit criteria |
|---|---|---|---|
| `P1` | In progress (dimension/shape now has proposition, soundness, canonical equalities, and field-level canonical-value lemmas such as `Canonical`, `ofNat_val_eq_of_canonical`, and canonical identities for `0/1`). | Missing richer algebraic rewrite lemmas across matrix/vector ops. | Downstream modules consume theorem-native shape and field canonicality lemmas directly (minimal ad-hoc shape hypotheses). |
| `P2` | In progress (added direct theorem-native concrete parameter witness, not only check soundness). | Some downstream arguments still rely on composite assumptions rather than extracted theorem APIs. | Parameter inequalities used by P10/P16/P20 come from theorem constants, not check-derived bridges. |
| `P3` | In progress (round-trip theorem now available directly, plus bool bridge and explicit shape/`ct`/`mulRq` compatibility lemmas for `cf`/`cfInv`). | Missing explicit linearity lemmas on map composition used in evaluation proofs. | Complete theorem API for inverse + linearity + `ct` interaction used by P10/P13/P15. |
| `P4` | In progress (`mulRq_size`, `hasRingDegreeShape_mulRq`, shape-check completeness, `schoolbookRaw_size`, `superneoBarBlock_size`, and canonicality-preservation lemmas for `set!/addAt/subAt/ct/takeFirstD` are theorem-native; check bridge kept only for regression). | Missing universal algebraic correctness proof of reduction semantics itself. | Theorem-level ring semantics (not only shape/size) sufficient for P10/P12 derivations. |
| `P5` | In progress (added theorem-native bounds `normInf* <= q`, `normInf* <= halfQ`, challenge-coefficient predicates, reusable entry/row bound combinators, and vector/matrix + subtraction bounds `<= 4` for all-challenge inputs via `coeffSub`). | Need broader compositional inequalities (e.g. add/scale/product-sensitive bounds) beyond challenge-coefficient regimes. | Norm obligations in P16/P17 discharged from theorem lemmas rather than check-only side conditions. |
| `P6` | In progress (both `splitRoundTrip_sound` and `splitRoundTrip_complete` are now available). | Reconstruction/bound statements still need richer decomposition lemmas beyond round-trip/check equivalence. | Universal decomposition theorem with bound guarantees and direct reuse in P20/P21 without check wrappers. |
| `P7` | In progress | Hypercube-indicator theorem still check-backed. | Full theorem for Boolean selector behavior of `eq` over `{0,1}^ell`. |
| `P8` | In progress | MLE equivalence still from executable checks/sanity predicates. | Quantified theorem equating inner-product and folding MLE formulations. |
| `P9` | In progress | Embedding correctness not fully theorem-native. | General embedding/unembedding bijection + linearity theorem suite. |
| `P10` | In progress (`p10CoreCheck_sound` + `p10CoreCheck_complete` give check/prop equivalence and `thm3CoreAssumption` + theorem-native precondition constructor are now available). | Universal derivation of the assumption from P1-P4/P9 lemmas is still missing. | Prove Theorem-3 core directly from P1-P4/P9 lemmas, keep checks only as regression path. |
| `P11` | In progress (added proposition-level linearity/matrix interfaces, bidirectional check/prop bridges, assumption-driven theorem APIs `..._of_assumption(s)` for add/scale/combined lift linearity, structural decomposition lemmas for chunking/single-block reduction, and universal check-assumption -> theorem-assumption conversion bridges). | Core linearity equalities are still not yet derived from lower algebraic lemmas. | Prove lift linearity/correctness directly from embedding/ring lemmas and keep checks as regression only. |
| `P12` | In progress (`dotVec_eq_dot_of_isDVec` is available; `MatrixRowsCompatible` and sound/complete bridges are now theorem-native so row guards can be consumed as props). | Matrix identity proof not yet fully derived from theorem stack. | Full theorem proof from P10/P11 lemmas without case assumptions. |
| `P13` | In progress | Eval-link remains check-oriented. | Quantified Remark-2 linkage theorem for all valid inputs. |
| `P14` | In progress | Eval-hom proof path still leans on check soundness. | Full Theorem-5 proof via P12/P13/P15 theorem interfaces. |
| `P15` | In progress (sound + complete bridges now exist for add/scale checks on vec/scalar homs). | Still missing richer abstract algebra lemmas beyond direct check equivalence. | Complete module-hom theorem API used directly by P14. |
| `P16` | In progress (critical boundary identified; precondition bridge in P20 now uses theorem constants directly, invertibility-window bool has sound/complete theorem bridges, direct invertibility extraction lemmas are available, and challenge-coefficient norm bounds now imply `withinInvertibilityWindow` for both direct vectors and `coeffSub` differences). | `lowNormInvertibility` remains an axiom. | Replace/justify axiom boundary with theorem or explicit trusted interface plus documented assumptions. |
| `P17` | In progress | Expansion guarantees still check-driven. | Universal sampling expansion theorem wired into P20/P21 proof path. |
| `P18` | In progress | SZ/eq-lift results are partial helper checks. | Quantified polynomial lemma set (SZ + eq-lift) used directly by P19/P20. |
| `P19` | In progress | Interpolation correctness/uniqueness not fully formalized. | Full interpolation theorem package used in P20 without check-only assumptions. |
| `P20` | Good shell | Lower obligations still mixed theorem/check-backed (invertibility is theorem-native; proposition→check subset bridges now include P6/P12/P15/P17/P18/P19 + P10-core at protocol reduction smoke level). | Keep dual APIs, but establish proposition-native constructor from theorem-only premises and keep check APIs as regression wrappers. |
| `P21` | Good shell | Final reduction still a shell over unresolved lower-level gaps. | End-to-end protocol-facing CE theorem with explicit assumptions and theorem-native dependencies (including row-shape compatibility carried as `MatrixRowsCompatible`). |

## Math Breakdown (Current Status)

Source references:
- `docs/superneo-paper/04_4_Preliminaries.md`
- `docs/superneo-paper/05_5_Embedding_products_with_evaluation_homomorphism.md`
- `docs/superneo-paper/11_B_Concrete_parameters.md`
- `docs/superneo-paper/12_C_Additional_Background.md`
- `docs/superneo-paper/13_D_Deferred_theorems_and_proofs.md`

| ID | Math item (paper) | Lean target file | Lean work item | Rust parity hook | Connection to SuperNeo | Current status |
|---|---|---|---|---|---|---|
| M1 | Definition 1 (field/ring/dimension setup) | `SuperNeo/Field.lean` + `SuperNeo/Dimensions.lean` | Fix concrete instantiation and basic structural lemmas | `neo-math` field/ring constants + generated shape checks | Base algebra and shapes used by every SuperNeo identity and theorem statement. | In progress (concrete setup implemented; formal lemmas pending) |
| M2 | Appendix B.2 concrete Goldilocks parameters | `SuperNeo/Parameters.lean` | Encode exact constants and bound checks | generated parameter sanity checks | Pins SuperNeo to the exact concrete parameter regime claimed in the paper. | In progress (constants/sanity implemented; proof layer pending) |
| M3 | Definition 2 (`cf`, `cf^-1`, `ct`) | `SuperNeo/CoeffMaps.lean` + `SuperNeo/Ring.lean` | Prove inverse/linearity properties for maps | `neo_math::cf`, `cf_inv`, `ct` | These maps bridge coefficient and ring views that SuperNeo repeatedly switches between. | In progress (round-trip + shape/`ct`/`mulRq` compatibility implemented; linearity lemmas pending) |
| M4 | Ring arithmetic in `F[X]/(X^54 + X^27 + 1)` | `SuperNeo/Ring.lean` | Prove reduction semantics and arithmetic sanity lemmas | `neo_math::Rq::mul` | Core multiplication law behind bar-transform, lift, and matrix-product equalities. | In progress (implemented + parity passing) |
| M5 | Definition 3 (centered `l_inf` norm) | `SuperNeo/Norm.lean` | Define centered representatives and prove basic norm bounds | `neo_math::inf_norm` + generated norm vectors | Norm bounds are prerequisites for low-norm assumptions and soundness-side constraints. | In progress (implementation + parity passing; base bounds, reusable entry/row bound combinators, and challenge-coefficient subtraction bounds now implemented) |
| M6 | `split_b` decomposition math | `SuperNeo/Decomp.lean` | Prove reconstruction and per-digit bound | generated Rust `splitCases` vectors | Needed for bounded digit decompositions used in SuperNeo’s concrete arithmetic arguments. | In progress (implementation + parity passing; proofs pending) |
| M7 | `eq` polynomial on Boolean hypercube | `SuperNeo/EqPoly.lean` | Prove indicator behavior on `{0,1}^ell` | generated Rust `eqCases` vectors | Supplies the selector polynomial used by MLE and sumcheck-style reasoning in SuperNeo. | In progress (implementation + parity passing; formal theorem pending) |
| M8 | MLE identity `tilde_v(r) = <v, r_hat>` | `SuperNeo/MLE.lean` | Prove equivalence of two MLE formulations | generated Rust `mleCases` vectors | Connects table view and folded view of evaluations used in SuperNeo reductions. | In progress (implementation + parity passing; proof pending) |
| M9 | Definition 7 coefficient embedding | `SuperNeo/Embedding.lean` | Prove element/vector/matrix embedding bijection | generated Rust embedding vectors/matrices | Embedding is the structural bridge from field objects to ring objects in SuperNeo. | In progress (implementation + parity passing; proof pending) |
| M10 | Theorem 3 inner-product transform | `SuperNeo/Ring.lean` + `SuperNeo/Checks.lean` | Prove `ct(bar(a)*bar(b))=<a,b>` for concrete bar matrix | `neo_math::superneo_bar_matrix`, `superneo_bar_block` | Key algebraic equivalence that powers efficient ring-domain product computations in SuperNeo. | In progress (numeric checks passing; formal theorem pending) |
| M11 | Definition 8 lifting transform | `SuperNeo/BarLift.lean` | Prove blockwise lift properties and linearity | generated Rust `barLiftVecCases` / `barLiftMatrixCases` | Lifting composes embeddings with bar-transform to map full vectors/matrices into ring form. | In progress (implementation + parity passing; proposition-level linearity/matrix interfaces, check/prop bridges, assumption-driven theorem interfaces, single-block structural reduction lemma, and check-assumption->theorem-assumption conversion layer added; core proof pending) |
| M12 | Theorem 4 + App D.1 (`Mz = ct(bar(M)z)`) | `SuperNeo/MatrixTransform.lean` | Row/block proof from Theorem 3 | generated Rust `matrixTransformCases` | Establishes matrix-vector equivalence that underlies SuperNeo’s computational shortcut. | In progress (implementation + parity passing; proposition-level row-shape interfaces and sound/complete bridges added; full proof pending) |
| M13 | Remark 2 evaluation/ct linkage | `SuperNeo/EvalLink.lean` | Prove coefficientwise scaling and ct-eval link | generated Rust `evalLinkCases` | Connects coefficient evaluation with ct, enabling later evaluation-homomorphism proofs. | In progress (implementation + parity passing; formal proof pending) |
| M14 | Theorem 5 + App D.2 evaluation homomorphism | `SuperNeo/EvalHom.lean` | Prove linear-combination preservation under evaluation | generated Rust `evalHomCases` | Gives the homomorphic evaluation property used to justify transformed linear algebra steps. | In progress (implementation + parity passing; formal proof pending) |
| M15 | Definition 15 module homomorphisms | `SuperNeo/ModuleHom.lean` | Abstract module-hom lemmas reused by M14 | module-hom linearity sanity checks | Provides the abstract linearity framework that Theorem 5 instantiates. | In progress (interfaces + sanity checks implemented; theorem layer pending) |
| M16 | Theorem 8 low-norm invertibility | `SuperNeo/InvertibilityAxioms.lean` | Add assumption/axiom boundary and concrete precondition checks | Appendix B.2 constants + D.7 bound interface | Captures the invertibility condition required by SuperNeo’s soundness-critical reduction step. | In progress (assumption boundary + preconditions implemented; theorem bridges from all-challenge direct and subtraction norms to window/invertibility added) |
| M17 | Definition 17 + Theorem 9 (`C`, expansion factor) | `SuperNeo/SamplingSet.lean` | Formalize set conditions and expansion-factor theorem interface | generated Rust `samplingCases` | Formalizes sampling guarantees that control error amplification in SuperNeo analysis. | In progress (implementation + parity passing; formal theorem pending) |
| M18 | Appendix C Lemma 5/6 (Schwartz-Zippel and eq-lifting) | `SuperNeo/PolyLemmas.lean` | Add reusable polynomial lemmas for later proofs | generated Rust `eqLiftCases` + SZ interface sanity | Supplies probabilistic polynomial tools used in SuperNeo’s deferred proof chain. | In progress (implementation + parity passing; proof layer pending) |
| M19 | Polynomial interpolation/evaluation math | `SuperNeo/Interp.lean` | Prove interpolation correctness against sampled points | local Rust generator interpolation vectors | Supports recovery/consistency arguments for polynomial objects used in SuperNeo checks. | In progress (implementation + parity passing; proof lemmas pending) |
| M20 | Executable cross-check harness | `Main.lean` + `SuperNeo/Checks.lean` | Keep deterministic Rust-vs-Lean checks green | `rust-vectors/src/main.rs` generated vectors | Acts as the executable witness that Lean computations match SuperNeo Rust math instances. | Done (all checks currently pass) |

### Status Summary

| State | Count |
|---|---|
| Done | 1 |
| In progress | 19 |
| Partial | 0 |
| Not started | 0 |
