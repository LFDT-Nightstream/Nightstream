# SuperNeo Lean Cross-Check (Standalone)

This folder is intentionally outside `crates/` and independent from the Rust workspace.

It provides a Lean implementation of core SuperNeo/Neo math checks and verifies them
against vectors generated directly from Rust (`neo-math`).

It also supports the reverse direction for primitive conformance:
Lean-generated Goldilocks golden vectors consumed by Rust tests in `neo-math`.

## Bidirectional Golden-Vector Strategy

- Rust -> Lean: use for large generated fixtures (trace/layout/protocol-style vectors) where Rust is the practical fixture source.
- Lean -> Rust: use for spec-critical primitive conformance (Goldilocks field ops) where Lean is the reference model.
- Keep both directions: system-level parity checks in Lean, primitive-level conformance checks in Rust.

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
- `SuperNeo/GoldilocksGolden.lean`: Lean exporter for Goldilocks primitive golden vectors
- `SuperNeo/Generated/GoldilocksGolden.csv`: Lean-generated primitive conformance vectors for Rust
- `SuperNeo/RingGolden.lean`: Lean exporter for ring-multiplication golden vectors
- `SuperNeo/Generated/RingGolden.csv`: Lean-generated ring conformance vectors for Rust
- `SuperNeo/EqMleGolden.lean`: Lean exporter for eq-polynomial and MLE conformance vectors
- `SuperNeo/Generated/EqMleGolden.csv`: Lean-generated eq/MLE conformance vectors for Rust
- `SuperNeo/P9P11P12Golden.lean`: Lean exporter for embedding/bar-lift/matrix-transform conformance vectors
- `SuperNeo/Generated/P9P11P12Golden.csv`: Lean-generated P9/P11/P12 conformance vectors for Rust
- `rust-vectors/`: standalone Rust generator crate

## Regenerate Vectors From Rust (Rust -> Lean)

```bash
cargo run --manifest-path formal/superneo-lean/rust-vectors/Cargo.toml
```

## Regenerate Goldilocks Vectors From Lean (Lean -> Rust)

```bash
cd formal/superneo-lean
lake exe goldilocks-golden > SuperNeo/Generated/GoldilocksGolden.csv
```

## Regenerate Ring Vectors From Lean (Lean -> Rust)

```bash
cd formal/superneo-lean
lake exe ring-golden > SuperNeo/Generated/RingGolden.csv
```

## Regenerate Eq/MLE Vectors From Lean (Lean -> Rust)

```bash
cd formal/superneo-lean
lake exe eq-mle-golden > SuperNeo/Generated/EqMleGolden.csv
```

## Regenerate P9/P11/P12 Vectors From Lean (Lean -> Rust)

```bash
cd formal/superneo-lean
lake exe p9-p11-p12-golden > SuperNeo/Generated/P9P11P12Golden.csv
```

## Run Rust Conformance Tests Against Lean Goldens

```bash
cargo test -p neo-math --release goldilocks_matches_lean_golden_vectors
cargo test -p neo-math --release ring_mul_matches_lean_golden_vectors
cargo test -p neo-memory --release eq_mle_matches_lean_golden_vectors
cargo test -p neo-math --release p9_p11_p12_matches_lean_golden_vectors
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
| `invertibility_cases` | Concrete parameter preconditions for low-norm invertibility interface are satisfied. | Deterministic constant checks | Keep invertibility dependency explicit via assumption-parameterized interface (no global theorem axiom). |
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
| `P2` | `Parameters.lean` | Appendix B.2 constants and inequalities are fixed concretely. | - | `P10`, `P16` | Theorem-native constant APIs are now exposed (`eta/d/b/k/B/T/...`) and consumed by downstream precondition lemmas; check sanity remains as regression. |
| `P3` | `CoeffMaps.lean`, `Ring.lean` (`ct`) | Coefficient maps and constant-term bridge are algebraically sound. | `P1` | `P9`, `P10`, `P13`, `P15` | Round-trip/sanity checks pass; theorem API now includes explicit shape/`ct`/`mulRq` compatibility under `cf`/`cfInv` plus linearity over vector add/scale (`cf_vecAdd`, `cfInv_vecAdd`, `cf_vecScale`, `cfInv_vecScale`) and multiplication compatibility bridges (`cf_mulRq_cfInv`, `cfInv_mulRq_cfInv`, `ct_mulRq_cfInv`). |
| `P4` | `Ring.lean` | Quotient-ring multiplication semantics are correct. | `P1` | `P10` | Rust parity checks pass; theorem-native closed-form `Phi_81` reduction semantics are exposed (`reducePhi81Coeff` cases + unified `reducePhi81Coeff_formula`, `mulRqCoeffSpec` + case lemmas, public raw-coefficient accessor/rewrite layer (`mulRqRawCoeffSpec`, `mulRqCoeffSpec_of_{le25,eq26,ge27}_raw`), `reducePhi81Coeff_eq_mulRqCoeffSpec`, `mulRq_eq_ofFn_coeffSpec`, `mulRq_ct_formula`, `mulRq_ct_formula_explicit`, and `mulRq_coeff_of_le25/eq26/ge27` + unified `mulRq_coeff_formula`), canonicality/idempotence bridges are in place (`reducePhi81Coeff_canonical`, `reducePhi81_allCanonical`, `mulRq_allCanonical`, `mulRq_coeff_canonical`, `mulRqCoeffSpec_canonical`, `reducePhi81_idempotent_of_shape_allCanonical`, `reducePhi81_mulRq_idempotent`), and quotient-normal-form completeness/uniqueness/extensionality is closed via `mulRqQuotientSpec` + `mulRq_eq_of_quotientSpec` + `mulRq_quotientSpec_iff` + `mulRqQuotientSpec_unique`, with direct rewrite constructors (`mulRqQuotientSpec_iff_size_and_mulRq_coeff`, `mulRqQuotientSpec_iff_size_and_fin_mulRq_coeff`, `mulRqQuotientSpec_iff_hasRingDegreeShape_and_fin_coeffSpec`, `mulRq_eq_of_size_and_mulRq_coeff`, `mulRq_eq_of_size_and_fin_mulRq_coeff`, `mulRq_eq_of_hasRingDegreeShape_and_coeffSpec`, `mulRq_eq_of_hasRingDegreeShape_and_fin_coeffSpec`) plus transport lemmas (`mulRqQuotientSpec_coeff_eq`, `mulRqQuotientSpec_getElem_eq_mulRq_getElem`, `mulRqQuotientSpec_getElem_eq_pair`, `ct_eq_of_mulRqQuotientSpec_pair`, `reducePhi81_eq_of_mulRqQuotientSpec_pair`, `reducePhi81_eq_mulRq_of_mulRqQuotientSpec`); explicit hardening layer added with `getCoeff`/`getElemBang_eq_getCoeff`, shaped quotient spec (`mulRqQuotientSpecShaped`), and typed ring wrapper `Rq` (`truncateToRing`, `Rq.mul`, `Rq.mulQuotientSpec{,Shaped}_iff`). |
| `P5` | `Norm.lean` | Centered `l_inf` norm definition and bounds behave as required. | `P1` | `P6`, `P16`, `P17` | Rust parity + sanity pass; theorem layer now includes reusable entry/row bound constructors, operation-sensitive bounds (`normInfF_{add,sub,mul}_le_{q,halfQ}`, `normInfCoeffs_{vecAdd,vecScale,mulRq,coeffSub}_le_{q,halfQ}`), direct SuperNeo-operation wrappers (`normInfCoeffs_superneoBarBlock_le_{q,halfQ}`, `normInfCoeffs_barLiftVec_le_{q,halfQ}`, `normInfCoeffMatrix_barLiftMatrix_le_{q,halfQ}`), strengthened challenge-compositional bounds (`normInfF_add_le_four_of_isChallengeCoeff`, `normInfF_mul_le_four_of_isChallengeCoeff`, `normInfCoeffs_le_four_of_allChallenge_{add,sub,scale}`), and compositional APIs at both entry/coefficient and operand-norm layers (`normInfCoeffs_entry_le`, `normInfCoeffs_le_of_hasRingDegreeShape_and_coeff_bound`, `normInfCoeffs_{vecAdd,vecScale,coeffSub}_le_of_entry_bound`, `normInfCoeffs_mulRq_le_of_coeffSpec_bound`, `normInfCoeffs_{vecAdd,vecScale,coeffSub,mulRq}_le_of_norm_bounds`, `normInfF_mulRqCoeffSpec_le_of_rawCoeffBound`, `normInfCoeffs_mulRq_le_of_rawCoeffBound`, `normInfCoeffs_mulRq_le_of_norm_bounds_via_rawCoeff`), plus assumption-free concrete raw-coefficient fallback bridges (`normInfF_add_sub_le_halfQ`, `normInfF_mulRqRawCoeffSpec_le_halfQ`, `normInfCoeffs_mulRq_le_of_norm_bounds_via_rawCoeff_halfQ`, `normInfCoeffs_mulRq_le_of_norm_bounds_via_rawCoeff_halfQ_le`). |
| `P6` | `Decomp.lean` | `split_b` recomposes exactly with per-digit bounds. | `P5` | `P20` | Rust parity + invariants pass; theorem layer now includes bool↔prop bridges for per-digit bounds and roundtrip (`digitsWithinBase_sound/complete/iff_prop`, `splitRoundTrip_sound_prop`, `splitRoundTrip_complete_prop`, `splitRoundTrip_iff_prop`), monotonicity helpers (`digitsWithinBaseProp_mono`, `digitsWithinBase_mono`), and direct quantified extraction lemmas from successful checks (`splitRoundTrip_base_ge_two`, `splitRoundTrip_recompose_eq`, `splitRoundTrip_digitsWithinBaseProp`, `splitRoundTrip_digit_bound`); fully constructive decomposition proofs still pending. |
| `P7` | `EqPoly.lean` | `eq` behaves as Boolean selector on hypercube points. | `P1` | `P8`, `P18`, `P19` | Rust parity + indicator checks pass; proposition-level bool↔prop bridges now available for Boolean-point predicate and indicator check (`isBoolF_sound/complete/iff_prop`, `eqHypercubeIndicatorProp`, `eqHypercubeIndicator_sound/complete/iff_prop`); full quantified theorem pending. |
| `P8` | `MLE.lean` | `\tilde v(r)=<v,\hat r>` and folding form are equivalent. | `P7` | `P18` | Rust parity + identity checks pass; proposition-level bool↔prop bridges are expanded (`mleIdentity_sound`, `mleIdentity_complete`, `mleIdentity_iff_prop`, explicit size-form bridge `mleIdentityPropEq_iff_prop`, and extraction lemmas `mleIdentity_size_eq`/`mleIdentity_eval_eq`); quantified theorem pending. |
| `P9` | `Embedding.lean` | Definition 7 embedding is a vector/matrix bridge with round-trip properties. | `P3` | `P10`, `P11` | Rust parity + round-trip checks pass; proposition-level bool↔prop bridges for vector/matrix round-trips are now available (`embeddingVecRoundTrip_sound/complete/iff_prop`, `embeddingMatrixRoundTrip_sound/complete/iff_prop`) plus direct extraction lemmas for shape/equality from successful checks (`embeddingVecRoundTrip_size_mod_eq_zero`, `embeddingVecRoundTrip_unembed_embed_eq`, `embeddingMatrixRoundTrip_rows_mod_ok`, `embeddingMatrixRoundTrip_unembed_embed_eq`); full theorem layer pending. |
| `P10` | `Thm3Core.lean`, `Ring.lean`, `Checks.lean` (`superneo_cases`) | Theorem 3 core inner-product transform identity. | `P1`, `P2`, `P3`, `P4`, `P9` | `P11`, `P12`, `P21` | In progress (check/prop equivalence plus theorem-native `thm3CoreAssumption` interface added; full universal proof pending). |
| `P11` | `BarLift.lean` | Definition 8 lifted transform is correct and linear. | `P9`, `P10` | `P12` | Rust parity + linearity checks pass; proposition-level linearity wrappers and explicit theorem-native additivity/homogeneity assumption interfaces are now added (`p11AdditivityAssumption`, `p11HomogeneityAssumption`), plus structural lemmas (`chunkBlocks_size`, `barLiftVec_singleBlock`). |
| `P12` | `MatrixTransform.lean` | Theorem 4 matrix-vector transform identity. | `P10`, `P11` | `P13`, `P14`, `P20`, `P21` | Rust parity + identity checks pass; proposition-level bool↔prop bridges (`matrixTransformIdentityProp`, `matrixTransformIdentity_iff_prop`) and theorem/check assumption interfaces (`p12MatrixTransformAssumption`, `p12MatrixTransformCheckAssumption`) are available, and a theorem-native closure from Theorem-3 assumptions is now wired (`matrixTransformEq_of_thm3CoreAssumption`, `p12MatrixTransformAssumption_of_thm3CoreAssumption`, `p12MatrixTransformCheckAssumption_of_thm3CoreAssumption`). |
| `P13` | `EvalLink.lean` | Remark 2 coefficientwise evaluation/`ct` linkage holds. | `P3`, `P12` | `P14` | Rust parity + identity checks pass; proposition-level bool↔prop bridges for matrix specialization are available (`evalLinkForMatrix_sound_full`, `evalLinkForMatrix_complete`, `evalLinkForMatrix_iff_prop`), theorem/check assumption interfaces include both global and fixed-`m,r` forms (`p13EvalLinkAssumption{,For}`, `p13EvalLinkCheckAssumption{,For}`), and an assumption-free theorem-native derivation is available at both identity and matrix levels (`evalLinkIdentityProp_of_size_eq`, `evalLinkIdentity_true_of_size_eq`, `evalLinkForMatrixProp_from_defs`, `evalLinkForMatrix_true_from_defs`, `p13EvalLinkAssumption_from_defs`, `p13EvalLinkCheckAssumption_from_defs`). |
| `P14` | `EvalHom.lean` | Theorem 5 evaluation homomorphism for linear combinations. | `P12`, `P13`, `P15` | `P20`, `P21` | Rust parity + homomorphism checks pass; proposition-level bool↔prop bridges (`evalHom2_sound_full`, `evalHom2_complete`, `evalHom2_iff_prop`) are now available; theorem-native assumption boundary (`p14EvalHomAssumption`) and preconditioned check-assumption bridge (`p14EvalHomCheckAssumption`) with bidirectional conversion are now wired; full theorem proof pending. |
| `P15` | `ModuleHom.lean` | Abstract module-hom linearity obligations are available. | `P3` | `P14` | Witness sanity checks pass; abstract theorem layer now includes aggregate check↔prop bridges for vector/scalar module linearity pairs (`vecModuleCheckPair/vecModulePropPair`, `scalarModuleCheckPair/scalarModulePropPair` with sound/complete/iff lemmas), and P20 now consumes these pair bridges directly in proposition/check conversions; broader algebraic theorem layer still pending. |
| `P16` | `InvertibilityAxioms.lean` | Theorem 8 invertibility preconditions and interface boundary. | `P2`, `P5` | `P17`, `P20`, `P21` | Preconditions proven computationally; theorem-native challenge-subtraction window/invertibility bridges added; explicit `..._of_assumption` APIs and protocol-context trusted boundary (`hLowNormInvertibility`) are now in place; core invertibility still axiomized. |
| `P17` | `SamplingSet.lean` | Definition 17 + Theorem 9 expansion-factor interface. | `P5`, `P16` | `P20`, `P21` | Rust parity + bound checks pass; universal theorem pending. |
| `P18` | `PolyLemmas.lean` | Eq-lifting and Schwartz-Zippel helper lemmas. | `P7`, `P8` | `P19`, `P20`, `P21` | Sanity and table checks pass; full general lemmas pending. |
| `P19` | `Interp.lean` | Interpolation/evaluation correctness and consistency. | `P7`, `P18` | `P20`, `P21` | Rust parity checks pass; uniqueness/proof lemmas pending. |
| `P20` | `P20.lean` + `Checks.lean` | Arithmetic side-conditions needed by protocol-level reduction compose cleanly. | `P6`, `P12`, `P14`, `P16`, `P17`, `P18`, `P19` | `P21` | In progress (both proposition-native and check-driven constructors are implemented; theorem-native matrix row-compatibility (`MatrixRowsCompatible`) is included and bridged to/from checks; P12 now also exposes theorem/check-assumption consumption paths in P20 (`p20ArithmeticBundle_of_props_with_matrixTransform_assumption`, `..._with_matrixTransform_checkAssumption`) plus direct Theorem-3-derived matrix-transform wiring (`p20ArithmeticBundle_of_props_with_thm3CoreAssumption`); direct Thm3+P14 assumption/check combined paths are now explicit (`p20ArithmeticBundle_of_props_with_thm3CoreAssumption_with_evalHom_assumption`, `..._with_evalHom_checkAssumption`); P12/P14 mixed-mode constructor quadrants are explicit (`...with_matrixTransform_assumption_with_evalHom_checkAssumption`, `...with_matrixTransform_checkAssumption_with_evalHom_assumption`) in addition to assumption/assumption and check/check paths; P14 threads via full `evalHom2Prop`; proposition→module-check recovery derives the size guard from P14 directly and reuses aggregate ModuleHom pair bridges (`vecModuleCheckPair_of_propPair`, `scalarModuleCheckPair_of_propPair`); explicit invertibility-window obligation `p20InvertibilityWindowProp invDelta` plus assumption-driven inverse extraction `p20InvertibilityWitness_of_assumption` are wired; and a full proposition↔check equivalence theorem for the backward-compatible check surface is available (`p20ArithmeticBundle_iff_checks`). |
| `P21` | `P21.lean`, `ProtocolRelations.lean`, `ProtocolReduction.lean` | End-to-end SuperNeo protocol theorem from completed math stack. | `P10`, `P12`, `P14`, `P16`, `P17`, `P18`, `P19`, `P20` | Final claim | In progress (hardened protocol relation layer completed: shape/arithmetic split uses theorem-native row compatibility, explicit `invDelta` invertibility-window threading, CE witness/norm obligations, `..._of_props`/`..._of_checks` skeleton composition, and `..._with_invertibility` protocol theorems exposing `∃ deltaInv`; mirrored assumption/check-assumption protocol reduction paths for P14 exist in both `ProtocolReduction` and direct `P21` constructors; P21 has direct P12-assumption entrypoints (`p21ProtocolTarget_of_props_with_matrixTransform_assumption`, `..._checkAssumption`, `..._with_matrixTransform_assumption_with_evalHom_assumption`) plus a direct Theorem-3-derived P12 protocol constructor (`p21ProtocolTarget_of_props_with_thm3CoreAssumption`); `ProtocolReduction` mirrors the same Theorem-3-derived P12 claim-level constructor (`p20ForClaim_of_props_with_thm3CoreAssumption`) and now also adds thm3-skeleton convenience variants that derive `hP12Eq` from `ClaimShapeValid` automatically for both evalHom assumption/check and with-invertibility paths; plus lean check-subset constructor/equivalence layers for protocol/full targets (`p21ProtocolTarget_of_check_subset`, `p21ProtocolTarget_iff_check_subset`, `p21FullMathTarget_iff_check_subset`) that avoid unnecessary module-check dependencies at protocol target level; full reduction theorem still pending). |

### Tracked Status and Exit Criteria

| Proof ID | Status now | Missing now | Exit criteria |
|---|---|---|---|
| `P1` | Done (dimension/shape proposition + soundness/canonical equalities are in place; named theorem-native shape projection lemmas and matrix/vector size rewrite lemmas are now added and consumed in protocol reduction paths). | None. | Downstream modules consume theorem-native shape and field canonicality lemmas directly (minimal ad-hoc shape hypotheses). |
| `P2` | Done (direct theorem-native constant APIs are exposed in `Parameters.lean` and P16/P20 precondition obligations consume those theorem constants directly; check booleans remain regression-only). | None. | Parameter inequalities used by P10/P16/P20 come from theorem constants, not check-derived bridges. |
| `P3` | Done (round-trip theorem and bool bridge are in place; explicit shape/`ct`/`mulRq` compatibility plus linearity lemmas on map composition are now theorem-native). | None. | Downstream modules can consume inverse + linearity + `ct` interaction via named theorem API (`cf/cfInv` add/scale + `mulRq`/`ct` compatibility) without ad-hoc rewrites. |
| `P4` | Done (`mulRq_size`, `hasRingDegreeShape_mulRq`, shape-check completeness, `schoolbookRaw_size`, `superneoBarBlock_size`, closed-form reduction semantics (`reducePhi81Coeff`/`mulRqCoeffSpec`/`mulRq_coeff_formula`), canonicality/idempotence bridges, quotient-normal-form interfaces and extensional closure (`mulRq_eq_of_quotientSpec`, `mulRq_quotientSpec_iff`, `mulRqQuotientSpec_unique`), plus hardening interfaces (`getCoeff`, `getElemBang_eq_getCoeff`, `mulRqQuotientSpecShaped`, typed `Rq` wrapper with `Rq.mulQuotientSpec{,Shaped}_iff`) are theorem-native). | Optional future hardening: complete migration of all ring formulas from raw `get!` statements to `getCoeff`-native statements and add stronger explicit truncation-equality rewrite lemmas for non-canonical inputs. | Theorem-level ring semantics (not only shape/size) sufficient for P10/P12 derivations. |
| `P5` | In progress (added theorem-native base bounds `normInf* <= q/halfQ`, reusable entry/row bound combinators, operation-sensitive inequalities for add/scale/mul/sub over field/coeff vectors, direct SuperNeo-operation wrappers for `superneoBarBlock`/`barLiftVec`/`barLiftMatrix`, challenge-compositional bounds `<= 4`, compositional constructors at both entry/coefficient and operand-norm layers for add/sub/scale, coeff-spec-driven multiplication, explicit raw-coefficient bridge lemmas for multiplication, and a new assumption-free concrete raw-coefficient fallback path at `halfQ`). | Remaining work is mainly tighter-than-`halfQ` multiplication/field-operation bounds from operand norms (current fully internal raw-coefficient fallback is intentionally coarse). | Norm obligations in P16/P17 discharged from theorem lemmas rather than check-only side conditions. |
| `P6` | In progress (both `splitRoundTrip_sound`/`complete`, full bool↔prop bridges for decomposition/per-digit bounds, monotonicity lemmas, and direct quantified extraction lemmas from successful checks are now available). | Still missing a fully constructive theorem layer that derives reconstruction and per-digit constraints from decomposition definitions directly (not only via check predicates). | Universal decomposition theorem with bound guarantees and direct reuse in P20/P21 without check wrappers. |
| `P7` | In progress (bool↔prop interfaces for Boolean-point predicates and hypercube-indicator checks are now available). | Still missing the fully quantified selector theorem over all Boolean vectors without relying on check wrappers. | Full theorem for Boolean selector behavior of `eq` over `{0,1}^ell`. |
| `P8` | In progress (bool↔prop bridge layer for `mleIdentity` is now expanded with iff, explicit size-form proposition bridge, and direct extraction lemmas for size/equality from successful checks). | Still missing quantified theorem over all valid `v,r` (beyond check-bridge interface). | Quantified theorem equating inner-product and folding MLE formulations. |
| `P9` | In progress (bool↔prop bridges now include explicit iff forms and direct shape/equality extraction lemmas from successful checks). | Embedding correctness/linearity is still not fully theorem-native beyond round-trip bridge lemmas. | General embedding/unembedding bijection + linearity theorem suite. |
| `P10` | In progress (`p10CoreCheck_sound` + `p10CoreCheck_complete` give check/prop equivalence and `thm3CoreAssumption` + theorem-native precondition constructor are now available). | Universal derivation of the assumption from P1-P4/P9 lemmas is still missing. | Prove Theorem-3 core directly from P1-P4/P9 lemmas, keep checks only as regression path. |
| `P11` | In progress (added proposition-level linearity/matrix interfaces, bidirectional check/prop bridges, assumption-driven theorem APIs `..._of_assumption(s)` for add/scale/combined lift linearity, structural decomposition lemmas for chunking/single-block reduction, and universal check-assumption -> theorem-assumption conversion bridges). | Core linearity equalities are still not yet derived from lower algebraic lemmas. | Prove lift linearity/correctness directly from embedding/ring lemmas and keep checks as regression only. |
| `P12` | In progress (`dotVec_eq_dot_of_isDVec` is available; `MatrixRowsCompatible` and sound/complete bridges are theorem-native; explicit bool↔prop interface (`matrixTransformIdentityProp`, `matrixTransformIdentity_iff_prop`) and theorem/check assumption interfaces are available, and theorem-native closure from Theorem-3 assumptions is now proved (`matrixTransformEq_of_thm3CoreAssumption`, `p12MatrixTransformAssumption_of_thm3CoreAssumption`, `p12MatrixTransformCheckAssumption_of_thm3CoreAssumption`)). | Remaining gap is deriving/justifying the Theorem-3 assumption itself from lower layers (P10 universal proof), not the P12 closure wiring. | End-to-end P12 theorem path fully grounded by lower proven assumptions (without introducing stronger ad-hoc premises). |
| `P13` | In progress (matrix-specialized bool↔prop interfaces are present, including direct equivalence theorem; theorem/check assumption interfaces include global and fixed-`m,r` forms `p13EvalLinkAssumption{,For}` / `p13EvalLinkCheckAssumption{,For}` with bidirectional conversion/direct access; and quantified assumption-free linkage is now available via `evalLinkIdentityProp_of_size_eq` plus `p13EvalLinkAssumption_from_defs` / `p13EvalLinkCheckAssumption_from_defs`). | Remaining work is mostly downstream: consuming this theorem-native P13 path to reduce P14 wrapper reliance. | P14 constructors consume theorem-native P13 path directly (checks remain regression wrappers only). |
| `P14` | In progress (full bool↔prop bridges are now present, including direct equivalence theorem, and consumed by P20 proposition→check recovery; theorem-native assumption interface now wraps `evalHom2Prop` directly, preconditioned check-assumption interface is available, and bidirectional conversion/direct constructors are in place). | Core theorem still not derived from abstract P12/P13/P15 algebraic lemmas. | Full Theorem-5 proof via P12/P13/P15 theorem interfaces. |
| `P15` | In progress (sound + complete bridges now exist for add/scale checks on vec/scalar homs). | Still missing richer abstract algebra lemmas beyond direct check equivalence. | Complete module-hom theorem API used directly by P14. |
| `P16` | In progress (critical boundary identified; precondition bridge in P20 now uses theorem constants directly, invertibility-window bool has sound/complete theorem bridges, direct invertibility extraction lemmas are available, challenge-coefficient norm bounds imply `withinInvertibilityWindow` for both direct vectors and `coeffSub` differences, `invertible_of_*_of_assumption` APIs are available, protocol context carries `hLowNormInvertibility` as an explicit trusted interface, protocol-side `invDelta` now threads through P20/P21 with assumption-driven inverse witness extraction, and unparameterized global invertibility axiom wrappers were removed). | Theorem 8 itself is still not formally proven in Lean; invertibility remains an explicit input assumption. | Replace assumption boundary with proved theorem, or keep as explicit trusted contract with documented provenance. |
| `P17` | In progress | Expansion guarantees still check-driven. | Universal sampling expansion theorem wired into P20/P21 proof path. |
| `P18` | In progress | SZ/eq-lift results are partial helper checks. | Quantified polynomial lemma set (SZ + eq-lift) used directly by P19/P20. |
| `P19` | In progress | Interpolation correctness/uniqueness not fully formalized. | Full interpolation theorem package used in P20 without check-only assumptions. |
| `P20` | Good shell | Lower obligations still mixed theorem/check-backed (invertibility has explicit `invDelta` window + assumption-driven witness extraction; proposition→check bridges are fully bundled via `p20ArithmeticBundle_props_imply_checks`/`p20ArithmeticBundle_iff_checks`; proposition-native constructors now expose explicit P12/P14 mode quadrants plus a direct Theorem-3-derived P12 route `p20ArithmeticBundle_of_props_with_thm3CoreAssumption`). | Keep dual APIs, and continue converging toward theorem-only premises for P14 (P13→P14 derivation) while preserving check APIs as regression wrappers. |
| `P21` | Good shell | Final reduction still a shell over unresolved lower-level gaps (despite direct P21 assumption/check-assumption constructors for the P14 leg, direct P12 assumption/check-assumption constructors in both `P21` and `ProtocolReduction`, new Theorem-3-derived P12 entrypoints, explicit protocol-target matrix/eval mode quadrants including check/check, mirrored protocol-skeleton `...with_invertibility` endpoints, protocol/full-target check-subset iff bridges, and protocol-surface canonical props/check bridges including P16-window via `protocol_props_iff_checks` with compatibility `smoke_*` aliases). | End-to-end protocol-facing CE theorem with explicit assumptions and theorem-native dependencies (including row-shape compatibility and explicit `invDelta` invertibility threading). |

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
| M2 | Appendix B.2 concrete Goldilocks parameters | `SuperNeo/Parameters.lean` | Encode exact constants and bound checks | generated parameter sanity checks | Pins SuperNeo to the exact concrete parameter regime claimed in the paper. | Done (constants + sanity checks implemented, and extracted theorem-level constant APIs are now available for downstream proofs). |
| M3 | Definition 2 (`cf`, `cf^-1`, `ct`) | `SuperNeo/CoeffMaps.lean` + `SuperNeo/Ring.lean` | Prove inverse/linearity properties for maps | `neo_math::cf`, `cf_inv`, `ct` | These maps bridge coefficient and ring views that SuperNeo repeatedly switches between. | Done (inverse, round-trip, shape, `ct`, and `mulRq` compatibility are implemented with explicit linearity lemmas for vector add/scale map composition). |
| M4 | Ring arithmetic in `F[X]/(X^54 + X^27 + 1)` | `SuperNeo/Ring.lean` | Prove reduction semantics and arithmetic sanity lemmas | `neo_math::Rq::mul` | Core multiplication law behind bar-transform, lift, and matrix-product equalities. | Done (implemented + parity passing; explicit closed-form coefficient semantics, canonicality/idempotence bridges, quotient-normal-form completeness/uniqueness, extensional rewrite constructors from shape+coeff equalities, and hardening interfaces (`getCoeff`, `getElemBang_eq_getCoeff`, shaped quotient spec, typed `Rq`) are available). |
| M5 | Definition 3 (centered `l_inf` norm) | `SuperNeo/Norm.lean` | Define centered representatives and prove basic norm bounds | `neo_math::inf_norm` + generated norm vectors | Norm bounds are prerequisites for low-norm assumptions and soundness-side constraints. | In progress (implementation + parity passing; base bounds, reusable entry/row combinators, operation-sensitive add/scale/mul/sub bounds, SuperNeo-operation wrappers for lift/bar paths, challenge-compositional `<=4` bounds for add/sub/scale, compositional constructors from entry/coefficient assumptions plus operand-norm propagation for add/sub/scale, raw-coefficient multiplication bridges, and an internal coarse `halfQ` raw-coefficient fallback path are implemented) |
| M6 | `split_b` decomposition math | `SuperNeo/Decomp.lean` | Prove reconstruction and per-digit bound | generated Rust `splitCases` vectors | Needed for bounded digit decompositions used in SuperNeo’s concrete arithmetic arguments. | In progress (implementation + parity passing; roundtrip/per-digit bool↔prop bridges, monotonicity helpers, and quantified extraction lemmas from `splitRoundTrip` are in place; full constructive reconstruction/bound proofs from definitions are pending) |
| M7 | `eq` polynomial on Boolean hypercube | `SuperNeo/EqPoly.lean` | Prove indicator behavior on `{0,1}^ell` | generated Rust `eqCases` vectors | Supplies the selector polynomial used by MLE and sumcheck-style reasoning in SuperNeo. | In progress (implementation + parity passing; proposition-level bool↔prop interfaces for `isBoolF` and indicator checks are in place; fully quantified selector theorem pending) |
| M8 | MLE identity `tilde_v(r) = <v, r_hat>` | `SuperNeo/MLE.lean` | Prove equivalence of two MLE formulations | generated Rust `mleCases` vectors | Connects table view and folded view of evaluations used in SuperNeo reductions. | In progress (implementation + parity passing; proposition-level bool↔prop and extraction interfaces are in place; fully quantified proof pending) |
| M9 | Definition 7 coefficient embedding | `SuperNeo/Embedding.lean` | Prove element/vector/matrix embedding bijection | generated Rust embedding vectors/matrices | Embedding is the structural bridge from field objects to ring objects in SuperNeo. | In progress (implementation + parity passing; bool↔prop and extraction interfaces for vector/matrix round-trips are in place; full bijection/linearity proof suite pending) |
| M10 | Theorem 3 inner-product transform | `SuperNeo/Ring.lean` + `SuperNeo/Checks.lean` | Prove `ct(bar(a)*bar(b))=<a,b>` for concrete bar matrix | `neo_math::superneo_bar_matrix`, `superneo_bar_block` | Key algebraic equivalence that powers efficient ring-domain product computations in SuperNeo. | In progress (numeric checks passing; formal theorem pending) |
| M11 | Definition 8 lifting transform | `SuperNeo/BarLift.lean` | Prove blockwise lift properties and linearity | generated Rust `barLiftVecCases` / `barLiftMatrixCases` | Lifting composes embeddings with bar-transform to map full vectors/matrices into ring form. | In progress (implementation + parity passing; proposition-level linearity/matrix interfaces, check/prop bridges, assumption-driven theorem interfaces, single-block structural reduction lemma, and check-assumption->theorem-assumption conversion layer added; core proof pending) |
| M12 | Theorem 4 + App D.1 (`Mz = ct(bar(M)z)`) | `SuperNeo/MatrixTransform.lean` | Row/block proof from Theorem 3 | generated Rust `matrixTransformCases` | Establishes matrix-vector equivalence that underlies SuperNeo’s computational shortcut. | In progress (implementation + parity passing; proposition-level row-shape interfaces and sound/complete bridges added; full proof pending) |
| M13 | Remark 2 evaluation/ct linkage | `SuperNeo/EvalLink.lean` | Prove coefficientwise scaling and ct-eval link | generated Rust `evalLinkCases` | Connects coefficient evaluation with ct, enabling later evaluation-homomorphism proofs. | In progress (implementation + parity passing; quantified size-compatible linkage theorem and assumption-free global/check assumption constructors are now present; downstream theorem-native P14 integration remains) |
| M14 | Theorem 5 + App D.2 evaluation homomorphism | `SuperNeo/EvalHom.lean` | Prove linear-combination preservation under evaluation | generated Rust `evalHomCases` | Gives the homomorphic evaluation property used to justify transformed linear algebra steps. | In progress (implementation + parity passing; formal proof pending) |
| M15 | Definition 15 module homomorphisms | `SuperNeo/ModuleHom.lean` | Abstract module-hom lemmas reused by M14 | module-hom linearity sanity checks | Provides the abstract linearity framework that Theorem 5 instantiates. | In progress (interfaces + sanity checks implemented; theorem layer pending) |
| M16 | Theorem 8 low-norm invertibility | `SuperNeo/InvertibilityAxioms.lean` | Add explicit assumption boundary and concrete precondition checks | Appendix B.2 constants + D.7 bound interface | Captures the invertibility condition required by SuperNeo’s soundness-critical reduction step. | In progress (explicit assumption boundary + preconditions implemented; theorem bridges from all-challenge direct and subtraction norms to window/invertibility added; no global theorem axiom wrapper) |
| M17 | Definition 17 + Theorem 9 (`C`, expansion factor) | `SuperNeo/SamplingSet.lean` | Formalize set conditions and expansion-factor theorem interface | generated Rust `samplingCases` | Formalizes sampling guarantees that control error amplification in SuperNeo analysis. | In progress (implementation + parity passing; formal theorem pending) |
| M18 | Appendix C Lemma 5/6 (Schwartz-Zippel and eq-lifting) | `SuperNeo/PolyLemmas.lean` | Add reusable polynomial lemmas for later proofs | generated Rust `eqLiftCases` + SZ interface sanity | Supplies probabilistic polynomial tools used in SuperNeo’s deferred proof chain. | In progress (implementation + parity passing; proof layer pending) |
| M19 | Polynomial interpolation/evaluation math | `SuperNeo/Interp.lean` | Prove interpolation correctness against sampled points | local Rust generator interpolation vectors | Supports recovery/consistency arguments for polynomial objects used in SuperNeo checks. | In progress (implementation + parity passing; proof lemmas pending) |
| M20 | Executable cross-check harness | `Main.lean` + `SuperNeo/Checks.lean` | Keep deterministic Rust-vs-Lean checks green | `rust-vectors/src/main.rs` generated vectors | Acts as the executable witness that Lean computations match SuperNeo Rust math instances. | Done (all checks currently pass) |

### Status Summary

| State | Count |
|---|---|
| Done | 4 |
| In progress | 16 |
| Partial | 0 |
| Not started | 0 |
