# FPR-NIFS-BRIDGE / FPR-OBLIGATION-EXACT / FPR-OBLIGATION-NECESSITY

```text
property_ids:
  FPR-NIFS-BRIDGE
  FPR-OBLIGATION-EXACT
  FPR-OBLIGATION-NECESSITY

claim:
  Target: specify the recursive verifier independently by the paper-level
  PiCCS -> PiRLC -> PiDEC composition and HyperNova F' transition; prove that a
  reduced, explicitly grouped obligation set accepts exactly that transition;
  then prove that concrete Rust/R1CS acceptance refines the reduced obligation
  set. The equivalence and concrete refinement remain open.

  For every retained obligation family, removing that family while retaining
  the others admits a concrete invalid transition. Any family proved to follow
  from the remaining obligations is removed from the retained set and recorded
  separately as eliminated. This is inclusion-minimality relative to the
  selected protocol primitives. It is not a claim that no algebraically
  equivalent backend can use fewer gates.

trust_direction:
  paper semantics
    -> independent typed NIFS/F' transition
    -> sound, complete, inclusion-minimal obligation tree
    -> concrete Rust/R1CS refinement
    -> exact materialized leaf costs

forbidden_shortcuts:
  - Treating an old verifier, R1CS relation, row count, or satisfying witness as
    the semantic oracle.
  - Defining both sides of an equivalence from the same caller-supplied
    predicates and calling the result paper equivalence.
  - Rewriting Rust checks in Lean without a theorem to the independent paper
    relation.
  - Assuming Accepted, Shape, CE.Holds, or the desired transition inside a
    bridge premise.
  - Treating a digest, self-consistent hash chain, or generated manifest as
    protocol authority.

assumptions:
  - The model-level SuperNeo algebra, extraction, relaxed-binding, SumCheck,
    Fiat-Shamir, and final-decider boundaries remain exactly those named by the
    active folding and security specifications.
  - Backend minimality is measured only after semantic minimality, relative to
    a fixed field, hash, commitment representation, and lowering vocabulary.

non_goals:
  - Certifying any historical 200M/258M estimate or current formula-only count.
  - Proving global minimum R1CS size across all equivalent arithmetizations.
  - Using public message equations as a substitute for private CE openings or
    concrete relation membership.

paper_sources:
  - SuperNeo Definitions 9, 10, 12, 13, and 14; Lemmas 3 and 4; Theorems 6 and 7.
  - HyperNova multi-folding compatibility and Construction 2.

rust_surfaces:
  - crates/neo-fold-clean/src/paper/nifs/**
  - crates/neo-fold-clean/src/paper/reductions/pi_ccs_split_nc_circuit/**
  - crates/neo-fold-clean/src/paper/reductions/pi_rlc_circuit/**
  - crates/neo-fold-clean/src/paper/reductions/pi_dec_circuit.rs
  - crates/neo-fold-clean/src/paper/f_prime/r1cs.rs

lean_surfaces:
  - Nightstream/SuperNeo/Folding/{PiCCS,PiRLC,PiDEC,Composition,Nifs}.lean
  - Nightstream/SuperNeo/SumCheck/{Polynomial,VerifierCertificate}.lean
  - Nightstream/SuperNeo/Sampling/FirstAccepted.lean
  - Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/**
  - Nightstream/SuperNeo/Folding/Nifs/NonInteractive/**
  - Nightstream/Protocol/FPrime/Paper.lean
  - Nightstream/HyperNova/**
  - Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/NifsPaper/**
  - Nightstream/Protocol/FPrime/**

obligation_tree:
  hypernova.construction2.fprime
    dispatch
      next_pc_from_control
      selected_augmented_function
      application_transition
    branch
      base
        iteration_is_zero
        initial_state_equality
        default_running_vector
        no_nifs_fold
      recursive
        iteration_is_positive
        prior_fresh_instance_parse
        prior_hash_and_enc_inst_link
        prior_pc_range_and_selection
        selected_slot_nifs
          pi_ccs
            source_and_output_shape
            fe_sumcheck
            nc_sumcheck
            output_opening_authority
          pi_rlc
            transcript_and_challenge_sampling
            shared_coefficients
            commitment_combination
            public_input_combination
            evaluation_combination
            parent_output_binding
          pi_dec
            parent_and_child_shape
            commitment_recomposition
            public_input_recomposition
            evaluation_recomposition
            child_opening_authority
          composition
            shared_pirccs_pirlc_inputs
            shared_pirlc_pidec_parent
            extraction_or_named_bad_event
            delayed_nc_authority
        non_selected_slots_unchanged
    public_output
      exact_typed_preimage
      output_hash

minimal_evaluator_contract:
  - This evaluator remains blocked on PaperJoint-to-production input, layout,
    and lift refinement, output-evaluation authority, degree bounds,
    and a sound/complete joint-`Q`-to-SplitNc refinement. PaperJoint now fixes the
    finite coefficient-block layout, derives a canonical Boolean-table
    coefficient transform, proves its table-level zero equivalence and exact
    agreement with an independently recursive MLE evaluator,
    exposes the deterministic alpha/gamma mixing-root dichotomy, and fixes
    joint-SumCheck output-point ownership. It now also defines pointwise
    `F`/`NC`/`Eval`/`Q` from explicit extension-carrier tables and proves the
    exact signed `T_abs - sum_x Q` identity. It now also derives the exact
    constant-first signed gamma coefficients, proves executable Horner
    evaluation equals that identity, and defines one explicit arbitrary-point
    joint polynomial whose Boolean sum is the semantic initial value. Canonical
    expected rounds and the terminal are derived from that same polynomial, so
    executable acceptance reduces to table truth, a mixing root, or a named
    SumCheck round collision without a caller-supplied expected callback or
    honesty proof. The sole joint object is now constructed from independent
    CCS matrices/assignments, typed norm sources, and carried matrix-image data;
    its coefficient truth is proved equivalent to those three semantic
    families. It explicitly leaves external Boolean-leaf/production-bit
    ordering, production input/layout/lift refinement, degree
    enforcement, SplitNc refinement, and production integration open.
    The evaluator may not take a semantic
    `Attempt` as its certificate just to obtain ghost fields or carried
    challenges.
  - The reduced verifier is a deterministic partial evaluator from one typed
    input and one raw certificate to an optional canonical output. It is not a
    second record containing the same fields as `Nifs.Accepted`.
  - PiCCS output structure, commitment, and public input are materialized from
    the authoritative input. The shared CE point is derived from the FE
    SumCheck challenge prefix, never supplied independently. In the production
    split protocol, `s_col` is separately derived from the NC challenge prefix
    and `y_zcol` remains typed output payload checked by the NC terminal
    identity.
  - The PiRLC parent is computed with one verifier-derived challenge vector;
    it is not accepted as an independent prover-supplied statement.
  - PiDEC children inherit stage, structure, and point from the computed parent
    and supply only the payload that cannot be derived. Recomposition remains
    checked.
  - F' computes branch selection, `pcNext`, `zNext`, selected-slot replacement,
    inactive-slot copies, the output preimage, and `x`. Only actual failure
    conditions remain checks.
  - Exactness is proved first for the rich input/certificate/output execution.
    An existential public-output theorem is a corollary and may not replace it.

target_obligation_classification:
  must_be_checked:
    - SplitNc FE and NC terminal identities, including typed `y_zcol` output
      authority.
    - PiDEC commitment, public-input, and evaluation recomposition.
    - F' base initial-state equality, recursive prior-PC range, prior public
      link, and selected NIFS verification.
  must_be_computed_canonically:
    - PiCCS preserved output fields, FE-derived shared point, and NC-derived
      `s_col` sidecar.
    - PiRLC combined parent and one verifier-derived scalar vector, with each
      scalar assembled from its own complete coefficient vector.
    - PiDEC child stage, structure, and point.
    - F' branch, control/application output, running-vector update/copies, and
      output digest preimage.
  must_be_structural_direct_dataflow:
    - PiCCS-to-PiRLC complete product identity.
    - PiRLC-to-PiDEC parent identity.
  must_remain_security_boundary:
    - SumCheck and mixing roots, sampler shortfall, relaxed-binding collision,
      hash collision, extraction/rewind failure, and final-child membership.
current_model_status:
  modeled_partial:
    - Generic finite SumCheck coefficient messages whose canonical shape is
      checked by acceptance, Horner evaluation, derived degree, exact
      claimed-chain checking, and conditional projection into the symbolic
      model.
    - Generic first-accepted bounded sampling, named shortfall, and conditional
      agreement with a terminating reference stream.
    - A production-shaped PiRLC sampler contract rooted at one post-PiCCS
      state. It threads a verifier-owned source/successor state across all
      `K+k` scalar coordinates, gives each scalar its own first-accepted
      coefficient execution, assembles the complete coefficient vector into
      that scalar, proves accepted-candidate provenance per coefficient,
      requires separate coefficient-validity and scalar-assembly strong-set
      laws, and makes shortfall at any coordinate fail-closed.
    - Independent production-alphabet mathematics: 16-bit candidates reject
      exactly `65535`; the accepted domain is bijective with
      `Fin 13107 × Fin 5`; centered symbols lie in `[-2,2]`; 54-of-64 bounded
      success is exactly a terminating least-cursor reference execution within
      the bound; and every successful cursor satisfies `48 < consumed ≤ 64`.
      This proves the fourth-window arithmetic, not a concrete Poseidon2 stream
      or successor transcript state.
    - An abstract deterministic production block schedule in which one block
      call jointly owns 16 candidates and its successor state. Lean proves
      that a successful 54-of-64 least cursor consumes exactly four complete
      blocks and that batch state threading uses the same four-block state.
      `PiRlcChallenge.TranscriptMachine` now instantiates that schedule with a
      pure eight-lane Goldilocks state machine, overwrite absorption, the exact
      scalar/digest domain pairs, the independently extracted Poseidon2
      permutation, and lane-major little-endian 16-bit chunk order. It proves
      the lane/part ordering and that every successful fixed execution reaches
      the same four-block successor state. This is executable implementation
      semantics, not yet a theorem about the native Rust transcript, transcript
      gadget, generated R1CS trace, or the exact reached post-PiCCS state.
    - Typed replay for the SumCheck/PiRLC challenge carriers currently present
      in `Nifs.Attempt`, together with formal blindness witnesses. The exact
      represented pre-PiRLC prefix now computes `Replay.postPiCcsState`, and
      `PiRlcSampler.ReplayBridge` proves that canonical replay plus an explicit
      sampler-refinement premise binds every carried PiRLC coordinate to the
      scalar assembled from its own bounded first-accepted coefficient
      execution in the transcript-chained batch at that computed state.
    - PaperJoint coefficient blocks, canonical squarefree Boolean-table
      coefficient transform, table-level residual zero equivalence, exact
      canonical-polynomial/recursive-MLE evaluation equivalence,
      one shared typed Boolean-domain order, explicit paper-level matrix and
      sparse-polynomial CCS residual tables with unconditional single/batch
      zero equivalence,
      canonical strict-norm cubic tables for every typed `K+k` source with
      single/batch equivalence to centered `normBounded 2` under the explicit
      no-zero-divisor boundary,
      explicit carried coefficient-matrix image tables, canonical prior-point
      equality-weighted evaluation, and claimed-minus-derived residuals with
      single/batch equivalence to the carried evaluation equations,
      explicit pointwise `F`/`NC`/`Eval`/`Q` over extension-carrier tables and
      the exact signed `T_abs - sum_x Q = -CCS - norm + carried` identity,
      exact consecutive carried gamma support, exact constant-first signed
      coefficient blocks, an independent unsampled alpha-polynomial/scalar
      coefficient object, exact specialization into the executable Horner
      list, coefficient truth iff the explicit table obligations, and candidate
      finite SumCheck initial-claim equivalence,
      exact equality of the recursive/canonical MLE with the explicit finite
      `sum_x eq(x,r) * table[x]` over that same typed order,
      exact finite target-shift identity, exponent-zero support mismatch,
      strict `b=2` centered-norm/cubic equivalence conditional on no zero
      divisors and distinct canonical residues,
      an explicit arbitrary-point joint polynomial, exact equality between its
      Boolean completion sum and `sum_x Q`, canonical expected rounds and
      terminal evaluation from that same polynomial, deterministic signed
      `MixingRoot`, executable acceptance reducing to table truth, that mixing
      root, or a named SumCheck round collision without a caller-supplied truth
      path, construction of the sole joint object from independent CCS, norm,
      and carried source data, exact equivalence between its coefficient truth
      and those three semantic families, and joint challenge-vector
      output-point binding.
  not_yet_modeled:
    - Integration of the finite SumCheck certificate into PiCCS and
      NonInteractive.
    - Expected-polynomial degree bounds plus production SplitNc FE/NC terminal
      and output-evaluation authority. Typed one-round-per-variable arity and
      the candidate one-joint initial, expected-round, and terminal truth path
      are now derived from one explicit joint polynomial.
    - Complete alpha/gamma and production SplitNc public-coin schedule.
    - External numeric row and paper/production bit-order refinement for the
      now-shared typed Boolean domain and canonical table/MLE convention.
    - Reviewed approval or rejection of the paper's `T_local` versus shifted-`Q`
      correction. The coherent candidate is now used by an explicit pointwise
      `Q` model with a proved signed identity, but that is not an erratum
      decision or production refinement.
    - Kernel instantiation of the concrete Goldilocks Euclid/no-zero-divisor
      boundary, production assignment/order refinement for the typed norm
      tables, and placement of the now-derived norm cubic inside the
      extension-field `Q`.
    - Refinement of the independently constructed paper CCS tables to
      `Concrete.ccsSatisfied` plus their base-to-extension placement;
      production assignment/order and base-to-extension refinement for the
      typed norm tables; refinement of the explicit carried coefficient
      matrices to concrete ring multiplication, instantiation of the
      base-to-extension lift, and placement of those residuals; root counting
      for the now-explicit signed coefficient object; and
      sound/complete SplitNc refinement.
    - Production FE challenge-to-point and NC sidecar authority.
    - Concrete `enc_str(F'_j)` to selected NIFS structure/key refinement.
    - Canonical parsing and exact production public-input carrier.

The fixed rejection-sampler arithmetization is not a protocol obligation. Its
only authority-bearing contract is that successful execution refines the
verifier-owned transcript derivation, returns admissible strong-set elements,
and has an explicit distribution/error theorem. A cheaper implementation of
that contract is permitted only after its concrete implementation is proved to
satisfy the contract and the Rust/R1CS refinement gate closes. The current
generic sampling theorem authorizes no production constraint removal.

candidate_schedule_reductions_not_yet_authorized:
  - The partial schedule currently omits a final PiDEC-child absorb and absorbs
    only PiCCS output evaluations before PiRLC.
  - Replay exactness proves only deterministic reconstruction of that selected
    skeleton.
  - `canonicalEvents_replacePiCcsOutputPoints` and the explicit
    `piCcsOutputProjectionSufficiency` gap show that the schedule is blind to
    authority fields.
  - These omissions remain candidates until output/point sufficiency, complete
    paper scheduling, and concrete transcript refinement are proved.

current_gap_boundary:
  - Independent NIFS semantics: `Nifs.Attempt`, `Nifs.Wiring`,
    `Nifs.Accepted`, and `Nifs.PaperNifsTransition` now compose the typed
    PiCCS, PiRLC, and PiDEC relations without importing implementation code.
    `Nifs.complete` and `Nifs.paperNifsTransition_complete` prove honest model
    and external-transition completeness, while
    `Nifs.accepted_inputsValid_or_badEvent` retains the extractor, uniqueness,
    rewind arithmetization, final-child membership, and named bad events as
    explicit boundaries. This is the semantic foundation, not production
    conformance and not permission to remove rows.
  - Public-input carrier: fresh CCS claims expose 257 field inputs, while the
    production CE path carries five complete ring columns (270 coefficients).
    The final 13 coefficients are not generic zero padding after PiRLC: ring
    multiplication may make them nonzero and PiDEC intentionally recomposes
    them. The active Concrete model currently projects `take 257`. A reviewed
    embedding/relation refinement is required; neither truncation nor silently
    redefining the paper public input closes the bridge.
  - PiCCS paper erratum boundary: Section 7.3 and Appendix D.4 define the
    carried-evaluation terms in `Q` at absolute exponent `2K+k+I`, but define
    `T(C)` at unshifted exponent `I`. Read literally, coefficient independence
    cannot yield the paper's Equation (9). The coherent candidate convention
    is `T_abs(C) = C^(2K+k) * T_local(C)`, but that correction must remain
    explicit and reviewed before a production Lemma-7 or production
    SumCheck-initial claim is accepted. Lean now proves this finite shift
    identity and a genuine
    exponent-zero support mismatch under positive paper dimensions. It also
    defines the candidate pointwise `Q` over explicit extension-carrier tables
    and proves the corresponding exact signed `T_abs - sum_x Q` identity, its
    exact signed coefficient/Horner form, and the resulting candidate
    SumCheck-initial equality. Those theorems do not approve the erratum or
    instantiate production data. The displayed
    norm-product ranges are also mutually inconsistent. Lean now derives the
    production `b=2` roots from centered `|z|<2`, proves the canonical
    representatives `q-1,0,1` distinct, and proves equivalence with
    `(z+1)z(z-1)=0` under the explicit no-zero-divisor boundary. It still must
    close that boundary for the concrete modulus and place the cubic in the
    formalized extension-field `Q`; no display range is an assumption.
  - PiCCS: decoded FE/NC rows already yield actual `SumCheck.Accepted` values,
    but that is not yet a complete verifier bridge. `SumCheck.Instance`
    currently also carries semantic ghost data (`trueInitial` and each round's
    `expected` polynomial), while `PiCCS.Arithmetization` is supplied as an
    external premise rather than constructed from the authoritative input,
    output evaluations, and verifier challenges. The noninteractive
    certificate must exclude those ghost fields; Lean must materialize the
    production FE/NC initial and terminal identities and prove the truth path
    separately.
    The legacy symbolic `SumCheck.Instance` and current NonInteractive event
    carrier still use function-valued claims and declared degree metadata.
    Separately, `SumCheck.Finite` now provides coefficient-list messages,
    canonical validation, Horner evaluation, length-derived degree, an exact
    executable claimed-chain checker, and a one-way symbolic projection when
    the semantic truth path is supplied independently. PaperJoint now computes
    the candidate one-joint claimed initial as `T_abs` and true initial as the
    explicit `sum_x Q`, proves claim truth iff its signed polynomial vanishes,
    derives every expected round and the terminal from that same explicit
    polynomial, enforces typed one-round-per-variable arity, and composes finite
    acceptance with the generic round-collision reduction. It does not yet
    prove the expected degree bound, refine this one-joint path to production
    SplitNc, or feed the NonInteractive schedule. `PiCCS.Shape` still only
    requires all outputs to share an arbitrary point; `BoundOutputs` binds the
    isolated one-joint candidate to its challenge vector, but the production FE
    challenge-to-output bridge required by Section 7.3 remains open.
    A generated typed source/output column map and exact value-binding theorem
    are also required before constructing an unconditional paper accepted
    attempt.
    The production split additionally derives `r'` from the FE challenge
    prefix and `s_col'` from the NC challenge prefix, then checks FE against
    `y_ring` and NC against a separate `y_zcol` sidecar. Generic
    `CE.Instance`/`PiCCS.Attempt` carries neither `s_col` nor `y_zcol`, and its
    accepted predicate currently links neither terminal value to an output.
    Those fields need a typed SplitNc statement and an erasure/refinement
    theorem; they must not be hidden inside an opaque transcript witness.
    Separately, SuperNeo Section 7.3 presents one SumCheck for the mixed
    polynomial `Q`, whereas the production-shaped Lean/Rust path uses distinct
    FE and NC SumChecks. That split must be proved complete and sound for the
    same unmixed CCS, prior-evaluation, and norm obligations (with its changed
    challenge/error accounting); implementation correspondence alone is not
    such a proof.
    PaperJoint now fixes the paper's finite joint-`Q` coefficient-block
    skeleton and a verifier-derived squarefree table coefficient transform.
    A shared typed Boolean domain now owns low/high order once. The CCS branch
    derives its leaves from explicit finite matrices, assignment columns, and
    a sparse polynomial and proves its single/batch zero equivalence without
    a supplied evaluator or iff. The norm branch places the independently
    derived cubic on every typed source coordinate and proves single/batch
    equivalence to centered `normBounded 2`, conditional only on the explicit
    no-zero-divisor boundary. `TableResidualData.residualizationBoundary`
    combines those independently derived tables and proves coefficient truth
    iff the table obligations hold. `ConcreteJointData.toJointData` now builds
    the sole joint object from the independent CCS, norm, and carried inputs,
    and `coefficientTruth_iff_semanticTruth` proves that its unsampled
    coefficient truth is exactly their semantic conjunction. Lean also proves
    that evaluating the resulting canonical flat polynomial equals an
    independently recursive multilinear evaluator at every dimension-checked
    point. This is not yet the production arithmetization
    because external numeric row/bit ordering, the bridge to
    `Concrete.ccsSatisfied`, production assignment packing, and
    base-to-extension placement are unrefined. The carried branch now derives
    coefficient-expanded matrix images from explicit matrices and assignments,
    computes their prior-point values by the canonical equality-weighted
    hypercube sum, and proves claimed-minus-derived residual zero iff the
    carried evaluation equation holds. Its concrete ring-coefficient matrix
    refinement, base-to-extension lift, and placement in `Q` remain open. The
    slice also names
    the target-exponent mismatch, proves the finite target-shift identity,
    derives the strict `b=2` norm cubic independently of the inconsistent
    paper displays, exposes deterministic alpha/gamma
    `MixingRoot`, and makes every output point equal the joint SumCheck
    challenge vector. It now proves the candidate pointwise-`Q` signed identity,
    exact signed coefficient/Horner serialization, exact specialization from
    the independent unsampled coefficient object, coefficient truth iff the
    explicit table obligations, and executable finite acceptance reducing to
    the independent semantic conjunction, a signed mixing root, or a named
    SumCheck round collision without a caller-supplied truth path. It does not
    yet connect the signed object to root-counting probability, prove the
    expected degree bound, establish production SplitNc terminal/output
    authority, or refine the production FE/NC split.
  - PiRLC: the exact sampled challenge vector, role-to-CE carrier wiring,
    output-to-PiDEC-parent equality, and quotient/remainder-to-ring-combination
    theorem are still required.
  - PiDEC: strict public recomposition can be related to paper public equations,
    but the production Concrete.relationSemantics split/recompose algebra and
    private child CE openings are still required for knowledge reduction.
  - Composition: shared attempts, final CE.Holds witnesses, extraction and
    arithmetization boundaries, and delayed y_zcol/NC authority must be joined
    before Composition.fold_knowledge_or_bad_event is applicable.
  - F': `Protocol.FPrime.Paper` now states an independent Construction-2
    family indexed by a fixed augmented-function identity `j`, with separate
    HyperNova `ell` and SuperNeo `k` axes. Acceptance requires the control
    output to dispatch to that fixed `j`, evaluates fixed `F_j`, binds the
    fresh and selected running statements to the structure selected by the
    hashed verifier key and prior slot, preserves every non-selected slot, and
    hashes the exact typed output preimage. `Paper.Completeness` constructs the
    canonical base and recursive outputs rather than assuming a completed
    branch. Every active step retains the exact accepted NIFS attempt and
    exposes `Nifs.PaperNifsTransition`. This is still model-level: concrete key
    generation/parsing, Fiat-Shamir/Poseidon2 schedules, hash binding,
    `enc_inst` refinement, full NIVC extraction, and the production
    `Protocol.FPrime.Step`/R1CS bridge remain open.
  - Noninteractive NIFS: `Nifs.PaperNifsTransition` models the interactive
    paper phase equations. NonInteractive now defines a typed partial schedule
    and deterministic replay for the FE/NC round challenges and PiRLC
    coordinates already carried by `Nifs.Attempt`; canonical acceptance is
    equivalent to exact oracle materialization. It remains over
    function-valued round messages and an abstract `Oracle`, omits alpha/gamma,
    initial/terminal/configuration and output-point authority, and permits
    constant or colliding oracle implementations. Its formal blindness
    witnesses therefore block any claim that this partial schedule is
    sufficient.
    `FirstAccepted` now gives generic bounded first-accepted sampling, exact
    named `Shortfall`, a relational reference stream with a least finite
    stopping witness, exact conditional bounded/reference agreement, and an
    accepted-candidate preimage for every selected symbol. `PiRlcSampler`
    specializes that contract to the actual nested shape: `ResponseRefinesAt`
    requires one transcript-chained coefficient execution for each of the
    `K+k` replay scalars, `Specification.assemble` builds each scalar from its
    full selected coefficient vector, `StrongSetLaw` separates accepted-
    coefficient validity from scalar-assembly validity, and shortfall at any
    coordinate excludes the whole refinement instead of authorizing padding
    or defaults.
    `Replay.postPiCcsState` now folds the exact represented pre-PiRLC prefix,
    and `ReplayBridge.acceptsCanonical_challenges_eq_sampled` proves that
    canonical replay plus `ReplayResponseRefines` binds every carried PiRLC
    scalar to its transcript-chained sampled coefficient vector at the
    computed state. `ProductionAlphabet.acceptedFactorization` independently
    proves that the accepted 16-bit domain is exactly
    `Fin 13107 × Fin 5`; `sample54of64_eq_some_iff_reference_within` proves the
    conditional 54-of-64 reference equivalence; and
    `successful_cursor_in_fourth_digest_window` proves
    `48 < consumed ≤ 64`. `ProductionSchedule` additionally forces candidates
    and successor state to come from the same abstract block calls and proves
    that successful batch threading advances by exactly four complete blocks.
    `ProductionStrongSet` proves every sampled coefficient vector lies
    pointwise in `[-2,2]`, every pairwise difference lies in `[-4,4]`, distinct
    vectors expose a nonzero difference coordinate, the minimal sufficient
    integer threshold is therefore `5`, and the expansion-factor arithmetic is
    exactly `2 * 54 * 2 = 216`.
    These are still only model-level theorems. A pure production-shaped
    Poseidon2 block machine now fixes overwrite absorption, domain pairs,
    canonical lane extraction, and concrete chunk ordering, but there is no
    proof yet that the native transcript, transcript gadget, or generated rows
    refine that machine or begin from the exact post-PiCCS state. The current
    carrier is still incomplete, and centered Goldilocks/quotient-ring
    embedding, the Theorem-8 low-norm invertibility boundary, rotation-matrix
    refinement, distribution/bias, and emitted-R1CS refinement remain open.
    Exact uniformity must not be assumed from
    `digest32()` bytes either: canonical Goldilocks lanes are not uniform
    64-bit strings. The security target needs an explicit statistical-distance
    or min-entropy bound under a stated sponge/random-oracle model. No
    `pi_rlc.challenge` row change is authorized by these sampler theorems.

failure_class:
  A row-decoded verifier accepts an invalid paper transition; a required
  authority surface is replaced by a self-consistent digest; a retained check
  is redundant; or an omitted check admits a concrete invalid transition.

counterexample_or_witness:
  Required per retained obligation family. Countermodels mutate one semantic
  authority coordinate while satisfying every other reduced obligation. A
  family may be removed only when Lean proves it derived from the remainder.

lean_theorems:
  - Proved: independent typed NIFS acceptance and exact PiCCS-to-PiRLC and
    PiRLC-to-PiDEC wiring (`Nifs.Accepted`, `Nifs.Wiring`).
  - Proved: honest independent NIFS phase and external-transition completeness
    (`Nifs.complete`, `Nifs.paperNifsTransition_complete`).
  - Proved: accepted NIFS inputs are valid or an explicit composition bad event
    occurs, under the stated extraction boundaries
    (`Nifs.accepted_inputsValid_or_badEvent`).
  - Proved: independent fixed-`j` HyperNova Construction-2/F' carrier,
    selected NIFS edge, and constructive base/recursive output existence
    (`Paper.Holds`, `Paper.PaperFPrimeStep`, `Paper.selected_nifs_edge`,
    `Paper.selected_nifs_transition`, `Paper.base_exists_holds`,
    `Paper.recursive_exists_holds`).
  - Proved: finite verifier-visible SumCheck encoding and exact claimed-chain
    checker, with semantic ghosts excluded from the certificate
    (`Finite.Message.canonicalCheck_eq_true_iff`,
    `Finite.check_eq_true_iff_accepted`,
    `Finite.accepted_implies_symbolicAccepted_and_truthPath`).
  - Proved: generic first-accepted bounded/reference semantics, least consumed
    cursor, exact output length, accepted-candidate provenance, and explicit
    shortfall; plus the production-shaped PiRLC batch contract with one
    transcript-chained coefficient execution per scalar, separate coefficient
    and assembly strong-set laws, fail-closed per-coordinate shortfall, exact
    computation of the represented post-PiCCS replay state, and conditional
    equality between every carried PiRLC scalar and its assembled sampled
    coefficient vector rooted at that state.
  - Proved: deterministic replay exactness for the current partial NIFS
    carrier, plus formal blindness to FE/NC envelopes and PiCCS output points.
  - Proved: PaperJoint finite coefficient layout, verifier-derived squarefree
    table coefficient transform, table-level coefficient-zero iff leaf-zero
    residualization, canonical-polynomial/recursive-MLE evaluation equivalence,
    exact recursive-MLE/equality-weighted-hypercube-sum equivalence,
    one shared typed Boolean-domain order and explicit paper-level CCS
    matrix/sparse-polynomial residual tables with unconditional single/batch
    zero equivalence,
    canonical strict-norm cubic tables for every typed `K+k` source with
    single/batch equivalence to centered `normBounded 2` under the explicit
    no-zero-divisor boundary,
    explicit carried coefficient-matrix image tables, exact canonical
    prior-point hypercube evaluation, and claimed-minus-derived residuals with
    single/batch equivalence to the carried evaluation equations,
    explicit pointwise `F`/`NC`/`Eval`/`Q` and the exact signed
    `T_abs - sum_x Q = -CCS - norm + carried` identity,
    exact consecutive carried gamma support, signed constant-first coefficient
    serialization, independent unsampled coefficient object, exact
    specialization into executable Horner evaluation, coefficient truth iff
    explicit table obligations, and candidate finite SumCheck initial binding,
    exact finite target-shift identity, exponent-zero support mismatch,
    strict `b=2` centered-norm/cubic equivalence under the explicit
    no-zero-divisor boundary and distinct canonical root representatives,
    one explicit joint polynomial owning initial/round/terminal truth, typed
    one-round-per-variable arity, construction of the sole joint object from
    independent CCS/norm/carried inputs, exact equivalence between its
    coefficient truth and their semantic conjunction, executable acceptance
    reducing to that conjunction, a signed mixing root, or a named SumCheck
    round collision without a caller-supplied truth path, and joint
    challenge-vector output-point binding.
  - Missing: integration of those slices into one PiCCS/NIFS verifier,
    complete public-coin coverage, external Boolean-leaf/bit-order refinement,
    concrete CCS/norm/ring-coefficient and base-to-extension refinements,
    production residual placement, expected degree bounds, SplitNc terminal
    authority/refinement, root counting, complete-carrier and concrete-stream/
    fixed-bound/strong-set/distribution production sampler refinement, and
    Poseidon2/Rust/R1CS refinement.
  - Missing: Construction-2 hash-binding and full recursive NIVC
    knowledge-soundness theorem over the explicit NIFS bad events.
  - Missing: production PiCCS Accepted refinement.
  - Missing: production PiRLC Accepted refinement.
  - Missing: production PiDEC Accepted plus private-opening refinement.
  - Missing: concrete Composition.fold_knowledge_or_bad_event instantiation.
  - Missing: MinimalRecursiveVerifierAccepts iff PaperNifsTransition.
  - Missing: per-family necessity or derivation theorems.

axiom_report:
  The independent NIFS and partial replay theorems are in
  `tests/Axioms/Paper.lean`; finite SumCheck, generic first-accepted sampling,
  and PaperJoint have dedicated fail-closed guards in
  `tests/Axioms/{SumCheckFinite,FirstAccepted,PiCCSPaperJoint}.lean`. All are
  imported by the aggregate axioms gate. The production and F' bridge reports
  remain open. Every completed bridge theorem must be added to the gate; local
  protocol conclusions are forbidden as assumptions.

proof_hash:
  Open.

conformance_status:
  open. Existing row-decoded checklist and cost-manifest theorems are local
  correspondence/accounting evidence only.

retest_commands:
  - cd formal/nightstream-lean && ./scripts/validate.sh static
  - cd formal/nightstream-lean && ./scripts/validate.sh build
  - cd formal/nightstream-lean && ./scripts/validate.sh axioms
  - cd formal/nightstream-lean && ./scripts/validate.sh check
```
