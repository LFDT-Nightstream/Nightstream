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
  set. The canonical fixed-active raw NIFS checker and fixed-one F' evaluator
  now exactly characterize model-level physical acceptance and execution.
  Paper/security closure and concrete Rust/R1CS refinement remain open.

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
  - Nightstream/SuperNeo/Folding/Nifs/ConcretePhi81/**
  - Nightstream/SuperNeo/Folding/Nifs/NonInteractive/**
  - Nightstream/Protocol/FPrime/Paper.lean
  - Nightstream/Protocol/FPrime/ConcretePhi81/**
  - Nightstream/HyperNova/**
  - Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/NifsPaper/**
  - Nightstream/Protocol/FPrime/**

legacy_artifact_migration:
  - The active Lean package imports none of the legacy formal projects.
  - Three Rust artifact generators still target `formal/superneo-lean`: the
    projection identity, projection-binding shape, and transcript-schedule
    diagnostics. The old directory therefore cannot be deleted yet.
  - Packed Mod-5 now has a selective active schema and isolated role-point
    refinement. Its physical decoder/image bridge remains open and it
    authorizes no row removal. Aggregate acceptance now has an active schema-2
    arity-56 leaf, a handwritten evaluator, and exact refinement of its nine
    normalized rows to the independent three-family semantics. The deprecated
    arity-48 artifact remains diagnostic only; recursive source decoding and
    physical placement are still open.
  - The current gadget-native fixed selector is a cost formula, not an emitted
    relation or combined assignment materializer. A generated physical bridge
    may therefore claim only the recursive branch until selector emission and
    selector soundness are separately implemented and proved.
  - The recursive aggregate-acceptance bridge must consume the transitive
    sparse column expansions produced by Rust's actual `build_source_terms`
    path. Its compact artifact must record expansion provenance, exact Boolean
    row ownership, source and active row placement for all chunks, and a
    lossless independently extracted CCS-row image. Lean derives the 960-chunk
    census and every total from those records; neither `ChunkBitOuterImage` nor
    a hand-reconstructed contiguous-lane formula is physical conformance.
  - Projection identity, projection-binding shape, and transcript-schedule
    artifacts should be migrated only if their metadata is still consumed by
    the current compiler/cost tree; obsolete diagnostic fields should be
    deleted rather than ported.
  - No old minimal-verifier, evaluation-homomorphism wrapper, authority bundle,
    or assumption record should be copied. Active typed replacements already
    exist and are stronger.

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
  - This evaluator remains blocked on production input/lift refinement,
    output-evaluation authority, degree bounds, and a direct strong-reduction
    proof for production SplitNc with explicit bad-event bounds. Exact
    transcript equivalence with the paper's single displayed `Q` is not the
    target: Lean proves that the paper's square `ColumnLayout` cannot
    instantiate a complete 54-lane Phi81 carrier. Both arithmetizations must
    instead refine the same independently stated Section 7.3 relation.
    PaperJoint now fixes the
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
    ordering, production input/lift refinement, degree enforcement, direct
    SplitNc security reduction, and production integration open.
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
  fixed_one_canonical_profile:
    - The typed singleton slot derives dispatch; no dispatch check remains.
    - The canonical input carrier constructs the one-based prior counter and
      fresh relation structure from verifier-owned setup; no prior-slot or
      expected-structure check remains on that carrier.
    - Exactly two outer physical checks remain: positive recursive iteration
      and the prior public-input link. The third retained family is the complete
      raw selected-NIFS verifier, whose successful result is computed rather
      than accepted as a caller predicate.
    - Semantic validity of that selected transition still requires the
      separately named source/output authority and bad-event closure premises.
    - These are model-level eliminations. Production row removal additionally
      requires a decoder refinement proving that Rust/R1CS constructs this
      canonical carrier rather than accepting either omitted value from the
      prover.
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
    - A finite executable checker for that fixed sampler contract. It computes
      each 64-candidate prefix, selects the first 54 accepted coefficients,
      assembles the resulting `RingF`, and compares every coordinate with the
      carried challenge. Lean proves the Boolean batch result equivalent to
      `Sampler.CertificateAccepted`; `Classical.choose` appears only when the
      proof reconstructs typed execution evidence from successful computation,
      never in the checker. This does not yet refine the concrete Poseidon2
      transcript, Rust sampler, R1CS rows, or any constraint count.
    - A canonical fixed-active NIFS carrier that materializes the sole relation
      structure, stages, incoming parent, and all source structures from typed
      verifier-owned inputs. Generic checked-parent acceptance is exactly
      equivalent to four retained incoming families: shared point plus
      commitment, public-input, and evaluation recomposition. The complete raw
      NIFS Boolean checker composes those families with the exact Split-NC,
      sampler, and outgoing PiDEC checkers and is proved equivalent to
      `ConcretePhi81.Accepted`.
    - A payload-minimal fixed-one F' evaluator whose input contains no selected
      slot, raw prior counter, relation structure, or stages. Its physical
      checker owns only positive iteration, the complete prior public-input
      link, and the canonical raw NIFS certificate. Successful execution is
      exactly characterized and computes the complete output. Soundness to the
      independent F' relation remains conditional on explicit semantic and
      security closure premises; honest completeness preserves sampler
      shortfall as an explicit outcome.
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
      output-point binding. The paper-joint executable checker now consumes a
      separate `ProtocolPolynomial.VerifierInput` containing only the sparse
      constraint polynomial, public prior point, and public carried
      coefficients. Hidden assignments and image tables remain only in the
      semantic soundness source, and Lean proves that sources agreeing on the
      three authoritative fields project to the same verifier input.
      The transcript-bound wrapper now passes that complete verifier input
      together with the prior transcript state through a typed `Statement`;
      the former arbitrary `Context` parameter is gone, and the SumCheck
      degree ceiling is computed from the explicit sparse monomial syntax plus
      the strict-norm branch instead of accepted from either a free checker
      argument or caller-inflatable declared degree metadata.
  not_yet_modeled:
    - A Rust/R1CS decoder refinement into the payload-minimal canonical NIFS/F'
      carrier, including proof that every omitted field is verifier-constructed
      rather than accepted from witness columns.
    - A complete production protocol -> phase -> family -> leaf cost tree in
      which every materialized row has one Rust emitter and one Lean semantic
      or refinement owner. Existing local stage trees do not yet close this
      whole-relation accounting boundary.
    - Concrete native/gadget/R1CS refinement of the complete production
      Poseidon2 coin/state schedule and its exact typed FE/NC carrier. The
      model-level schedule now derives every pre-SumCheck coin, absorbs each
      message before deriving the next challenge, threads the FE successor
      directly into NC, and constructs an accepted honest exact transcript.
    - Direct cryptographic soundness and changed error accounting for the
      production two-SumCheck reduction. Exact transcript equivalence with the
      paper's single displayed `Q` is deliberately not required; the direct
      paper square carrier is impossible for Phi81. Semantic equivalence to the
      independent Section 7.3 relation, deterministic phase soundness, and
      honest exact completeness are proved. Output-authority and named
      bad-event probability bounds remain open.
    - Proof that the abstract transcript and output absorption do not omit or
      collide on their typed arguments. `TranscriptReplayCollision` and
      `TranscriptStateCollision` cover the complete statement-plus-round
      replay; `OutputAbsorptionCollision` covers the complete incoming-state
      plus output-message pair. Concrete Poseidon2 refinement must discharge
      and bound these events.
    - Native/R1CS conformance for the now-complete model-level alpha/gamma and
      SplitNc public-coin schedule.
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
      for the now-explicit signed coefficient object; plus concrete source,
      lift, and output-authority refinement for SplitNc.
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
  - Honest-baseline provenance: the current generated
    `FPrimeFixedCarrierNifsArtifact` comes from a test-local `1 x 257`
    all-zero R1CS fixture passed through the direct-CCS frontend. It is not the
    output of the production `compile_fixed_point` F' compiler and does not
    export the exact matrices, sparse constraint polynomial, numeric-row
    padding map, or authoritative source ordering needed for semantic
    refinement. Its dimensions, digest, decoded values, and acceptance flags
    remain diagnostic artifact evidence only; they are neither semantic
    authority nor permission to remove rows.
  - Public-input carrier: fresh CCS claims expose 257 field inputs, while the
    production CE path carries five complete ring columns (270 coefficients).
    The final 13 coefficients are not generic zero padding after PiRLC: ring
    multiplication may make them nonzero and PiDEC intentionally recomposes
    them. The active Phi81 relation now owns a typed 270-field public carrier,
    inserts thirteen verifier-fixed zeros in fresh assignments and matrices,
    and proves exact scalar CCS residual/zero-set preservation. This does not
    preserve nonconstant coefficient images or commitments: old private column
    257 changes block/lane under the repair. Rust still exposes the 257-field
    source. A production adapter must therefore construct the 270-field source,
    recompute CE images and Ajtai commitments, and prove exact refinement;
    neither truncation nor claim reuse closes the bridge.
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
    Section 7.3 also assumes a square relation with `M_1 = I` and reuses the
    first matrix evaluation as the assignment/norm evaluation. The active
    thirteen-port relation has `bit` at port zero and no identity role.
    Production Split-NC is therefore a direct-assignment protocol variant, not
    a literal instantiation of that displayed verifier flow. Its independent
    NC soundness/completeness and `y_zcol` authority must close directly from
    the same assignment used by CCS/CE; port zero must never be treated as the
    paper identity matrix.
  - PiCCS: decoded FE/NC rows already yield actual `SumCheck.Accepted` values,
    but that is not yet a complete production verifier bridge. The symbolic
    `SumCheck.Instance` used only in semantic reductions carries `trueInitial`
    and expected-polynomial ghosts; the executable paper-joint certificate
    does not. Its algebraic checker now accepts a minimal public
    `VerifierInput` plus raw finite round/output messages and cannot read
    semantic assignment/image tables. The transcript wrapper receives the
    prior state and that public input in one typed statement, but its abstract
    functions may still ignore/collide until Poseidon2 refinement. Production
    SplitNc still supplies `PiCCS.Arithmetization` as an external premise
    rather than constructing complete FE/NC truth from authoritative input,
    output evaluations, and verifier challenges. The independent SplitNc
    verifier layer now exposes only the sparse CCS polynomial, prior row point,
    running coefficient claims, and value-only `y_ring`/`y_zcol` output product.
    Its first FE polynomial module fixes the production-shaped row/lane product,
    `row || lane` coordinate order, 54-live-lane zero extension, exact fresh and
    carried gamma exponents, verifier initial claim, and one terminal formula
    shared by source-derived and message-derived paths. Lean proves that the
    existing source-bound `y_ring` predicate is sufficient for terminal
    equality without requiring the independently owned `y_zcol` branch. It now
    also proves, from independently derived CCS and carried-evaluation
    residuals, the exact signed identity between the verifier initial claim and
    the full typed row-by-lane Boolean sum. Consequently honest FE truth equals
    the generic recursive SumCheck cube sum without a caller-supplied carried
    selector premise. The mixed-width FE checker now keeps physical row and
    three-slot lane messages distinct, derives the uniform semantic degree
    view without widening serialization, proves fixed-challenge completeness,
    and proves that acceptance implies independent FE truth, a named
    alpha/gamma mixing root, or a fixed-degree SumCheck collision. A canonical
    FE phase evaluator now parallels the NC evaluator, and a protocol module
    threads FE's exact outgoing transcript state into NC, derives the shared
    output point pair, and proves deterministic soundness for the complete
    Section 7.3 obligation set while keeping missing `y_ring`/`y_zcol` source
    authority explicit. Concrete coin derivation, Poseidon2 refinement,
    honest transcript completeness, and Rust/R1CS decoding remain open. The
    independent NC layer now also derives a complete padded column/lane table solely from
    the full-carrier source assignments, fixes exact column-then-lane point
    serialization with fail-closed decoding, and proves that its nested MLE
    restricts on the Boolean cube to the independently defined diagonal cubic.
    Lean now proves this exact padded Boolean relation equivalent to
    full-carrier norm truth; only the soundness direction needs the explicit
    base-field no-zero-divisors premise. The NC polynomial separately names
    the paper-relative, paper-joint, and production Split-V1 gamma schedules,
    proves the joint and Split-V1 shifts exactly, and proves its literal zero
    initial claim equals the generic recursive Boolean-cube sum for every
    honest source and every schedule. Split-V1 is `gamma` times the relative
    paper mixture, so `gamma = 0` is an unconditional bad root for any witness.
    For the paper-relative schedule, Lean also proves an exact deterministic
    equivalence: a zero mixture means full-carrier truth, a selector root, or a
    gamma-mixing root. The concrete-carrier layer now defines the exact norm
    `a² - 7b²`, proves the conjugate identity, and derives extension-field
    no-zero-divisors from two explicit premises: base-field no-zero-divisors
    and projective nonresiduosity of seven. From those premises Lean proves the
    exact Split-V1 decomposition: zero acceptance means full-carrier truth, a
    selector root, a paper-relative gamma-mixing root, or the extra
    `gamma = 0` root. The active dependency-light project still lacks closed
    Goldilocks primality and seven-nonresidue certificates; the corresponding
    Mathlib-backed proofs in the deprecated SuperNeo project use a different
    carrier and toolchain and are not silently imported as authority. The NC
    terminal bridge now
    proves that a source-bound, canonically padded `y_zcol` evaluates to the
    independently derived mixed terminal for all three named gamma schedules.
    The converse is deliberately not claimed: a kernel-checked necessity
    fixture proves that the scalar terminal equality alone can accept a forged
    `y_zcol` that is not source-bound. The current native order derives the NC
    test point before validating that sidecar, so its authority must be closed
    by a downstream proof/opening or by a protocol-order change; it cannot be
    justified as an ordinary non-adaptive polynomial test. Lean also proves
    that every honest NC round is a polynomial of degree at most four and
    materializes it as exactly five constant-first coefficients. A
    protocol-local checker now parses exactly five slots per NC round, replays
    the Boolean-sum/challenge-forwarding/terminal chain without semantic
    ghosts, and proves algebraic completeness from the independent NC truth
    path after an exact-arity challenge vector is fixed. The exact
    message-before-challenge honest constructor now derives the concrete
    Poseidon2 coins, emits each FE then NC round before its challenge, packages
    the typed certificates into the exact physical carrier, and proves accepted
    source-bound execution. The fixed-carrier false-claim/bad-challenge
    reduction, authoritative output composition, and Rust/R1CS refinement
    remain open.
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
    acceptance with the generic round-collision reduction. The executable
    PaperJoint verifier ceiling and the new Split-FE ceiling are now computed
    from explicit sparse monomial syntax rather than declared degree metadata;
    Lean proves metadata independence. It still does not prove that the actual
    nonlinear polynomial and every derived round obey those ceilings, refine
    this one-joint path to production SplitNc, or feed the NonInteractive
    schedule. `PiCCS.Shape` still only
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
    A typed SplitNc statement, source-derived output carrier, exact sequential
    FE/NC replay, and complete model-level Poseidon2 schedule now exist, but
    their composition into the generic `PiCCS.Accepted`/NIFS surface and
    Rust/R1CS decoding is still open; those fields must not be hidden inside an
    opaque transcript witness.
    Separately, SuperNeo Section 7.3 presents one SumCheck for the mixed
    polynomial `Q`, whereas the production-shaped Lean/Rust path uses distinct
    FE and NC SumChecks. Lean now states the Section 7.3 CCS, norm, and carried
    obligations independently of either verifier and proves SplitNc semantic
    truth equivalent to that statement. The composed verifier proves
    deterministic soundness with output authority and every phase bad event
    explicit, and the exact sequential schedule has honest completeness.
    Exact transcript equivalence to the single-`Q` protocol is neither possible
    as a direct carrier instantiation nor required. The direct SplitNc
    bad-event bounds and changed challenge/error accounting remain open;
    implementation correspondence alone cannot close them.
    Production/native refinement must also reconcile three
    different message languages. Native SumCheck accepts coefficient lengths
    from zero through its configured ceiling and interprets the empty list as
    zero. The current circuit path rejects only an overlong list, then panics
    when an allowed empty list reaches its nonempty Horner gadget. Lean's
    generic canonical finite message rejects empty lists and redundant trailing
    zeros. The independent NC layer now uses a protocol-specific exact
    five-coefficient carrier and explicitly accepts the padded five-zero
    polynomial while rejecting raw widths 0, 1, 4, and 6. Production adoption
    still requires native and R1CS acceptance/rejection parity, sequential
    transcript replay, and cost-tree reconciliation. The independent NC
    ceiling is four, but the current fixed production shape can still allocate
    wider shared FE/NC-tail frames. This is first a correctness and shape
    repair, not evidence of a row reduction.
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
  - PiRLC: the five 54-lane production `X` rings now decode directly into the
    typed 270-coordinate Phi81 public carrier. The list-level
    `phi81Combine` is proved equal to the independent typed public-input action,
    and exact projection-reduction artifacts imply the typed public-input
    equation without consuming the caller-supplied `AlgebraRefinement.x`
    field. The lane-major packed-column decoder is now proved equal to that
    direct typed decoder, and the typed output is proved equal to the typed
    strict-PiDEC parent. Production R1CS satisfaction must still construct the
    named artifacts; exact sampled-challenge provenance and the remaining
    commitment/evaluation role wiring remain open.
  - PiDEC: the production 270-cell public-input layout is now decoded into the
    typed Phi81 carrier by an explicit, bijective lane-major-to-block-major
    permutation. Strict semantic acceptance implies the exact typed public-input
    recomposition equation. This theorem starts from `PiDec.StrictAccepted`; it
    does not yet prove that production R1CS satisfaction establishes strict
    acceptance. Zero-tail authority, commitment and evaluation decoding, the
    production `Concrete.relationSemantics` split/recompose algebra, and private
    child CE openings remain required for knowledge reduction.
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
    At the local candidate-acceptance boundary, the active Lean model now
    independently proves that the proposed seven paired output-bit equations,
    one radix-three product aggregate, and one root binding are sound,
    complete, and uniquely extend the sixteen Boolean source bits to the
    verifier-owned rejection decision. The aggregate proof establishes both
    radix images are below the Goldilocks modulus before eliminating modular
    aliases. Separate countermodels show each of the three retained families
    is necessary relative to the other two. This is model-level only: exact
    recursive bit decoding, Boolean-row ownership, physical row placement,
    and fixed-selector behavior remain separate artifact/refinement gates.
    The production outer-image exporter passes on a 64-chunk private sampler
    fixture with sparse terminal-bit decoding, but that fixture is not the
    fixed F' relation. After removing a non-semantic eager reservation across
    all 56 gate-matrix triplet vectors, the actual recursive materializer was
    attempted under the mandatory 300-second cap. Compilation consumed 37.91
    seconds and the test ran for roughly 4 minutes 22 seconds before the cap
    killed it without an outer-image census; peak RSS was not recovered. The
    temporary caller was removed. Moreover, the deprecated local generator
    records gate arity 48 while the active schema has arity 56. Consequently
    neither the timed-out recursive attempt nor the deprecated artifact is
    conformance evidence. The fresh arity-56 leaf artifact and leaf-local
    semantic refinement are now closed. A selective or otherwise tractable
    960-chunk production outer-image bridge remains open.
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
  - Proved (model-level): the proposed one-chunk aggregate-acceptance relation
    is sound, complete, and uniquely extendable against the verifier-owned
    production rejection predicate, with a no-wrap radix-three proof and
    independent necessity countermodels for output bitness, the product
    aggregate, and root binding. Exactness assumes `EuclidPrime goldilocksP`
    and `SevenNonresidue`. Proved separately (artifact-checked, leaf-local):
    the active schema-2 arity-56 payload has 40 role bindings, nine normalized
    rows, and 25 sparse polynomial terms; a handwritten evaluator proves its
    seven Boolean-pair rows, product aggregate, and root binding equivalent to
    the independent relation and verifier meaning. Recursive source decoding,
    physical placement, selectors, the 960-chunk image, and row removal remain
    open.
  - Proved: deterministic replay exactness for the current partial NIFS
    carrier, plus formal blindness to FE/NC envelopes and PiCCS output points.
  - Proved: an independent Section 7.3 statement over the sole production
    source family; exact equivalence with SplitNc semantic truth and
    uncompressed residual truth; impossibility of directly instantiating the
    paper square `ColumnLayout` with a complete Phi81 carrier; verifier-owned
    alpha/beta/gamma coin projection from one Poseidon2 execution; exact
    mixed-width FE/NC carrier round trips; one complete FE-to-NC schedule; and
    honest message-before-challenge construction yielding accepted,
    source-bound exact transcripts.
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
  - Proved: one independent concrete Phi81 fixed-active NIFS composition with
    exact one-fresh plus fourteen-running source ownership, checked incoming
    parent authority, a single typed SplitNc PiCCS -> PiRLC -> PiDEC dataflow,
    complete parent-and-children results, exact physical evaluator
    characterization, conditional semantic soundness with explicit output
    binding and named bad-event exclusion, and honest completeness without a
    hidden total-sampler premise. The canonical honest PiCCS prefix either
    extends to an accepted NIFS/F' result or exposes one exact coordinate whose
    fixed 54-of-64 sampler prefix shortfalls. The older theorem taking one
    challenge vector valid after every accepted PiCCS certificate remains only
    as a stronger compatibility surface
    (`ConcretePhi81.{Context,Accepted,Holds}`,
    `ConcretePhi81.complete_or_samplerShortfall`,
    `FixedActive.{ResultTransition,Evaluator.run_complete_or_samplerShortfall}`,
    `SemanticPremises.exists_resultTransition_or_samplerShortfall`,
    `ActiveEvaluator.{run_eq_some_iff_physicalChecks,run_sound_of_closure,
    exists_run_and_holds_or_samplerShortfall}`).
  - Proved (model-level): the canonical fixed-active carrier constructs
    structure, stages, parent presence, and source consistency by construction;
    its incoming checked-parent verifier retains exactly shared point plus the
    three strict PiDEC recomposition families. A deterministic raw-certificate
    checker then composes that incoming boundary with Split-NC PiCCS, the exact
    54-of-64 sampler, and outgoing PiDEC, and accepts exactly
    `ConcretePhi81.Accepted`
    (`FixedActive.Canonical.{Context.sourceStructures,
    RunningAuthority.accepted_iff_equations,
    Checker.check_eq_true_iff_accepted}`).
  - Proved (model-level): the fixed-one F' carrier removes selected slot, raw
    prior counter, relation structure, and stages from prover-controlled input.
    Its checker retains the two outer equations plus the canonical raw NIFS
    verifier; its fail-closed evaluator computes the complete output and is
    exact to those physical checks. Conditional soundness and honest
    completeness are exported separately so physical acceptance cannot be
    mistaken for source/output authority or a security reduction
    (`ActiveEvaluator.FixedOneCanonical.{check_eq_true_iff_accepted,
    run_eq_some_iff_physicalChecks,run_sound_of_closure,
    exists_run_and_holds_or_samplerShortfall}`).
  - Proved (model-level): the canonical outgoing `Pi_RLC` attempt retains only
    source-structure consistency as a public verification obligation. Fresh
    input stage, shared point, combined parent stage, commitment, public input,
    and evaluation equations are exact consequences of the verifier-computed
    `OutputProduct.materialize` and `PiRLC.combinedOutput` dataflow
    (`ConcretePhi81.DerivedPiRlc.equations_iff_sourceStructures`). This is an
    obligation elimination theorem, not yet permission to delete arithmetic
    needed to compute the parent or any production row.
  - Proved (model-level): the fixed Phi81 54-of-64 sampler is now decided by a
    deterministic Boolean batch checker exact to the prior proof-carrying
    sampler boundary
    (`ConcretePhi81.Sampler.Checker.certificateCheck_eq_true_iff_accepted`).
    This removes an existential checker oracle from the semantic verifier; it
    does not establish Poseidon2/Rust/R1CS conformance or authorize row
    removal.
  - Proved (model-level): the canonical outgoing `Pi_DEC` certificate carries
    only each child's commitment, public input, and evaluations. Child
    structure, point, and fresh stage are inherited from the verifier-computed
    parent, whose combined stage is already canonical. Generic `PiDEC.Accepted`
    is exactly equivalent to the remaining commitment, public-input, and
    evaluation recomposition equations
    (`ConcretePhi81.DerivedPiDec.accepted_iff_recomposition`). This removes
    duplicate semantic authority from the model. It does not yet authorize
    deleting arithmetic that computes recomposition, nor any production row;
    exact Rust/R1CS refinement and row ownership remain required.
  - Proved (model-level): one explicit 270-coordinate honest-baseline source
    satisfies the independent CCS, norm, and carried-evaluation statement
    (`HonestBaseline.Sources.paperHolds`); a valid combined opening with its
    exact canonical PiDEC children constructs strict checked incoming authority
    (`HonestBaseline.RunningAuthority.accepted_of_combinedOpening`); and a
    deliberately degenerate zero-key, `Unit`-transcript context binds those
    pieces into `HonestBaseline.Context.semanticPremises`. Its repeated
    candidate value two decodes to the centered-zero coefficient, and generic
    first-accepted executions construct the exact bounded all-zero RingF batch
    (`HonestBaseline.Context.samplerBound`). This closes
    `HonestBaseline.Context.honestPremises` and yields one physically accepted
    certificate with an independent fixed-active semantic result transition
    (`HonestBaseline.Context.exists_resultTransition`). This model fixture does
    not establish production provenance, Poseidon2, artifact or Rust/R1CS
    conformance, a complete outer F' invocation, or authorization for row
    removal.
  - Proved: an exact generic six-family fixed-active F' obligation plan over
    the actual typed input/selected-slot/fold-result language. For the approved
    one-slot model profile, dispatch is derived already on the raw carrier, so
    the raw exact plan retains five families. A smaller canonical carrier then
    omits the necessarily-one prior counter and reconstructs the fresh relation
    structure from verifier-owned setup. Its exact plan retains only positive
    iteration, the prior public-input link, and selected NIFS; prior-slot,
    expected-structure, and dispatch are recorded in an exhaustive disjoint
    eliminated ledger and proved from construction
    (`FixedOneCanonical.{obligations_iff_active,holds_iff_active,
    holds_projection_iff}` and
    `ObligationPlan.FixedOne.{Raw.exact,Canonical.eliminated_hold,
    Canonical.exact}`). The generic theorem that a complete actual witness set
    establishes inclusion-minimal soundness remains available for profiles
    where all six checks are genuinely independent
    (`ObligationPlan.Global.{exact,lift_local_necessary,Witnesses,
    inclusionMinimalSound}`). The canonical global case language ranges over
    both the complete input and selected result, so a removed input check is
    not accidentally fixed before its countermodel is chosen. One explicit
    shared model setup and machine now supplies concrete removal witnesses for
    all three retained canonical families and closes model-level
    inclusion-minimality
    (`ObligationPlan.FixedOne.Minimality.{iteration_necessary,
    priorPublicInput_necessary,selectedNifs_necessary,
    inclusionMinimalSound}`). This is a logical-independence result for the
    canonical obligations, not production-machine conformance or a gate-count
    lower bound.
  - Proved: actual-type conditional removal interfaces for iteration,
    prior-slot, prior-public-link, selected structure, selected NIFS, and
    dispatch. The selected-NIFS witness deterministically mutates only the
    derived parent stage while preserving every child and is rejected by
    parent uniqueness. The other families retain explicit bad-input and
    context-stability premises; none is represented only by a Boolean proxy.
    Their independent honest-baseline constructors now return either the
    actual typed removal witness or one exact bounded-sampler shortfall; no
    necessity theorem needs one fixed challenge vector to work after every
    accepted PiCCS certificate.
  - Proved (production public-input slice): the strict PiDEC compiler's 270
    active `x` cells are lane-major (`lane * 5 + block`), while the typed Phi81
    relation is block-major (`block * 54 + lane`). The bridge proves the exact
    transpose bijection, decodes parent and children into the typed carrier,
    proves decode commutes with scalar recomposition, and derives typed
    recomposition from strict semantic acceptance
    (`NifsPaper.PiDec.PublicInputBridge.{packedSlot_exact,decode_combine,
    strictAccepted_typedPublicInputEquation}`). This is a representation and
    semantic-equation theorem only: it emits no constraints and authorizes no
    row removal.
  - Proved (production public-input slice): the five production PiRLC `X`
    rings decode into the same typed carrier as the lane-major packed columns;
    production `phi81Combine` equals the independent typed Phi81 action; exact
    projection artifacts imply the typed combination equation; and the result
    is exactly the typed explicitly named parent claim
    (`NifsPaper.PiRlc.PublicInputBridge.{decode_assembledX,
    decodeXRings_phi81Combine,
    typedOutput_eq_parent_of_refinement}`). The original local-layout
    specialization is deliberately conditional: generated PiRLC projection
    traces use global F' columns, while strict PiDEC layouts use local columns.
    `NifsPaper.RelabeledCarrier` now defines the structural map and proves that
    commitment, public-input, and point decoding commute with
    `Relabel.assignment`; consequently
    `typedPiRlcPiDecPublicInputComposition_relabel` closes the typed public-input
    composition for an explicitly relabeled parent claim. These are conditional
    representation/algebra theorems, not proof that R1CS satisfaction
    constructs the artifacts or permission to remove rows.
  - Proved (model-level carrier codec): `PiRlc.CarrierCodec.canonical` owns
    commitment flattening, the lane-major five-ring `X` transpose, and a
    matrix-count-indexed `RingK` evaluation encoding. Its
    `canonical_artifact matrixCount` theorem removes caller-supplied layout
    laws, but deliberately makes no production-conformance claim. This closes
    value-level serialization only, not row decoding or value authority.
  - Proved (model-level fixed-point shape contract):
    `FPrimeFullHistorySelectiveCcs.FixedPointShape.Refinement` checks equality
    of the terminal, selective, and materialized headers; derives the exact
    270-field public width; and checks that emitted matrix count, polynomial
    arity, and independent selective-relation arity are all thirteen. No
    concrete Rust snapshot is checked yet, and matrix payloads, assignment
    ordering, compiler convergence, and exact layout padding remain outside
    this theorem. Fixed-three-row artifacts therefore remain diagnostics.
  - Proved (model-level payload placement boundary): semantic matrix roles are
    now bijective with physical ports `0..12`; a successfully decoded compact
    bundle can be dimension-matched to the fixed-point header and transported
    into the role-indexed relation with only the independent polynomial. No
    complete production bundle exists yet, and this theorem does not derive
    matrix coefficients from F' semantics. Canonical assignment/matrix
    construction and equality to the decoded payload remain mandatory before
    artifact evidence can establish production conformance.
  - Proved (artifact-checked diagnostic carriers): `PiRlc.TraceCarrier`
    extracts `23 + 2 * matrixCount` paper-public leaves by type, excluding
    evaluation padding and delayed NC. The current generated recursive and
    terminal fixtures explicitly instantiate `PiRlc.DiagnosticProfile` with
    three evaluation rows, 29 public traces, and two delayed-NC traces.
    Their widths and parent-column identities are checked with guarded
    `native_decide`; they are diagnostics, not the active production shape.
    `PiRlc.ClaimShapeAlignment` proves that both fixtures fail the independently
    derived thirteen-matrix selective-relation shape. No `13 -> 3`
    compression theorem exists, so these artifacts cannot authorize row
    removal or production acceptance. Removing the remaining artifact-level
    `Lean.trustCompiler` dependency requires a proved indexed column-map
    generator, not another replay of the same generated lists.
  - Missing: a complete Rust/R1CS decoder refinement into the canonical carrier and a
    production-conformant instantiation of the independently proved semantic
    boundaries. The generic six-family witness bundle is no longer the right
    minimality target for this profile because Lean proves three of its
    families derived. The zero-valued 270-coordinate model fixture closes
    logical inclusion-minimality of the remaining three, but it cannot
    exercise production source ordering, matrix/lane placement,
    commitment-key sensitivity, nonzero PiDEC recomposition, Poseidon2, or
    decoder authority. Therefore production row removal is not yet
    established.
  - Missing: direct bad-event/error bounds, external Boolean-leaf/bit-order
    refinement,
    concrete CCS/norm/ring-coefficient and base-to-extension refinements,
    production residual placement, construction of SplitNc output-source
    authority, root counting, complete-carrier and concrete-stream/
    fixed-bound/strong-set/distribution production sampler refinement, and
    Poseidon2/Rust/R1CS refinement.
  - Missing: Construction-2 hash-binding and full recursive NIVC
    knowledge-soundness theorem over the explicit NIFS bad events.
  - Missing: production PiCCS Accepted refinement.
  - Missing: production PiRLC Accepted refinement.
  - Missing: R1CS-satisfaction-to-strict-production-PiDEC acceptance,
    zero-tail authority, commitment/evaluation decoding, and private-opening
    refinement. The typed public-input recomposition slice is closed separately
    above.
  - Missing: concrete Composition.fold_knowledge_or_bad_event instantiation.
  - Missing: Rust/R1CS decoder refinement into the canonical fixed-one physical
    evaluator and exact ownership of every resulting row. The production
    lifecycle still has an empty-running bootstrap arm, a 257/270 carrier
    mismatch, omitted `y_zcol` authority, and no single typed Rust invocation
    matching the Lean canonical `Context`.

axiom_report:
  The independent NIFS and partial replay theorems are in
  `tests/Axioms/Paper.lean`; finite SumCheck, generic first-accepted sampling,
  and PaperJoint have dedicated fail-closed guards in
  `tests/Axioms/{SumCheckFinite,FirstAccepted,PiCCSPaperJoint}.lean`.
  Fixed-active F' evaluator, semantic-boundary, exact-plan, and conditional
  necessity theorems, together with the honest-baseline source, strict running
  authority, and semantic-premise theorems, are guarded in
  `tests/Axioms/Protocol.lean`. All are imported by the aggregate axioms gate.
  The bounded `validate.sh static`, `build`, `axioms`, and `check` gates passed
  with this model-level milestone present. The Rust/R1CS bridge report remains
  open. Every completed bridge theorem must be added to the gate; local
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
