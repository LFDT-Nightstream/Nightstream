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
  - This protocol-first work treats Poseidon2 and Ajtai as selected primitives.
    It must prove their protocol inputs, parameters, domain separation,
    transcript/commitment dataflow, recomposition, and explicit hash-collision
    or binding-failure boundaries, but does not re-prove the permutation,
    commitment construction, or underlying hardness assumption.
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
  - Public-input carrier: the logical F-prime instance exposes 257 field
    inputs, while the production CE path carries five complete ring columns
    (270 coefficients).
    The final 13 coefficients are not generic zero padding after PiRLC: ring
    multiplication may make them nonzero and PiDEC intentionally recomposes
    them. The active Phi81 relation now owns a typed 270-field public carrier,
    inserts thirteen verifier-fixed zeros in fresh assignments and matrices,
    and proves exact scalar CCS residual/zero-set preservation. This does not
    preserve nonconstant coefficient images or commitments: old private column
    257 changes block/lane under the repair. Rust's selective fixed-point path
    now constructs the physical 270-field source, pins coordinates `257..269`
    to zero, shifts the private suffix, and uses 270 as the SuperNeo public
    width. `AlignedCompiler.ProductionCarrier` artifact-checks the emitted
    public layout and all thirteen padding rows against the typed zero-pin
    semantics. The bounded stabilized fixed-point projected-emitter artifact
    now also exports every public-coordinate owner from its actual prepared
    layout in disjoint 256- and 14-record chunks. Lean proves that decoder is
    exactly constant-one, direct source fields `1..256`, then thirteen fixed
    zeros, and that its interpretation equals the independent typed public
    projection under the explicit source constant-one condition. This closes
    public-prefix connectivity for that bounded fixed-point profile only. The
    active post-PiDEC execution audit now separately joins every one of those
    270 writes across the live builder, normalized public assignment, and
    packed committed witness. Its two 135-record generated shards also carry
    the exact normalized source, width, centeredness, and alias policy. Lean
    decodes them into the canonical one-arm trace and derives the typed public
    projection without a caller-supplied exporter or public-dataflow premise;
    the conventional constant-one source fact remains separately owned. The
    complete private assignment and matrix map, CE coefficient images, aligned verifier-owned
    Ajtai key, commitments, and full decoder refinement remain open; neither
    truncation nor claim reuse closes those bridges.
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
    `y_zcol` that is not source-bound. The legacy sidecar-only order derived the
    NC test point before validating that value, so it could not be justified as
    an ordinary non-adaptive polynomial test. Production now uses the delayed
    raw-witness path described below; the counterexample remains the reason the
    public sidecar is transport rather than authority. The delayed
    old-point model is now exposed through the generic `PiCcsNc` facade and a
    fail-closed regression/axiom guard. Its fixed-270 refinement contract
    constructs the child table only from
    `SplitNc.Sources.Data.runningAssignments`, proves exact child/column
    lookup and 512-by-64 coverage, and specializes the 54-active plus
    ten-padding exact-or-`BadRoot` theorem to that table. It does not assume or
    read output `y_zcol` sidecars. This flat 9+6-round relation is a bounded
    diagnostic, not the active production statement: it has 512 flat columns,
    whereas the current active profile has 19 block rounds plus six lane rounds over
    11,437,038 coordinates. The production correspondence therefore proves the
    same old-point obligation as an exact 54-lane
    `PackedYZcolBoundAtBlock` equality; it must not coerce the fixed-270
    `OldPointSumcheckRelation` across that dimension boundary. The active Rust
    handoff is now the versioned raw block×lane path: native prove/verify read
    complete fresh `CcsWitness.Z` and ordered running `Mat` tables and the
    recursive circuit carries a typed pending projection. The older
    `running_output_evaluation` helper still reads `CeClaim.y_zcol`, but it is
    retained only as a diagnostic export and is not the production authority
    path. Lean owns the matching combined-NC
    checker over complete packed `Z` witnesses, the explicit
    `batchWeight = 0`/residual-root branch, the producer-beta projection root,
    fixed-degree SumCheck collision events, typed transcript domains and
    order, and adjacent/base/terminal one-fold composition. Its load-bearing
    conclusion is `SemanticFold.Holds`; the convenience projection to
    `Semantics.Paper.Holds` is weaker because it drops the parent/children
    equalities. Native terminal verification recomposes the pending parent
    from the authoritative full `WitnessMat` children. The fixed-profile Rust
    exporter now exposes those same fourteen ordered raw matrices, the
    `pending.old_block` and parent columns, and the exact terminal projection
    rows and physical placement. Lean derives terminal projection acceptance
    from satisfaction of those generated rows plus terminal CE, without a
    caller-provided execution/refinement premise. The recursive-circuit parent
    binding remains open. Generated full-`Z` geometry and the final 28
    physical alignment rows are artifact-checked. Exact sparse-row refinement
    for the other combined-NC, pending-state, and terminal equations,
    semantic-input rows, production PP coefficient equality, and accepted
    Ajtai opening/extraction remain explicit open edges. Lean also proves
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
  - Proved (model-level security partition): Construction-2's exact typed
    prior/next hash preimages contain the complete ordered running product.
    Given the lifecycle public-input link, the current running product equals
    the previous output product unless either the instance encoding aliases two
    digests or the hash aliases two distinct typed preimages
    (`Paper.PriorLink.{preimage_eq_or_securityFailure,
    running_eq_or_securityFailure}`). The concrete Phi81 projection combines
    that complete-child equality with canonical PiDEC children to recover the
    exact rich running slot. This refutes parent-only authority at the model
    boundary; it does not yet prove that Rust's nested accumulator digest and
    outer `state_x_out` Poseidon2 serialization instantiate the typed preimage
    or bound either collision event.
  - Proved (model-level obligation reduction): for strict PiDEC children, the
    child-specific authority is exactly commitment, the complete 270-coordinate
    public input, and the complete ordered evaluation array. Structure and
    evaluation point are inherited by every child and need occur only once per
    family; fresh stage is verifier-fixed and need not be serialized. Equality
    of this compact family carrier recovers both the exact ordered children and
    the checked parent cache
    (`ChildPayloadAuthority.parent_children_eq_of_familyPayload_eq`,
    `ConcretePhi81.AccumulatorBinding.parent_children_eq_or_failure`). The
    active prior-link theorem lifts this
    to exact rich-slot equality or one named encoding/hash failure
    (`ActiveSemantics.PriorLink.slot_eq_or_familyDigest_failure`). This result
    rules out blindly rehashing every inherited field in every child, but it
    does not yet classify Rust-only sidecars or prove a concrete serializer;
    therefore it authorizes no production row removal yet.
  - Proved (model-level conditional authority reduction): when relation
    structure is separately bound to verifier-owned setup and both candidate
    child families have explicit valid CE openings, the direct per-step state
    carrier reduces further to the evaluation point once plus the exact
    type-level ordered child commitment vector. Equal carriers recover every
    child public input and evaluation from the openings and then recover the
    strict PiDEC parent, or identify one child index with two distinct
    `b`-bounded openings of the same commitment
    (`ChildCommitmentAuthority.{children_eq_or_freshBindingCollision,
    parent_children_eq_or_freshBindingCollision}`). The nested-hash and active
    prior-link lifts preserve an exhaustive failure partition between concrete
    encoding/hash failure and that indexed opening collision
    (`ConcretePhi81.AccumulatorBinding.parent_children_eq_or_commitmentFailure`,
    `ActiveSemantics.PriorLink.slot_eq_or_commitmentDigest_failure`). Arity and
    relation structure are deliberately absent from this payload: arity is in
    the type and structure is a separate setup obligation. This is the smallest
    direct paper carrier presently justified under arbitrary valid child
    openings. It does not prove that the
    current recursive verifier extracts the required current child openings,
    reduce the collision to the production Ajtai/MSIS game, classify Rust-only
    sidecars, instantiate Poseidon2, or authorize row removal.
  - Proved (model-level stronger conditional reduction): if both child families
    are not merely valid but are the deterministic radix split of explicit
    valid combined parent openings, the complete family collapses further to
    one per-step evaluation point plus one combined parent commitment. Equal
    carriers recover the exact parent opening, parent statement, and ordered
    children, or expose two distinct `B`-bounded openings of the same parent
    commitment (`CanonicalParentAuthority.{parent_opening_eq_or_bindingCollision,
    parent_children_eq_or_bindingCollision}`). The nested-hash and active
    prior-link lifts retain separate compression and parent-opening failures
    (`ConcretePhi81.AccumulatorBinding.parent_children_eq_or_canonicalParentFailure`,
    `ActiveSemantics.PriorLink.slot_eq_or_canonicalParentDigest_failure`). This
    is the smallest direct paper carrier currently proved, but its premise is
    intentionally stronger than public PiDEC acceptance: the current verifier
    does not establish canonical private child openings, and Lean already proves
    signed-digit substitutions pass public recomposition. No implementation
    should select this carrier until the canonical-opening check is refined to
    production wires and shown cheaper than hashing the ordered child
    commitments.
    The active prior-link composition now removes the free
    `currentCanonical` proposition from this candidate boundary: an
    opening-derived NIFS context plus exact child-source validity derives the
    canonical family and therefore reduces equal parent handles to exact slot
    equality or the named compression/parent-opening failure
    (`ActiveSemantics.PriorLink.slot_eq_or_canonicalParentDigest_failure_of_openingSources`).
    This states the proof-backed contract precisely; it does not show that the
    present public-only Rust NIFS certificate establishes child-source validity
    for the computed split.
  - Proved (model-level obligation exactness and minimality): a canonical-parent
    opening verifier now accepts a point-plus-parent-commitment carrier and one
    private combined assignment, computes the parent public input, evaluation
    array, combined stage, and all fourteen radix-split children, and checks
    only obligations not true by construction
    (`PiDEC.CanonicalChildren.OpeningVerifier`). The generic relation retains
    commitment equality, the verifier-owned `B`-norm, and point validity. In
    the typed Phi81 specialization, point validity is intrinsic to
    `Point shape`, so exact parent CE membership and canonical PiDEC children
    follow from exactly two semantic leaves: typed Ajtai commitment equality
    and the complete-assignment combined norm
    (`CanonicalParentVerifier.parentHolds_iff_commitment_and_norm` and
    `CanonicalParentVerifier.canonicalChildren_of_commitment_and_norm`).
    Concrete Boolean-commitment
    and magnitude-`16_384` countermodels over a small typed Phi81 fixture show
    that deleting either retained family uniformly admits an invalid parent opening
    (`CanonicalParentVerifier.Minimality.plan_inclusionMinimalSound`). This
    closes the semantic specification of the canonical-opening check. It does
    not extract the opening from the current certificate, prove Ajtai/MSIS
    binding, materialize the computed fields in R1CS, or establish their cost.
  - Observed (source-level implementation diagnostic, not formal conformance):
    the current `NifsProof` serializes the PiCCS, PiRLC, and PiDEC public
    messages only. `pi_rlc::Output.witness` and `pi_dec::Children.witnesses`
    remain prover-side, and `NifsVCircuitMessages` supplies the combined claim
    and child claims without the combined assignment. The current recursive
    circuit therefore cannot instantiate the proved canonical-opening verifier
    without adding an opening witness or a proof-backed opening boundary. A
    complete cost comparison must include that new boundary; the parent hash
    cost alone is not an implementation candidate.
  - Proved (model-level carrier reduction): one point-plus-complete-assignment
    payload deterministically computes the combined parent and all fourteen
    ordered children under a verifier-owned key and relation structure. Every
    independent fixed-active NIFS result has such a payload and reconstructs
    both public result surfaces exactly
    (`CanonicalOpening.{OpeningPayload.children_ofCanonical,
    resultCarrier_complete}`). In the opening-derived incoming context, CE
    validity of the fourteen computed child sources implies the combined norm,
    canonical children, strict public PiDEC, and the existing running-authority
    interface (`CanonicalOpening.{combinedNorm,canonicalChildren,
    piDecAccepted,runningAuthority}`). The norm obligation is not deleted: its
    owner moves to child source validation. This is semantic representation
    completeness, not a compact public encoding: the assignment has the full
    CCS/CE carrier width, not the 270-field public width. Hashing that raw
    opening would scale with the entire recursive relation and may leak a
    low-norm private witness. It is therefore rejected as the accumulator
    handle; the useful result is the derived-check theorem, not a serializer.
    This is not Rust/R1CS refinement or permission to remove rows.
  - Proved (model-level): the ordered-child-commitment carrier does not need a
    second opening witness at the outer F' boundary. One accepted independent
    NIFS result contains a single source witness whose running assignments
    open all fourteen incoming CE children
    (`ResultTransition.inputRunningOpenings`), and the same realization
    exposes strict PiDEC recomposition for the exact carried parent
    (`FixedActive.ResultTransition.inputRunningPiDec`). The active F' context
    maps those values definitionally to the selected current slot
    (`Obligations.selectedInputAuthority`). Consequently, equal ordered-family
    handles recover the exact prior slot or one named compression/child-opening
    failure without accepting extra outer opening evidence
    (`PriorLink.slot_eq_or_commitmentDigest_failure_of_selectedNifs`). Its
    previous-family authority is likewise derived from the previous NIFS
    result transition under an explicit same-verifier-key equation, never
    supplied as an independent canonicality claim. This is still conditional
    on reaching both independent NIFS semantic transitions; it does not
    discharge physical transcript extraction, Ajtai/MSIS binding, Poseidon2
    collision bounds, Rust/R1CS refinement, or authorize row removal.
  - Proved (model-level representation refinement): the two reduced carriers
    now have one exact field order independent of Rust's legacy full-claim
    serializer. Point coordinates are encoded as ordered `(c0, c1)` pairs;
    commitments are child-major, row-major, then coefficient-major. Both
    encoders are injective for every typed shape, arity, and verifier-row count,
    including the zero-row edge case
    (`CarrierCodec.encodeCommitmentFamily_injective` and
    `CarrierCodec.encodeCanonicalParent_injective`). At the
    fixed Phi81 profile the carrier-only lengths are exactly
    `2 * rowVariables + 13_608` fields for the ordered fourteen-child
    commitment family and `2 * rowVariables + 972` fields for the canonical
    parent. Instantiating the generic accumulator scheme with either encoder
    proves that its serialization-collision branch is impossible; equal claim
    digests reduce directly to exact payload equality or the still-explicit
    hash-collision boundary
    (`{commitmentFamily,canonicalParent}_claim_eq_or_hashCollision`). These
    carrier-codec theorems alone do not choose a domain tag or establish
    Poseidon2/Rust/R1CS conformance.
  - Proved (model-level representation refinement): the ordered-child carrier
    now has one exact field-hash message independent of all legacy full-CE and
    nested-child-digest serializers. Its unique tag is
    `neo.fold.clean/f_prime/accumulator/ordered_child_commitments/v1`, packed
    exactly as Rust's length-prefixed seven-byte little-endian field words,
    followed immediately by the injective point-plus-fourteen-commitment
    carrier. Commitment width, child count, and point length are verifier-owned
    by the specialized profile/type and are not redundantly serialized. The
    complete message is injective, has exactly
    `10 + 2 * rowVariables + 13_608` fields, and equal supplied field-hash
    outputs imply exact carrier equality or a collision on two distinct field
    lists (`OrderedCommitmentMessage.{serialize_injective,
    digest_eq_or_fieldHashCollision}`). Composing this result with independent
    selected-NIFS input authority recovers the exact prior rich slot outside
    only that field-hash collision and one indexed Ajtai opening collision
    (`OrderedCommitmentPriorLink.slot_eq_or_failure_of_selectedNifs`). This
    closes the model-level message boundary; it does not prove that native Rust
    or R1CS recomputes this message, nor bound either cryptographic event.
  - Proved (artifact-checked layout plus model-level decoding): a prospective
    direct hash owner over the committed full-history PiDEC layout must read
    exactly ten domain constants, the relabeled parent `r` pairs, and all
    relabeled child `c_data` blocks in child order
    (`OrderedCommitmentSourceLayout.expectedSourceColumns_values`). The
    currently committed artifact has one extension-field point coordinate, so
    this source list is 13,620 fields: `10 + 2 + 14 * 972`. This is deliberately
    distinct from the 13,642-field formula at the separate twelve-variable
    Fibonacci diagnostic shape. The current Rust owner still emits fourteen
    conservative full-CE child hashes plus an aggregate hash, so this source
    layout is a checked implementation contract, not a claim of present
    call-site conformance. Exact-length decoding now maps the artifact's parent
    point and every 972-field child block into the independent typed ordered
    carrier without padding or default reads, and reserialization is proved
    equal to the complete raw field message
    (`OrderedCommitmentTypedDecoder.serialize_decodedPayload`). This theorem is
    deliberately shape-parameterized by an explicit point-dimension proof: it
    does not identify the one-coordinate/three-evaluation diagnostic artifact
    with the active production relation. Emitted Rust/R1CS trace membership and
    selected-NIFS semantic instantiation remain open.
  - Proved (artifact-checked primitive profile plus model-level composition):
    the existing generated width-eight Poseidon2 permutation contains exactly
    600 raw R1CS rows/fresh columns, partitioned into 344 S-box product rows and
    256 linear rows. The rate-four hash wrapper therefore has, for `L`
    already-materialized input fields and
    `P = ceil(L / 4) + 1`, exactly `L + 2` wrapper-linear rows,
    `344 * P` product rows, `256 * P` permutation-linear rows, and
    `L + 2 + 600 * P` total raw rows/fresh columns
    (`FPrimeFullHistory.Accumulator.Poseidon2Cost`). The cost leaf specializes
    the ordered carrier to its exact ten-field domain message; the
    canonical-parent candidate still keeps `domainFields` explicit because it
    has no approved concrete message. It deliberately excludes
    domain-constant allocation, carrier wire materialization, actual call-site
    emission, native parity, and gadget-native lowering, so it is not an
    end-to-end production count. For the current Fibonacci diagnostic
    `rowVariables = 12`, the ordered-message formula has 13,642 input fields,
    3,412 permutations, and 2,060,844 raw rows. The zero-domain
    canonical-parent diagnostic remains 150,998 raw rows. A raw-opening hash
    is intentionally absent: its input length would use the complete
    assignment carrier, and substituting the 270-field public width there is a
    category error. The canonical-parent candidate omits
    R1CS lowering and cost accounting for the now-specified two-check
    canonical-opening verifier and all computed downstream fields; it is
    therefore not yet the cheaper complete design.
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
  - Proved (model-level active output representation):
    `PiCcsOutputDigest.ActiveSemantics` encodes the complete typed SplitNc
    output tree in source, matrix, Phi81-lane, and `(c0,c1)` limb order and
    proves that the resulting pre-SIS field serialization is injective. The
    fixed-active arity derives exactly fifteen sources. `ActiveProfile`
    constructs the canonical batch shape directly from that arity and the
    independent thirteen-port selective relation, proves that forgetting batch
    counts recovers the exact independent relation shape, and derives a complete
    23,033-field message without a caller-supplied source or matrix count
    (`relationShape_eq`, `selective_serialize_length`). It also proves that the
    legacy three-matrix projection cannot inhabit this active shape
    (`selectiveShape_not_legacyProfile`). This is a typed representation length,
    not a row/column count. Shared domain tags and the active fixed-width vector
    codec have one owner in
    `PiCcsOutputDigest.Encoding`; the old 6,683-field three-matrix serializer
    remains a diagnostic compatibility path rather than an active authority
    surface.
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
  - Artifact-checked (bounded `y_zcol` slice): the fixed-point
    selective generator exports exact source definitions, exact compact A/B/C
    rows, source and derived column provenance, ownership for every emitted
    polynomial-evaluation, product-sum, and retained-check row, and the
    separately eliminated linear-definition program from the same
    structure-term emission path used by final matrices. The Lean sources
    establish unique physical-row ownership, exact rewrite/retained
    coefficients, deterministic provenance and satisfaction for eliminated
    definitions, compact-row soundness into the independent source projection
    obligations, and canonical assignment construction from deterministic
    source execution with a constant-one seed plus the two direct sampled-wire
    equations.
    Artifact computations are projected to proof-free coefficient records or
    Boolean summaries before evaluation. Kernel theorems prove exact ordered
    partition coverage and the per-part certificate bound, so neither the
    coefficient bridge nor assignment materialization evaluates a full
    proof-carrying artifact list.
    The generator rejects any derived rewrite with a nonzero base or with a
    predecessor or ordered factor payload different from the actual witness
    encoder. Lean proves that the normalized decoded recurrence stream exactly
    equals that exported witness registry and ties each constructed derived
    field to its registered field recurrence.
    This reverse theorem assumes no source-row satisfaction or decoded
    acceptance, but remains focused projection-execution completeness rather
    than completeness of a full paper-honest PiCCS/NIFS transition. The
    soundness theorem is explicitly conditional on
    the externally owned steady-selector and constant-one facts; the compact
    projection rows are gated and cannot establish which branch is active.
    With separately supplied upstream producer-column and `y_zcol`
    message/source bindings, the result composes with `ActiveBridge` to yield
    the typed message aggregate or its named projection bad-root event. This
    closes only the bounded fixture: it does not establish selector
    enforcement, upstream output authority, full production-carrier
    refinement, CCS/CE membership, or permission to remove rows globally. The
    focused correspondence facade, regression, fail-closed
    axiom guard, static checks, executable checks, and Rust artifact drift test
    pass. Two repository-wide builds and the repository-wide axiom report hit
    their fixed 900-second limit in unrelated legacy dependency cones without
    reaching the memory cap; those incomplete broad gates do not extend or
    weaken this bounded evidence claim.
  - Artifact-checked (bounded fixed-point carrier prefix): the same prepared
    layout now exports the complete 270-coordinate public owner schedule, all
    13 public-padding rows, all 38 private-alignment rows, and the exact three
    selector-domain plus one selector-total rows. Lean identifies the public
    vector with the independent typed carrier, proves unique physical
    ownership and exact coefficients for all three row families, derives
    residuals `-(z[0] * z[257+i])` and `-(z[0] * z[273+i])` for the two padding
    intervals, proves the public rows equivalent to typed `FixedPublicPadding`
    under explicit coordinate agreement, derives the Boolean and sum-to-one
    selector equations, and constructs the honest zero and unit-selector
    extensions. This closes physical rows for `257..270`, compiler-owned
    interval `273..311`, and the four selector equations only. It does not
    decode the private values after that interval,
    cover every selector-gated retained row, decide whether the Boolean rows
    are removable, or close the complete matrix bundle.
  - Corrected authority boundary: the generated `14 × 270` decoder under the
    historical `RawRunningDecoder` path reads incoming `CeClaim.X` public
    coordinates. It is artifact-checked for that public-prefix provenance,
    but it is not a decoder for `CcsWitness.Z`, `CeWitness.Z`, the private
    suffix, or the complete production assignment. Lean now separately states
    the actual generic packed-witness contract
    `Z[lane, block] = assignment[block * 54 + lane]`, proves both inverse
    directions and fresh-tail zero padding, constructs running source data
    definitionally from complete packed witnesses, and composes the existing
    Boolean combined-NC checker across the explicit one-fold recursive and
    terminal boundaries. A compact generated certificate now artifact-checks
    all fourteen full-width matrix coordinate bijections, the 54+10 lane
    partition, all 108 one-hot cells of a two-block bounded
    `CcsInstance::from_low_norm_assignment`/Ajtai commitment probe, and the
    `Commitment.data[row*54+lane]` index. It explicitly distinguishes bounded
    fixture κ=4 from protocol production κ=18. Lean proves the generic
    commitment-data bijection and the production-width flattened
    `matrixCommit` equation; native production PP coefficient equality and
    accepted opening/extraction remain open. The
    earlier raw-authority interpretation of the 270-column artifact is
    retracted without weakening its valid public-prefix facts.
  - Corrected production combined-NC bridge: the active delayed residual uses
    the block×lane domain over all 11,437,038 assignment coordinates: 19 block
    rounds followed by six lane rounds, covering 211,797 live blocks. The
    current artifact needs only 18 bits for block coverage; the nineteenth bit
    is retained by the versioned protocol format. Each
    block has 54 physical Phi81 lanes and ten verifier-computed virtual zero
    lanes; those ten values are polynomial padding, not separately owned
    physical rows. The raw table is computed from complete packed
    `CcsWitness.Z`/`CeWitness.Z` matrices, never from child
    `CeClaim.y_zcol` sidecars. Lean proves the exact Boolean-cube and terminal
    formulas, quartic round representability, degree-one residual-weight
    identity (including `batchWeight = 0`), degree-53 producer-beta
    exact-or-root result, and an exact claims-level terminal at the FE-derived
    transcript point. Public `y_zcol` is transport only. For a recursive edge,
    successor NC truth gives genuine openings of the exact ordered raw running
    assignments; strict Π_DEC and an exact canonical-parent commitment/norm
    predicate then give exact parent recomposition or
    `ParentOpeningBindingCollision`. The direct active theorem does not assume
    either that predicate or the raw-child commitment predicate: it
    case-splits both and returns their negations as specifically owned binding
    failures. Lean proves that the raw-child negation is equivalent to
    existence of one verifier-indexed packed matrix whose exact `matrixCommit`
    differs from its public running commitment. Neither predicate contains
    public inputs, evaluation arrays, child sidecars, or digest authority. The
    accepted delayed identity therefore
    closes the predecessor packed equation or returns a specifically typed
    algebraic, SumCheck, commitment, state, input, key, or opening failure. The
    terminal path applies the same reduction to fourteen complete raw child
    matrices. The first step is explicitly no-pending and every output closes
    one fold later.
    `terminalChecked_implies_baseAllPackedAndAllPaper_or_parentOpeningFailure_or_paperFailure`
    proves `BaseNc`, every packed equation, and Construction-2 for every step;
    if the independent paper track fails, the result still retains `BaseNc`
    and every packed equation. Only the dedicated `y_zcol` parent-opening tree
    can prevent `AllPacked`, and it is separate from `yRingUnbound` and the
    other paper/refinement failures. Neither tree contains
    `OutputBindingFailure` or generic `outputUnbound`. The unanchored
    extraction helper and older generic evaluator remain diagnostic seams; the guarded theorem
    removes generic output-unbound once packed. The separate 9+6 flat model is excluded by `flatColumnProjection_not_actionHom`.
    The fixed-profile correspondence now mirrors the literal native
    `project_raw_witnesses_at_block_point` loop: all 211,797 blocks are
    traversed in order with little-endian block weights, each of fourteen
    `WitnessMat` children yields 54 active lanes, ten virtual lanes are
    computed as zero, and Π_DEC performs the ordered radix recomposition.
    The active Rust exporter now owns the exact terminal row program and
    concrete placement for this computation. The current factored program has
    24,185,169 physical rows and 24,185,061 allocated/committed columns over
    `[22,834,865, 47,020,034)`. The explicitly retained pre-factorization
    baseline was 25,243,884 rows and 25,243,776 columns; the kernel-checked
    final-round factorization therefore saves exactly 1,058,715 recurring rows
    and columns. It
    exports the pending old-block and parent columns, the fourteen ordered raw
    `WitnessMat` bases shared with terminal Ajtai openings, the selected
    recursive arm, and the fixed profile identity. Lean proves the indexed
    compiler/generated row equality, unique ownership of the complete physical
    interval, the generated column-map inverse, a transparent assignment
    decoder from outgoing pending state plus those raw matrices, semantic
    equivalence of the factored final block, and an honest satisfying
    assignment. Consequently
    `productionRows_projectionOpeningAccepted` derives the existing terminal
    authority object from actual generated-row satisfaction and terminal CE;
    no caller supplies `PackedWitnessExecutionBinding.Accepted`, a desired
    projection equation, a child `CeClaim.y_zcol`, digest authority, or a
    raw-old-block execution-refinement proposition. The fixed terminal placement and
    raw-matrix dataflow are artifact-checked/Rust-conformant.
    `terminalRawProjectionRowsChecked_implies_terminalChecked` lifts those
    exact rows and terminal CE to the complete executable terminal checker.
    The model-proved authority headline
    `terminalRawProjectionRows_imply_baseAndAllPacked_or_parentOpeningFailure`
    yields `BaseNc ∧ AllPacked` or the typed parent-opening failure tree.
    Base, recursive-predecessor, and terminal handling are explicit. The
    raw-old-block authority branch has no execution-refinement or generic
    `outputUnbound` possibility. The stronger `AllPaper` wrapper is not the
    authority headline: its separate failure branch still contains independent
    public-input, source-product, child-opening, key-refinement, and `y_ring`
    gaps. This result does not by itself close the recursive-circuit parent,
    the broader end-to-end paper/NIFS refinement, or primitive-binding tracks.
    Production native prove/verify now selects the distinct delayed block×lane
    header, binds the complete accumulator/pending-family handle before
    sampling `betaBlock`, `producerBeta`, and `batchWeight`, runs the 25-round
    raw-witness oracle, and rejects legacy-format replay and nonzero virtual
    padding. Recursive state carries the old block point and 54-lane parent for
    exactly one fold; base admits absence only for the canonical zero running
    state. The terminal path recomputes child projections from opened raw
    witnesses, performs exact radix recomposition, and rejects reattached
    child-sidecar tampering. The circuit mirrors the challenge suffix and
    terminal formula, and CUDA fails closed for the unsupported production
    shape. This active dataflow is `rust-conformant`; the legacy sidecar helper
    remains non-authoritative. Exact generated rows for the raw-old-block
    terminal projection are now closed for this fixed placement, and only the
    exact 1,058,715-row/column final-round replacement is authorized. Other
    combined-NC/state/terminal families, production PP coefficient equality,
    and accepted Ajtai opening/extraction remain open, so the claim is not
    `security-reduced` and does not authorize row removal outside this exact
    terminal raw-old-block family.
  - Artifact-checked fresh public-prefix decoder: the bounded Rust exporter
    now identifies all 270 coordinates of
    `prior_link.fresh_public_inputs[0]` in exact order and records their
    normalized source columns and fail-closed selective dispositions in
    proof-free shards of 256 and 14 records. Lean proves complete coverage,
    the exact consecutive column formula, and unique logical ownership. A
    separate theorem reaches only the fresh source-product `publicInput`
    field, conditional on explicit per-coordinate value bindings and direct
    field-value dataflow. The artifact does not infer values from decoder
    labels: the constant-one row, 256 bit-link rows, thirteen padding rows,
    the remaining fresh fields, full-witness values, and commitment authority remain open.
  - Missing: composition of the now artifact-checked active public-write
    trace, both padding intervals, and selector equations into a full private-assignment
    and final-matrix Rust/R1CS refinement of the canonical carrier and a
    production-conformant instantiation of the independently proved semantic
    boundaries. The generic six-family witness bundle is no longer the right
    minimality target for this profile because Lean proves three of its
    families derived. The zero-valued 270-coordinate model fixture closes
    logical inclusion-minimality of the remaining three, but it cannot
    exercise production source ordering, matrix/lane placement,
    commitment-key sensitivity, nonzero PiDEC recomposition, Poseidon2, or
    decoder authority. Therefore row removal for this generic carrier and
    minimality target remains unestablished outside the exact terminal
    raw-old-block factorization authorized above.
  - Missing: direct bad-event/error bounds, external Boolean-leaf/bit-order
    refinement,
    concrete CCS/norm/ring-coefficient and base-to-extension refinements,
    production residual placement, construction of SplitNc output-source
    authority, root counting, complete-carrier and concrete-stream/
    fixed-bound/strong-set/distribution production sampler refinement, and
    Poseidon2/Rust/R1CS refinement.
  - Missing: concrete serialization for the unconditional public-payload
    family carrier; exact domain-tag/message serialization for the conditional
    canonical-parent codec; refinement from Rust's currently nested full-claim
    accumulator serialization into the ordered-message codec; allocation and
    call-site ownership for every resulting wire; extraction of the combined
    parent opening from the current certificate; R1CS lowering and exact cost
    ownership for the proved commitment/norm canonical-opening plan and its
    computed parent/children; reduction of each fresh- or combined-bound
    opening collision to the production Ajtai/MSIS game;
    Construction-2 Poseidon2/instance-encoding binding; and the full recursive
    NIVC knowledge-soundness theorem over the explicit NIFS bad events.
  - Missing: production PiCCS Accepted refinement.
  - Missing: proof that the production fixed-NIFS context and Rust relation
    instantiate `ActiveProfile.selectiveShape`, plus the exact Rust/R1CS
    source-column decoder for its shape-indexed output serialization. Until
    both close, the representation and legacy-mismatch theorems authorize no
    output-hash row removal.
  - Missing: production PiRLC Accepted refinement.
  - Missing: R1CS-satisfaction-to-strict-production-PiDEC acceptance,
    zero-tail authority, commitment/evaluation decoding, and private-opening
    refinement. The typed public-input recomposition slice is closed separately
    above.
  - Missing: concrete Composition.fold_knowledge_or_bad_event instantiation.
  - Closed at the artifact-independent boundary: for any admissible codec
    profile and certified call recipes, satisfaction of the selected
    source-aligned fixed-one Step/Terminal receipts is sound for the exact
    typed checker, and every admissible accepted execution constructs a
    satisfying assignment. These theorems also bind the Step result columns
    to the accepted output.
  - Closed at the model-level direct-call boundary:
    `DirectCalls.certifiedSubset` constructs exact physical recipes for
    `iterationZero`, `stateEqual`, `freshPublic`, `encodeInstance`, and
    `encodedEqual`. Their footprints are computed from the zero, equality, and
    affine row programs; every row has declared support and ownership; and
    honest active or inactive executions construct all required temporaries.
    `remainingCalls_exact` records the still-open `step`, `hashPrior`,
    `hashNext`, `nifsVerify`, `runningCheck`, and `freshCheck` recipes.
  - Closed at the native-to-lowering semantic boundary:
    `FixedOneLoweringAdapter.parameters` uses the universal native adapter's
    exact setup and paper machine. Its six `CallAlignment` theorems identify
    the open calls with totalized application, the prior hash at `i`, the next
    hash at `i + 1`, ordered NIFS verification, and the supplied exact running
    and fresh terminal checks. `stepAccepts_iff_directHolds` and
    `terminalAccepts_iff_transition` prove that the instantiated intrinsic
    programs accept exactly the frozen step and supplied paper terminal
    transitions. This is semantic alignment only: widths and footprints are
    shape placeholders and no complete codec family, physical recipe, row,
    generated artifact, or compiled-Rust theorem is supplied.
  - Closed for the first concrete production-codec slice:
    `ProductionIterationZeroCallRecipe` first separates the canonical
    bounded-natural zero test from the unrelated nonlinear fresh-public map.
    Given the permitted field/inversion laws and exact footprint alignment,
    it exports a complete typed recipe with exactly three rows, two
    one-coordinate auxiliary temporary bundles, and a mandatory receipt.
    `ProductionDigestCodecs.digestCodec_encode_exact` fixes the four
    Goldilocks coordinates in Rust lane order; the digest, rejecting
    optional-digest, and compact adapter codecs have exact round trips; and
    `encodeInstance_coordinates_exact` proves the direct affine encoder is
    five coordinate copies plus verifier-fixed one.
    `ProductionHashCallBoundary.paperHash_eq_none_iff` separately proves that
    the frozen totalized hash rejects exactly on an absent current state or
    failed duplicated-carrier alignment.
    `paperHash_encoding_eq_absent_iff` and
    `alignedCurrent_encoding_exact` fix the corresponding all-zero and
    presence-one coordinate vectors. The canonical absent-current witness in
    `no_nonoptionalCoreRefines` proves that the nonoptional four-lane sponge
    core cannot replace this wrapper on the complete typed domain. State and
    running codecs, alignment rows, typed sponge rows, and the two hash
    `CallRecipe` values remain open.
    `Goldilocks.NumericRowBridge` separately removes the representation gap
    between the existing numeric sparse-row semantics and the selected typed
    row language. It reduces every coefficient and assignment coordinate into
    the paper Goldilocks carrier, maps source columns only through an explicit
    `Nat -> ColumnId`, proves row-wise and whole-list satisfaction equivalence,
    and preserves source occurrences under caller-owned row ordinals with
    duplicate-free identities.
    `ProductionPoseidon2PermutationRecipe` now applies that bridge to one
    exact width-eight production permutation occurrence. It maps artifact
    columns `0..608` to verifier-one, eight visible inputs, and 600
    receipt-owned auxiliary temporaries; emits the 600 translated SSA rows
    followed by eight activation-gated visible-output copies; and proves exact
    608-row/600-temporary cost, row ownership and identity uniqueness, active
    soundness, temporary-only honest active completion, and inactive
    completion. The honest completion is reconstructed from the executable
    SSA interpreter rather than a generated witness. This is an
    artifact-checked permutation occurrence, not a complete sponge or either
    hash `CallRecipe`: absorption, padding, optional-wrapper presence and
    alignment rows, call-site placement, native Poseidon2 parity, and collision
    resistance remain open.
    `ProductionEncodeInstanceRecipe` realizes those coordinates as exactly six
    caller-owned rows, proves their support and uniqueness, allocates no
    temporary, and supplies active soundness plus active/inactive completeness.
    `ProductionEncodeInstanceCallRecipe` packages the same compiler as a
    complete typed `CallRecipe`: its exact six-row footprint and mandatory
    output/temporary/row receipt follow from the selected program. The
    selection boundary accepts a supplied full lowering profile but identifies
    only the optional-digest codec, compact-encoded codec, and
    `encodeInstance` footprint; it makes no claim about unrelated fields.
    `ProductionEncodedEqualCallRecipe` independently selects equality over
    that same six-coordinate compact codec. Under the permitted
    field/inversion laws and exact footprint alignment, it exports a complete
    typed recipe with exactly eighteen rows, auxiliary temporary bundles of
    widths six, six, and five, and a mandatory receipt. State/fresh/witness
    codecs, complete width agreement, nonlinear `freshPublic`, `stateEqual`,
    and the six calls recorded by `RemainingRecipes` remain separate.
  - Closed at the exact artifact-to-digest boundary:
    `FPrimeFullHistoryProductionDigestCodec.rows_decode_exact_xOut` proves
    that the recursive-output owner constructs one typed production digest
    whose selected codec coordinates are the four physical `xOutColumns` in
    lane order and whose decoder round trip succeeds.
    `output_and_terminal_rows_decode_same_digest` additionally composes the
    exact terminal delayed-link rows and proves their `terminalFreshDigest`
    is that same typed value.
    `decodedDigest_eq_logicalLinkDigest` and
    `terminalLogicalPublic_eq_encodePublicInput` additionally identify the
    selected digest with the independently reconstructed terminal logical-link
    digest and prove that the captured 257-coordinate public input is exactly
    the paper-owned canonical public encoder of that same value.
    `FPrimeFullHistoryProductionDigest.fullRows_finalState_latest_digest_and_logical_public`
    lifts both results through the exact full-row owner partition and
    identifies the final Construction-2 state's singleton fresh-public payload
    and terminal logical public input with the two canonical encodings of one
    typed digest.
    `fullRows_construct_currentPlainOwner` additionally constructs the
    isolated current 270-row assignment by copying the captured 257
    authoritative coordinates and setting the thirteen new padding
    coordinates to zero. It proves current-row satisfaction and identifies
    the current typed claim with that same digest.
    `FPrimeFullHistoryCurrentTerminalLinkPlacementSound` separately consumes a
    bounded certificate exported from one live two-step current synthesis. It
    proves the exact 527-column relabel, row interval
    `[9673389, 9673659)`, equality of all 270 mapped isolated rows with the
    generated range, exact producer-bit alignment, and equivalence of current
    row satisfaction with the frozen logical paper equality.
    `fullRows_and_currentTerminalPlacement_construct_plainOwner` composes that
    range with the captured recursive-output/final-state owners while keeping
    the two row premises separate. This closes one
    producer/consumer/paper-public codec boundary, its honest local
    completion, and one bounded current placement; it does not supply a
    generated aggregate for the whole current program, a universal
    profile/batch placement theorem, a codec for the complete state, the
    compact optional/linked result at this call site, whole-artifact equality
    with the selected receipt program, compiled-Rust semantics, or a
    Poseidon2 collision theorem.
  - Closed for the paper-singleton fresh-public semantic reduction:
    when the adapter's fresh-link callback is the audited plain source checker
    and its ordered fresh batch is exactly `[raw]`,
    `ProductionFreshPublicSingletonBridge` proves compact
    `freshPublic = encodeInstance` iff the 270-coordinate source check, iff the
    selected six-phase program (cost 273), iff the typed and logical paper
    public-input equality. `CanonicalPlainCarrierSerialization` proves the
    complete typed 270-coordinate flattening is injective, so raw acceptance
    loses no carrier information. `FPrimeProductionFreshPublicSingletonRows`
    then proves the selected singleton rows and the exact isolated current
    artifact are equivalent in both directions to the source program and the
    compact adapter equality. The exact cost equation is `273 = 270 + 3`:
    expected length, `m_in`, and vector length stay explicit host/source shape
    checks because the physical relation consumes a typed `Fin 1` claim.
    Multi-fresh batching, those three host checks, producer placement, and
    compiled-Rust semantics remain implementation-refinement obligations.
  - Missing: completion of those six recipes, a concrete production codec
    profile beyond the digest slice, concrete production selection of the
    other direct calls, Rust-emitted typed-program equality outside the
    terminal-link and public-link/XOut slices, and the remaining Rust/R1CS state,
    optional/compact-output, and receipt decoder refinement into that
    canonical fixed-one physical evaluator, followed by equality of every
    generated row with the selected IR compiler output. The existing
    selective-R1CS physical-stage and rewrite ledgers partition the relation
    emitted by the current production compiler, but that relation is not
    definitionally the selected canonical receipt program and may not be used
    as its row certificate. The production lifecycle still has an
    empty-running bootstrap arm; the implemented logical-257-to-physical-270
    carrier cutover is only locally artifact-checked; the fixed terminal
    raw-old-block `y_zcol` row authority is closed, but the recursive-circuit
    and complete active-evaluator refinements remain open; and no single typed
    Rust invocation yet matches the complete Lean canonical `Context`.
    In particular, the current lifecycle `Step.Semantics` exposes an arbitrary
    binary `freshLink` callback but not the paper's separate unary
    `freshPublic` and `encodeInstance` maps. The kernel-checked
    `PaperFreshLinkBoundary.currentInterface_admits_nonFactorizingFreshLink`
    countermodel proves that this factorization cannot be recovered from the
    abstract callback alone. At the typed logical-public-input layer,
    `CanonicalPublicInputLink.equalityFactorization` now proves the positive
    result for `[1 | enc_inst(digest)]`.
    `CanonicalPlainCarrierLink.equalityFactorization` additionally checks the
    complete plain typed carrier—`m_in = 270`, affine one, 256 untrusted field
    coordinates, and thirteen zero-padding coordinates—and
    `check_reduces_to_logicalPaperLink` proves that acceptance is exactly the
    zero completion of the logical paper link. `CanonicalPlainCarrierSource`
    models the untrusted variable-length source list, enforces length 270,
    fixes the exact affine/body/padding split and flattening order, and proves
    pointwise and batch reduction to the typed carrier. Both native lifecycle
    acceptance and the paper decider now invoke the same pure Rust predicate.
    That predicate interprets one verifier-owned six-instruction value in
    shape/affine/body/padding order. The Rust drift gate emits the same value;
    `CanonicalPublicInputLinkProgramRefinement.generated_plain_eq_canonical`
    proves it equals the typed Lean program,
    `generated_plain_cost` computes its 273 scalar obligations, and
    `generated_run_reduces_to_logicalPaperLink` proves the Lean interpreter
    reduces it to the paper equality for every raw claim. Runtime regressions
    mutate all 256 body coordinates, all thirteen padding coordinates, and
    every shape field. Both production call sites now retain the computed
    `EncInst` as the helper argument rather than erasing it into a free
    256-bit array. The production XOut preimage builder now likewise
    interprets one typed source schedule. Its Rust drift gate exports all four
    stateless/stateful × plain/Nebula variants. Lean proves exact equality to
    the independent schedule, exact first domain `0x4e460002`, present-only
    terminal Nebula marker/lane order, exact expansion to
    `encodeStateXOutPreimage` for every typed preimage, and program-derived
    field costs `23`, `28`, `27`, and `32`.
    `StateXOutProgramRefinement.generated_publicLink_accepts_computedXOut`
    composes frozen `XOut.compute` with the generated plain public-link
    program, fixing the outgoing affine-one and 256 little-endian bit
    coordinates. This is generated source-program refinement; Poseidon2
    remains opaque. The selected plain/stateless source program now also
    drives a nonoptional physical sponge receipt: its 23 fields induce six
    rate-four absorb rounds plus one padding round, hence exactly
    `23 + 2 + 600 * 7 = 4225` rows and 4,225 fresh columns. Lean checks the
    actual base-output, recursive-prior, and recursive-output owner slices
    against their reconstructed rows and proves contiguous duplicate-free
    row/column intervals and conservation. The exact captured intervals are
    `[6533,10758)`/`[6344,10569)`, `[218,4443)`/`[868073,872298)`, and
    `[11,4236)`/`[1127811,1132036)`. Erasing physical column identities also
    proves the three cores execute one identical pure sponge schedule. This
    artifact-checked slice is deliberately nonoptional: it does not yet bind
    the option-presence coordinate or the iteration, initial-state, running,
    and program-counter alignment required by totalized `paperHash`.
    Consequently it is not a `hashPrior` or `hashNext` `CallRecipe`, and it
    does not prove current whole-program ownership, native Poseidon2 parity,
    or collision resistance. At the typed receipt boundary,
    `FixedOneCanonicalAdapter.transition_iff_holds` now proves that the exact
    native state, proof, ordered prior/latest batches, NIFS context, and
    prior/next XOut calls instantiate one frozen fixed-one Construction-2
    transition. `nativeAccepted_with_boundaries_and_outgoing_iff_canonicalAccepts`
    composes native producer acceptance with the explicit lifecycle entry,
    incoming-link, stateful, Nebula, and delayed outgoing-link owners;
    `checkedRecorded_with_boundaries_and_outgoing_iff_canonicalAccepts`
    replaces modeled acceptance with the recorded result once the receipt
    checker has established exact control-flow/call conservation and replay.
    Hash and NIFS results remain typed contract values, not authority.
    Formal compiled-Rust semantics and proof that every raw production call
    constructs a checked receipt remain open.
    At the bounded checked-in full-history boundary,
    `FPrimeFullHistoryCircuit.exactSteps_of_fullRows_or_bad` now exposes the
    exact base and recursive `Step.Holds` witnesses instead of only packaging
    them into existential reachability.
    `FPrimeFullHistoryCanonicalSteps.fullRows_imply_frozenSteps_or_bad`
    composes those witnesses with the universal adapter, so satisfaction of
    all 4,193,134 captured rows yields acceptance of both concrete frozen
    fixed-one transitions, or the existing `BadEvent` (the reachable branch
    in this theorem is the named recursive PiRLC projection-root event).
    This is artifact-checked R1CS-to-frozen-step soundness for that snapshot.
    It neither identifies the captured rows with the selected canonical
    receipt program nor proves current Rust semantics, honest assignment
    construction, terminal-checker refinement, or a probability bound.
    Separately, the repaired isolated
    terminal artifact owns all 270 emitted rows, not only the 257 logical
    rows. `FPrimeTerminalLinkCanonicalRefinement.satisfies_iff_logicalPaperLink`
    proves exact row-to-paper equivalence under the explicit producer-bit
    alignment proposition. `FPrimeEncodingCanonicalBits.publicBit_eq_encodedBit`
    reconstructs each producer lane from the exact 532-row encoding owner and
    proves every physical bit equals the independent typed encoder.
    `satisfies_iff_logicalPaperLink_of_encodingRows` composes both owners, so
    only the exact `ProducerColumnsAligned` placement map remains at the
    artifact-independent boundary. A separate live two-step drift gate now
    exports the current `terminal.latest_link` range alone. Lean proves its
    exact producer map, fresh/padding column map, and row-list equality, then
    `generatedRows_iff_logicalPaperLink` discharges
    `ProducerColumnsAligned` for that bounded current range.
    `generatedRows_iff_sourceProgram` proves the same exact placed rows
    equivalent to the singleton production source program, preserving the
    exact `273 = 270 + 3` boundary because the three source shape obligations
    remain outside the typed physical block.
    `generatedRows_iff_freshPublic_eq_encodeInstance` then reaches the compact
    fixed-one prior-public equality under the audited source-link semantics.
    `generatedRows_iff_loweringPriorLinkAccepted` reaches the exact
    `Terminal.priorLinkAccepted` Boolean used by the typed lowering program,
    with the terminal fresh input and prior hash aligned explicitly.
    The source-program, compact, and lowering statements derive their digest
    from the recursive output owner rather than trusting a carried digest.
    This is the Construction-2 prior-public link, not the distinct terminal
    `freshCheck` call. The isolated Rust
    artifact gates check current row and witness digests; the terminal gate
    also checks the first failing padding row. The checked-in full-history
    snapshot itself does not discharge the current placement obligation:
    `FPrimeFullHistoryTerminalLogicalLinkSound.logicalCheck_of_rows` proves the
    frozen logical `[1 | enc_inst]` equality only for its captured 257-row
    prefix, while current plain production emits 270 rows.
    `FPrimeFullHistoryTerminalLinkDrift.generatedSnapshot_ne_currentPlainOwner`
    and `generatedSnapshot_missingPlainPaddingRows` prove the exact mismatch
    and thirteen-row deficit without modifying the generated snapshot. Those
    missing coordinates are verifier-fixed plain-carrier zero padding, not an
    application suffix.
    `FPrimeFullHistoryCurrentTerminalLinkCompletion.completedAssignment`
    closes the honest local-completion direction without pretending the stale
    artifact is current: it copies the captured affine coordinate and all 256
    producer/consumer values into the isolated 270-row owner and assigns zero
    to its thirteen padding columns. The row theorem constructs satisfaction
    of that current owner from the captured 257-row link, while the composed
    output theorem identifies its typed claim, selected digest codec, final
    Construction-2 payload, and captured paper public input with one digest.
    `fPrimeCircuit_complete_with_currentPlainDigest` lifts the same construction
    through an independent successful compiler witness.
    `FPrimeFullHistoryCurrentTerminalLinkPlacementSound.mapped_rows_eq_generated`
    separately supplies the exact current column map for the generated
    270-row interval, and
    `output_and_generated_rows_construct_currentPlainOwner` identifies its
    physical current carrier with the same selected digest and paper link.
    `fullRows_and_currentTerminalPlacement_construct_plainOwner` composes the
    current range alongside the captured aggregate without claiming that the
    stale aggregate contains those columns. No theorem here supplies a
    generated whole-current-program aggregate or universal placement across
    profiles and batches.
    `FPrimeTerminalLinkBatch` now gives the
    artifact-independent arbitrary-batch lift: its typed receipts are
    bijective with all `270 * batchSize` row positions, own exactly the
    corresponding public-column interval, compute row/public/committed/auxiliary
    costs definitionally, and prove exact R1CS-to-paper equivalence for every
    claim. Its one-claim specialization equals the isolated artifact.
    Production now interprets one verifier-owned three-instruction
    affine/body/padding schedule per claim in claim-major order. The Rust
    exporter emits that exact schedule; Lean proves equality with the selected
    typed program, cost `270` per claim, expansion to the complete local-owner
    order, and exact arbitrary-batch ownership/cost `270 * batchSize`.
    `TerminalLink.Program.compile` rejects every schedule whose expansion is
    not the complete selected receipt order and otherwise returns exactly the
    receipt-owned arbitrary-batch rows.
    `generated_plain_compile` proves that the Rust-emitted schedule compiles
    for every batch size, and
    `TerminalLink.LoweringRefinement.generatedPlain_accepts_iff_priorLinkAccepted`
    proves singleton acceptance is exactly the typed Terminal prior-link
    Boolean under explicit digest, fresh ordering, source-link, and producer
    alignment.
    `TerminalLink.PlacementRefinement.generatedPlain_compile_eq_currentPlacement`
    then relabels the checked singleton compiler output to the exact generated
    current full-history range. Its two acceptance theorems prove that the
    pulled source program, generated current rows, and output-derived typed
    `priorLinkAccepted` Boolean are extensionally identical at this bounded
    placement. This is source-program and artifact-checked lowering
    conformance, not compiled-Rust semantics. Separately, a
    two-claim drift gate captures the literal sparse rows through the private
    production emitter's isolation wrapper. Lean proves its exact `540` rows
    and `797` columns equal the selected `rows 2` compiler output;
    `generated_rows_eq_compiler_output` states the stronger literal equation
    between that capture and the checked compiler applied to the Rust-emitted
    program, while bijective receipt ownership covers every captured row.
    That is bounded
    artifact evidence, not a universal theorem for the Rust batch loop. The
    remaining obligation is physical equality of the concrete
    recipe-instantiated selected lowering calls `freshPublic`,
    `encodeInstance`, and `encodedEqual` with emitted rows (or a proved
    refinement from the distinct current compiler relation), a generated
    whole-current-program or universal full-history placement theorem, formal
    compiled-Rust interpreter semantics,
    raw-production-input-to-checked-receipt refinement, and terminal-checker
    integration. The captured full-history step theorem does not discharge
    any of these.

axiom_report:
  The independent NIFS and partial replay theorems are in
  `tests/Axioms/Paper.lean`; finite SumCheck, generic first-accepted sampling,
  and PaperJoint have dedicated fail-closed guards in
  `tests/Axioms/{SumCheckFinite,FirstAccepted,PiCCSPaperJoint}.lean`.
  Fixed-active F' evaluator, semantic-boundary, exact-plan, conditional
  necessity, full-child prior-link, canonical-PiDEC-child, conditional
  ordered-child-commitment authority, opening-derived authority,
  honest-baseline source, strict running
  authority, and semantic-premise theorems are guarded in
  `tests/Axioms/Protocol.lean`. Reduced-carrier codec, exact ordered-message,
  exact ordered-message prior-link, and Poseidon2-cost theorems, plus the
  canonical-parent opening exactness/minimality theorems, have dedicated guards in
  `tests/Axioms/FPrimeAccumulatorCarrierCodec.lean` and
  `tests/Axioms/FPrimeAccumulatorOrderedCommitmentMessage.lean`,
  `tests/Axioms/FPrimeAccumulatorOrderedCommitmentPriorLink.lean`,
  `tests/Axioms/FPrimeAccumulatorOrderedCommitmentSourceLayout.lean`,
  `tests/Axioms/FPrimeAccumulatorOrderedCommitmentTypedDecoder.lean`,
  `tests/Axioms/FPrimeAccumulatorPoseidon2Cost.lean`,
  `tests/Axioms/NifsConcretePhi81CanonicalOpening.lean`, and
  `tests/Axioms/PiDecCanonicalParentOpeningVerifier.lean`. The complete active
  PiCCS output serializer and fixed-active profile bridge are guarded by
  `tests/Axioms/Implementation/PiCcsOutputActiveSemantics.lean`. All are imported by the
  aggregate axioms gate. The bounded `validate.sh static` and `axioms` gates,
  plus focused module and interface builds, passed for this slice; the full
  `build` and `check` gates were not rerun. The Rust/R1CS bridge report remains
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
