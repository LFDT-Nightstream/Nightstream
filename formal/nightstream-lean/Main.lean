import Nightstream

set_option maxRecDepth 16384

/-!
`lake exe check` — the executable assurance gate.

Every line of output is a computed result: the envelope probes actually run
`Envelope.check`, and the drift gate actually reads the mapped Rust sources
and verifies the symbol anchors the model claims parity with. Any failure
exits nonzero. Run from `formal/nightstream-lean` (spec §13).
-/

open Nightstream.HyperNova.Construction2
open Nightstream.Implementation.FPrime.Envelope

abbrev ProbeState := State Nat Unit Unit

def probeInitial : ProbeState where
  chunkCount := 0
  stepCount := 0
  z0 := 7
  zi := 7
  initialSemanticState := 11
  semanticState := 11
  pc := 1
  accumulatorDigest := 0
  publicTrace := 0
  proof := .initial

def probeNext : ProbeState where
  chunkCount := 1
  stepCount := 2
  z0 := 7
  zi := 13
  initialSemanticState := 11
  semanticState := 17
  pc := 1
  accumulatorDigest := 19
  publicTrace := 13
  proof := .active () [(), ()]

/-- The empty-step forgery Rust rejects with `Error::EmptyStep`. -/
def probeEmptyStep : ProbeState :=
  { probeNext with stepCount := 0, proof := .active () [] }

/-- Envelope probes: expected boolean per case, all computed here. -/
def envelopeProbes : List (String × Bool × Bool) :=
  [ ("envelope_honest_transition", check 2 probeInitial probeNext, true)
  , ("envelope_rejects_empty_step", check 0 probeInitial probeEmptyStep, false)
  , ("envelope_rejects_count_forgery", check 1 probeInitial probeNext, false)
  ]

open Nightstream.Implementation.R1CS Nightstream.Implementation.R1CS.CanonicalU64 in
/--
R1CS artifact probes: recompute both exported witness verdicts over the
exact rows `canonicalU64_sound` is stated on. The forged vector must fail —
it is the `x + p` malleability the canonicity gate exists to reject. The
row-level drift gate lives in Rust
(`cargo test -p neo-fold-clean --test gadgets_lean_artifact`).
-/
def artifactProbes : List (String × Bool × Bool) :=
  [ ("r1cs_artifact_honest_witness", decide (Satisfies rows (assignmentOf honestWitness)), true)
  , ("r1cs_artifact_forged_noncanonical", decide (Satisfies rows (assignmentOf forgedWitness)), false)
  , ("r1cs_u64_increment_honest", decide (Satisfies U64Increment.rows
      (assignmentOf U64Increment.honestWitness)), true)
  , ("r1cs_u64_increment_rejects_wrap", decide (Satisfies U64Increment.rows
      (assignmentOf U64Increment.overflowWitness)), false)
  , ("r1cs_u64_add_honest", decide (Satisfies U64Add.rows
      (assignmentOf U64Add.honestWitness)), true)
  , ("r1cs_u64_add_rejects_wrap", decide (Satisfies U64Add.rows
      (assignmentOf U64Add.overflowWitness)), false)
  , ("r1cs_fprime_counter_honest", decide (Satisfies FPrimeCounter.rows
      (assignmentOf FPrimeCounter.honestWitness)), true)
  , ("r1cs_fprime_counter_rejects_source_disconnect", decide (Satisfies FPrimeCounter.rows
      (assignmentOf FPrimeCounter.wrongSourceWitness)), false)
  , ("r1cs_fprime_counter_rejects_wrong_step", decide (Satisfies FPrimeCounter.rows
      (assignmentOf FPrimeCounter.wrongStepWitness)), false)
  , ("r1cs_fprime_counter_rejects_batch_size_forgery", decide (Satisfies FPrimeCounter.rows
      (assignmentOf FPrimeCounter.wrongRowsWitness)), false)
  , ("r1cs_fprime_encoding_exact_row_count",
      Nightstream.Implementation.R1CS.FPrimeEncoding.rows.length ==
        Nightstream.Implementation.R1CS.FPrimeEncoding.rowCount, true)
  , ("fprime_encoding_accepts_256_bits",
      Nightstream.Implementation.Encoding.FPrime.acceptsEncInstLength
        (List.replicate 256 false), true)
  , ("fprime_encoding_rejects_255_bits",
      Nightstream.Implementation.Encoding.FPrime.acceptsEncInstLength
        (List.replicate 255 false), false)
  , ("r1cs_fprime_terminal_link_exact_row_count",
      Nightstream.Implementation.R1CS.FPrimeTerminalLink.rows.length ==
        Nightstream.Implementation.R1CS.FPrimeTerminalLink.rowCount, true)
  , ("r1cs_fprime_state_link_exact_row_count",
      Nightstream.Implementation.R1CS.FPrimeStateLink.rows.length ==
        Nightstream.Implementation.R1CS.FPrimeStateLink.rowCount, true)
  , ("r1cs_fprime_base_state_exact_row_count",
      Nightstream.Implementation.R1CS.FPrimeBaseState.rows.length ==
        Nightstream.Implementation.R1CS.FPrimeBaseState.rowCount, true)
  , ("r1cs_fprime_base_program_exact_instruction_count",
      Nightstream.Implementation.R1CS.FPrimeBaseProgram.instructions.length ==
        Nightstream.Implementation.R1CS.FPrimeBaseProgram.rowCount, true)
  , ("r1cs_fprime_chunk_digest_binding_row_count",
      Nightstream.Implementation.R1CS.FPrimeChunkDigest.bindingRows.length == 4, true)
  , ("r1cs_fprime_ce_continuity_exact_row_count",
      Nightstream.Implementation.R1CS.FPrimeCeContinuity.continuityRows.length ==
        Nightstream.Implementation.R1CS.FPrimeCeContinuity.continuityRowCount, true)
  , ("r1cs_fprime_recursive_manifest_top_level_coverage",
      Nightstream.Implementation.R1CS.FPrimeRecursiveManifest.covers 0
        Nightstream.Implementation.R1CS.FPrimeRecursiveManifest.totalRows
        Nightstream.Implementation.R1CS.FPrimeRecursiveManifest.topLevelFamilies, true)
  , ("r1cs_fprime_recursive_manifest_nifs_coverage",
      Nightstream.Implementation.R1CS.FPrimeRecursiveManifest.covers 20038 2592246
        Nightstream.Implementation.R1CS.FPrimeRecursiveManifest.nifsFamilies, true)
  , ("r1cs_fprime_full_history_manifest_rows",
      Nightstream.Implementation.R1CS.FPrimeFullHistoryManifest.totalRows ==
        4076614, true)
  , ("r1cs_fprime_full_history_manifest_columns",
      Nightstream.Implementation.R1CS.FPrimeFullHistoryManifest.totalColumns ==
        3298653, true)
  , ("r1cs_fprime_full_history_top_level_coverage",
      Nightstream.Implementation.R1CS.FPrimeRecursiveManifest.covers 0
        Nightstream.Implementation.R1CS.FPrimeFullHistoryManifest.totalRows
        Nightstream.Implementation.R1CS.FPrimeFullHistoryManifest.topLevelFamilies,
      true)
  , ("r1cs_pirlc_projection_exact_row_count",
      Nightstream.Implementation.R1CS.PiRLCProjection.rows.length ==
        Nightstream.Implementation.R1CS.PiRLCProjection.rowCount, true)
  , ("r1cs_pirlc_projection_honest_satisfies",
      decide (Satisfies Nightstream.Implementation.R1CS.PiRLCProjection.rows
        (assignmentOf Nightstream.Implementation.R1CS.PiRLCProjection.honestWitness)), true)
  , ("r1cs_pirlc_projection_bad_root_satisfies",
      decide (Satisfies Nightstream.Implementation.R1CS.PiRLCProjection.rows
        (assignmentOf Nightstream.Implementation.R1CS.PiRLCProjection.badRootWitness)), true)
  , ("r1cs_pirlc_production_identity_census",
      Nightstream.Implementation.R1CS.FPrimeRecursiveManifest.projectionIdentityCount == 31, true)
  , ("r1cs_pirlc_production_pair_census",
      Nightstream.Implementation.R1CS.FPrimeRecursiveManifest.projectionPairCounts.all
        (fun count => count == 15), true)
  , ("r1cs_poseidon2_permutation_exact_row_count",
      Nightstream.Implementation.R1CS.Poseidon2Permutation.rows.length ==
        Nightstream.Implementation.R1CS.Poseidon2Permutation.rowCount, true)
  ]

open Nightstream.SuperNeo Nightstream.SuperNeo.Concrete in
/-- Executable cross-checks for the concrete Appendix-B.2 profile. -/
def parameterProbes : List (String × Bool × Bool) :=
  [ ("params_goldilocks_q", productionGlobalParams.q == 18446744069414584321, true)
  , ("params_fresh_bound", productionGlobalParams.b == 2, true)
  , ("params_decomposition_arity", productionGlobalParams.k == 14, true)
  , ("params_max_fresh", productionGlobalParams.maxFresh == 61, true)
  , ("params_big_b", productionGlobalParams.bigB == 16384, true)
  , ("params_expansion_t", productionGlobalParams.expansionT == 216, true)
  ]

namespace M2Probe

open Nightstream.SuperNeo.SumCheck

def ops : Ops Nat Nat where
  zero := 0
  one := 1
  add := Nat.add

def expected : Nat → Nat
  | 0 => 2
  | 1 => 3
  | _ => 7

def forged : Nat → Nat
  | 0 => 4
  | 1 => 4
  | _ => 7

def honestTranscript : Instance Nat Nat where
  claimedInitial := 5
  trueInitial := 5
  terminal := 7
  rounds := [{ claimed := expected, expected := expected, challenge := 2, degree := 2 }]
  maxDegree := 2
  challengeSetSize := 97

/-- False claim which passes only because the polynomials collide at challenge two. -/
def forgedTranscript : Instance Nat Nat where
  claimedInitial := 8
  trueInitial := 5
  terminal := 7
  rounds := [{ claimed := forged, expected := expected, challenge := 2, degree := 2 }]
  maxDegree := 2
  challengeSetSize := 97

def malformedTranscript : Instance Nat Nat :=
  { forgedTranscript with claimedInitial := 9 }

end M2Probe

/-- Executable M2 probes keep acceptance visibly separate from claim truth. -/
def foldingProbes : List (String × Bool × Bool) :=
  [ ("sumcheck_honest_accepts",
      Nightstream.SuperNeo.SumCheck.check M2Probe.ops M2Probe.honestTranscript, true)
  , ("sumcheck_forged_collision_acceptance_observed",
      Nightstream.SuperNeo.SumCheck.check M2Probe.ops M2Probe.forgedTranscript, true)
  , ("sumcheck_forged_claim_truth_is_false",
      decide (M2Probe.forgedTranscript.claimedInitial = M2Probe.forgedTranscript.trueInitial),
      false)
  , ("sumcheck_malformed_chain_is_rejected",
      Nightstream.SuperNeo.SumCheck.check M2Probe.ops M2Probe.malformedTranscript, false)
  ]

namespace M3Probe

open Nightstream.Protocol.FPrime

def hashSemantics : XOut.Semantics Unit Unit Unit Nat Unit Unit where
  hash := fun _ => 17
  nebulaDigest := id

def context : XOut.Context Unit Unit Unit Nat where
  params := ()
  structureDigest := ()
  piCcsHeader := ()
  publicInputLength := none
  initialSemanticState := 0

def stepSemantics : Step.Semantics Nat Nat Nat Nat Unit Unit where
  emptyRunning := 0
  initialNebula := none
  runningDigest := id
  chunkDigest := fun start fresh => 100 + start + fresh.length
  freshLink := fun digest fresh => digest == fresh
  nifsVerify := fun transcript running latest proof =>
    if proof = running + latest.length + transcript.chunkCount then
      some (running + latest.length)
    else
      none
  applicationStep := fun _ _ _ => true
  nebulaVerify := fun prior opening next => decide (opening = none ∧ next = prior)

abbrev ProbeState := State Nat Nat Nat Unit
abbrev ProbeInput := Step.Input Nat Unit Unit
abbrev ProbeProof := Step.Proof Nat Nat Unit

def initial : ProbeState where
  chunkCount := 0
  stepCount := 0
  z0 := 17
  zi := 17
  initialSemanticState := 0
  semanticState := 0
  pc := 1
  accumulatorDigest := 0
  publicTrace := 17
  proof := .initial

def baseInput : ProbeInput where
  nextLatest := [17]
  nebulaOpen := none
  nebulaNext := none

def baseProof : ProbeProof where
  fold := .noFold
  nebulaOpen := none
  semanticStateDigest := 0
  xOut := 17

def afterBase : ProbeState :=
  Step.advancedState stepSemantics initial 0 baseInput baseProof

def recursiveInput : ProbeInput where
  nextLatest := [17]
  nebulaOpen := none
  nebulaNext := none

def recursiveProof : ProbeProof where
  fold := .recursive 2
  nebulaOpen := none
  semanticStateDigest := 1
  xOut := 17

def afterRecursive : ProbeState :=
  Step.advancedState stepSemantics afterBase 1 recursiveInput recursiveProof

end M3Probe

/-- Executable M3 probes cover honest branches and single-coordinate forgeries. -/
def fPrimeSemanticProbes : List (String × Bool × Bool) :=
  [ ("fprime_base_accepts",
      Nightstream.Protocol.FPrime.Step.check M3Probe.hashSemantics
        M3Probe.stepSemantics .stateless M3Probe.context M3Probe.initial
        M3Probe.afterBase M3Probe.baseInput M3Probe.baseProof, true)
  , ("fprime_base_rejects_xout_forgery",
      Nightstream.Protocol.FPrime.Step.check M3Probe.hashSemantics
        M3Probe.stepSemantics .stateless M3Probe.context M3Probe.initial
        M3Probe.afterBase M3Probe.baseInput
        { M3Probe.baseProof with xOut := 18 }, false)
  , ("fprime_recursive_accepts",
      Nightstream.Protocol.FPrime.Step.check M3Probe.hashSemantics
        M3Probe.stepSemantics .stateless M3Probe.context M3Probe.afterBase
        M3Probe.afterRecursive M3Probe.recursiveInput M3Probe.recursiveProof, true)
  , ("fprime_recursive_rejects_nifs_forgery",
      Nightstream.Protocol.FPrime.Step.check M3Probe.hashSemantics
        M3Probe.stepSemantics .stateless M3Probe.context M3Probe.afterBase
        M3Probe.afterRecursive M3Probe.recursiveInput
        { M3Probe.recursiveProof with fold := .recursive 3 }, false)
  , ("fprime_xout_preimage_observes_counter",
      decide (Nightstream.Protocol.FPrime.XOut.preimage M3Probe.hashSemantics
        .stateless M3Probe.context M3Probe.afterBase ≠
        Nightstream.Protocol.FPrime.XOut.preimage M3Probe.hashSemantics
          .stateless M3Probe.context
          { M3Probe.afterBase with chunkCount := 2 }), true)
  ]

namespace M5Probe

open Nightstream.Protocol

abbrev TerminalClaim := TerminalCE.Claim Nat Nat Nat Nat Nat Bool
abbrev TerminalInstance := TerminalCE.Instance Nat Nat Nat Nat Nat Nat Nat Bool

def terminalSemantics : TerminalCE.Semantics Nat Nat Nat Nat Nat Nat Nat Bool where
  commit := id
  projectPublicInput := fun width witness =>
    if width = 1 then some witness else none
  normBounded := fun bound witness => decide (witness < bound)
  evaluationPointValid := fun relation point => decide (relation = point)
  evaluations := fun relation witness point =>
    if relation = point then some [witness + point] else none
  constantTerm := id
  sidecarValid := fun _ _ sidecar => sidecar

def terminalClaim : TerminalClaim where
  commitment := 2
  publicWidth := 1
  publicInput := 2
  point := 3
  evaluations := [5]
  constantTerms := [5]
  sidecar := true

def terminal : TerminalInstance where
  context := {
    relation := 3
    normBound := 10
    expectedPublicWidth := some 1
  }
  verifierChildren := [terminalClaim]
  recordedClaims := [terminalClaim]
  witnesses := [2]

def fPrimeAccepted : Except Nightstream.Implementation.Rust.FPrime.Error Unit → Bool
  | .ok _ => true
  | .error _ => false

def fPrimeRejectedEmpty : Except Nightstream.Implementation.Rust.FPrime.Error Unit → Bool
  | .error .emptyStep => true
  | _ => false

def terminalAccepted : Except Nightstream.Implementation.Rust.Terminal.Error Unit → Bool
  | .ok _ => true
  | .error _ => false

def terminalRejectedChildAuthority :
    Except Nightstream.Implementation.Rust.Terminal.Error Unit → Bool
  | .error .childAuthority => true
  | _ => false

end M5Probe

/-- Executable M5 probes exercise both Rust-shaped refinement programs. -/
def rustRefinementProbes : List (String × Bool × Bool) :=
  [ ("rust_fprime_base_accepts",
      M5Probe.fPrimeAccepted (Nightstream.Implementation.Rust.FPrime.verify
        M3Probe.hashSemantics M3Probe.stepSemantics .stateless M3Probe.context
        M3Probe.initial M3Probe.afterBase M3Probe.baseInput M3Probe.baseProof), true)
  , ("rust_fprime_rejects_empty_step",
      M5Probe.fPrimeRejectedEmpty (Nightstream.Implementation.Rust.FPrime.verify
        M3Probe.hashSemantics M3Probe.stepSemantics .stateless M3Probe.context
        M3Probe.initial M3Probe.afterBase
        { M3Probe.baseInput with nextLatest := [] } M3Probe.baseProof), true)
  , ("rust_terminal_ce_accepts",
      M5Probe.terminalAccepted (Nightstream.Implementation.Rust.Terminal.verify
        M5Probe.terminalSemantics M5Probe.terminal), true)
  , ("rust_terminal_ce_rejects_disconnected_child",
      M5Probe.terminalRejectedChildAuthority
        (Nightstream.Implementation.Rust.Terminal.verify M5Probe.terminalSemantics
          { M5Probe.terminal with recordedClaims := [] }), true)
  ]

namespace ProjectionProbe

open Nightstream.SuperNeo.ProjectionCheck

def ops : Ops Nat where
  zero := 0
  add := fun left right => (left + right) % 97
  mul := fun left right => (left * right) % 97

def fixedBetaForgery : Identity Nat where
  lhs := [90, 1]
  rhs := [0, 0]
  beta := 7
  maxDegree := 1

end ProjectionProbe

/-- The one-point check accepts a nonzero polynomial exactly on the named bad
root; it must never be presented as deterministic coefficient equality. -/
def projectionProbes : List (String × Bool × Bool) :=
  [ ("pirlc_projection_fixed_beta_accepts_root_collision",
      decide (Nightstream.SuperNeo.ProjectionCheck.Accepted
        ProjectionProbe.ops ProjectionProbe.fixedBetaForgery), true)
  , ("pirlc_projection_root_collision_is_not_exact",
      decide ProjectionProbe.fixedBetaForgery.Exact, false)
  ]

/--
Symbol anchors in the mapped Rust sources. If an anchor disappears, the
mapped surface has drifted and every parity claim resting on it is stale —
the gate fails instead of printing success. These anchors guard the symbols
the current model claims parity with; the generated-artifact content gate is
the Rust drift test above.
-/
def rustAnchors : List (String × String) :=
  [ ("../../crates/neo-fold-clean/src/paper/f_prime/native.rs",
     "Err(Error::EmptyStep)")
  , ("../../crates/neo-fold-clean/src/paper/construction2/transition.rs",
     "fn state_base_case_check")
  , ("../../crates/neo-fold-clean/src/paper/construction2/transition.rs",
     "fn advance_state")
  , ("../../crates/neo-fold-clean/src/paper/construction2/state.rs",
     "pub struct State")
  , ("../../crates/neo-fold-clean/src/engine/r1cs_circuit/u64.rs",
     "pub fn decompose_var_to_u64_bits")
  , ("../../crates/neo-fold-clean/src/paper/f_prime/r1cs.rs",
     "pub fn enforce_f_prime_counter_input_binding")
  , ("../../crates/neo-fold-clean/src/paper/f_prime/r1cs.rs",
     "pub fn enforce_f_prime_recursive_counter_transition")
  , ("../../crates/neo-fold-clean/src/paper/f_prime/r1cs.rs",
     "pub fn enforce_public_bits_encode_digest")
  , ("../../crates/neo-fold-clean/src/engine/decider.rs",
     "fn enforce_terminal_latest_link")
  , ("../../crates/neo-fold-clean/src/engine/decider.rs",
     "fn enforce_state_link")
  , ("../../crates/neo-fold-clean/src/paper/construction2/transition.rs",
     ".checked_add(1)")
  , ("../../crates/neo-fold-clean/src/paper/construction2/transition.rs",
     ".checked_add(fresh_count)")
  , ("../../crates/neo-fold-clean/src/paper/construction2/mod.rs",
     "CounterOverflow { counter: &'static str }")
  , ("../../crates/neo-fold-clean/tests/system/lifecycle_finalization.rs",
     "fn extend_rejects_chunk_counter_overflow")
  , ("../../crates/neo-fold-clean/tests/system/lifecycle_finalization.rs",
     "fn extend_rejects_step_counter_overflow")
  , ("../../crates/neo-fold-clean/src/paper/params.rs",
     "pub fn max_fresh_count")
  , ("../../crates/neo-fold-clean/tests/system/production_params.rs",
     "fn production_params_match_lean_m1_profile")
  , ("../../crates/neo-fold-clean/src/paper/nifs/verifier.rs",
     "let ccs_out_claims = pi_ccs::verify")
  , ("../../crates/neo-fold-clean/src/paper/nifs/verifier.rs",
     "let combined = pi_rlc::verify")
  , ("../../crates/neo-fold-clean/src/paper/nifs/verifier.rs",
     "let children = pi_dec::verify")
  , ("../../crates/neo-fold-clean/src/paper/reductions/pi_ccs.rs",
     "validate_adv_forwarding")
  , ("../../crates/neo-fold-clean/src/paper/reductions/pi_rlc.rs",
     "enforce_rlc_bound")
  , ("../../crates/neo-fold-clean/src/engine/r1cs_circuit/ring_action.rs",
     "pub fn enforce_ring_action_projection_batch")
  , ("../../crates/neo-fold-clean/src/paper/reductions/pi_dec.rs",
     "validate_child_count")
  , ("../../crates/neo-fold-clean/src/paper/digest.rs",
     "pub fn state_x_out_digest_with_mode")
  , ("../../crates/neo-fold-clean/src/paper/digest.rs",
     "pub fn public_trace_seed_digest")
  , ("../../crates/neo-fold-clean/src/paper/f_prime/native.rs",
     "RunningInstance::default()")
  , ("../../crates/neo-fold-clean/src/paper/f_prime/native.rs",
     "nifs::verify(")
  , ("../../crates/neo-fold-clean/src/paper/f_prime/native.rs",
     "pub fn f_prime_step_transcript")
  , ("../../crates/neo-fold-clean/src/paper/f_prime/native.rs",
     "b\"f_prime/chunk_digest\"")
  , ("../../crates/neo-fold-clean/src/paper/f_prime/native.rs",
     "Error::FoldProofVariantMismatch")
  , ("../../crates/neo-fold-clean/src/paper/f_prime/native.rs",
     "Error::StatelessSemanticInvariantViolated")
  , ("../../crates/neo-fold-clean/src/paper/f_prime/native.rs",
     "Error::NebulaOpenMismatch")
  , ("../../crates/neo-fold-clean/src/paper/f_prime/native.rs",
     "Error::XOutMismatch")
  , ("../../crates/neo-fold-clean/src/paper/construction2/state.rs",
     "pub nebula:")
  , ("../../crates/neo-fold-clean/src/lifecycle/verify.rs",
     "pub fn verify_uncompressed(")
  , ("../../crates/neo-fold-clean/src/lifecycle/verify.rs",
     "fn check_running_witnesses_authority(")
  , ("../../crates/neo-fold-clean/src/lifecycle/verify.rs",
     "pub fn verify_uncompressed_audit(")
  , ("../../crates/neo-fold-clean/src/paper/decider.rs",
     "pub fn validate_witness(")
  , ("../../crates/neo-fold-clean/src/paper/decider.rs",
     "Err(Error::Unsupported)")
  , ("../../crates/neo-fold-clean/tests/system/formal_conformance.rs",
     "fn conformance_manifest_fails_closed_on_rust_or_lean_drift")
  , ("../../crates/neo-fold-clean/tests/system/formal_conformance.rs",
     "fn compact_decider_is_explicitly_fail_closed")
  ]

def containsSubstr (haystack needle : String) : Bool :=
  (haystack.splitOn needle).length > 1

def main : IO UInt32 := do
  let mut ok := true

  for (name, got, expected) in
      envelopeProbes ++ artifactProbes ++ parameterProbes ++ foldingProbes ++
        fPrimeSemanticProbes ++ projectionProbes ++ rustRefinementProbes do
    let pass := got == expected
    IO.println s!"{name}={pass}"
    unless pass do ok := false

  for (path, needle) in rustAnchors do
    let pass ← do
      try
        let content ← IO.FS.readFile ⟨path⟩
        pure (containsSubstr content needle)
      catch _ =>
        pure false
    IO.println s!"rust_anchor {path} :: {needle} => {pass}"
    unless pass do ok := false

  IO.println "rust_conformance=M5-pass (supported uncompressed F-prime lifecycle and direct terminal CE); compact_decider=fail-closed-unsupported; DEC-SOUND=open"
  if ok then
    IO.println "check=pass"
    return 0
  else
    IO.println "check=FAIL"
    return 1
