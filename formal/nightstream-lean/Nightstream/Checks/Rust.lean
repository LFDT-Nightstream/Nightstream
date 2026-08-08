import Nightstream.Checks.Common
import Nightstream.Checks.Protocol
import Nightstream.Implementation.Rust.FPrime
import Nightstream.Implementation.Rust.Terminal
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleExport
import Nightstream.Protocol.Terminal.CE

namespace Nightstream.Checks.Rust

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
def probes : List Nightstream.Checks.Probe :=
  [ ⟨"rust_fprime_base_accepts", fun _ =>
      M5Probe.fPrimeAccepted (Nightstream.Implementation.Rust.FPrime.verify
        Protocol.M3Probe.hashSemantics Protocol.M3Probe.stepSemantics .stateless
        Protocol.M3Probe.context Protocol.M3Probe.initial Protocol.M3Probe.afterBase
        Protocol.M3Probe.baseInput Protocol.M3Probe.baseProof), true⟩
  , ⟨"rust_fprime_rejects_empty_step", fun _ =>
      M5Probe.fPrimeRejectedEmpty (Nightstream.Implementation.Rust.FPrime.verify
        Protocol.M3Probe.hashSemantics Protocol.M3Probe.stepSemantics .stateless
        Protocol.M3Probe.context Protocol.M3Probe.initial Protocol.M3Probe.afterBase
        { Protocol.M3Probe.baseInput with nextLatest := [] } Protocol.M3Probe.baseProof), true⟩
  , ⟨"rust_terminal_ce_accepts", fun _ =>
      M5Probe.terminalAccepted (Nightstream.Implementation.Rust.Terminal.verify
        M5Probe.terminalSemantics M5Probe.terminal), true⟩
  , ⟨"rust_terminal_ce_rejects_disconnected_child", fun _ =>
      M5Probe.terminalRejectedChildAuthority
        (Nightstream.Implementation.Rust.Terminal.verify M5Probe.terminalSemantics
          { M5Probe.terminal with recordedClaims := [] }), true⟩
  ]

/-- Symbol anchors in mapped Rust sources. Their contents remain a conformance
gate rather than proof authority. -/
def anchors : List (String × String) :=
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
     "fn extend_rejects_chunk_counter_past_the_canonical_field_range")
  , ("../../crates/neo-fold-clean/tests/system/lifecycle_finalization.rs",
     "fn extend_rejects_step_counter_past_the_canonical_field_range")
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
     "RunningInstance::canonical_zero(")
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
  , ("../../crates/neo-fold-clean/tests/system/formal_conformance.rs",
     "fn conformance_manifest_fails_closed_on_rust_or_lean_drift")
  , ("../../crates/neo-fold-clean/src/frontends/r1cs_f_prime/terminal_r1cs/lifecycle.rs",
     "pub fn finish_with_spartan(")
  , ("../../crates/neo-fold-clean/src/frontends/r1cs_f_prime/terminal_r1cs/lifecycle.rs",
     "pub fn verify_spartan(")
  , ("../../crates/neo-fold-clean/src/frontends/r1cs_f_prime/terminal_r1cs/lifecycle.rs",
     "RepeatedR1CSSNARK::<TerminalSpartanEngine>::prove_direct")
  , ("../../crates/neo-fold-clean/tests/system/lean_native_ccs_manifest.rs",
     "fn terminal_r1cs_proves_and_verifies_with_spartan_and_whir")
  , ("../../crates/wip-spartan/src/spartan/parallel_repetition.rs",
     "direct-r1cs/parallel-3/v1")
  , ("../../crates/wip-spartan/src/sumcheck.rs",
     "fn verify_parallel_3(")
  ]

def containsSubstr (haystack needle : String) : Bool :=
  (haystack.splitOn needle).length > 1

private def flush : IO Unit :=
  IO.getStdout >>= IO.FS.Stream.flush

def runAnchors : IO Bool := do
  let mut ok := true
  for (path, needle) in anchors do
    let pass ← do
      try
        let content ← IO.FS.readFile ⟨path⟩
        pure (containsSubstr content needle)
      catch _ =>
        pure false
    IO.println s!"rust_anchor {path} :: {needle} => {pass}"
    flush
    unless pass do
      ok := false
  pure ok

def runWasmModuleArtifact : IO Bool := do
  let path := "../../crates/neo-wasm/tests/fixtures/wasm_benchmark_42x6.module.json"
  let expected :=
    Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleExport.render
  let pass ← do
    try
      let content ← IO.FS.readFile ⟨path⟩
      pure (content == expected)
    catch _ =>
      pure false
  IO.println s!"rust_artifact {path} :: lean_wasm_module => {pass}"
  flush
  pure pass

/-- Emit a conformance token only after the complete result is known. -/
def resultLine : Bool → String
  | true =>
      "rust_conformance=M5-reopened (functional probes and artifact checks pass; Rust-originated provenance audit open); direct_terminal_spartan=artifact-checked-bounded-lockstep; generic_compact_decider=not-exposed; DEC-SOUND=open"
  | false =>
      "rust_conformance=M5-fail; no Rust-conformant claim is established; DEC-SOUND=open"

def run : IO Bool := do
  let probesOk ← Nightstream.Checks.runProbes probes
  let anchorsOk ← runAnchors
  let wasmModuleOk ← runWasmModuleArtifact
  let pass := probesOk && anchorsOk && wasmModuleOk
  IO.println (resultLine pass)
  flush
  pure pass

end Nightstream.Checks.Rust
