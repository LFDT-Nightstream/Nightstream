import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.Composition
import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.Generated.Receipts

/-!
Contract: bounded differential conformance for the public native F' step.

Owns:
- one executable certificate over the eleven generated proof-free receipts;
- the generic consequence that every generated member conserves the observed
  call-site trace and agrees with the normalized receipt-free oracle replay.

Does not own:
- Rust generation or drift detection;
- lifecycle boundary receipts or their truth;
- R1CS lowering, terminal closure, or production-wide conformance.

The certificate contains 11 proof-free records.  The largest trace has 21
events, 11 transcript appends containing 35 fields, one eight-field opaque
snapshot, and a 27-field state-output preimage.  It contains no witness matrix
or proof-carrying Lean structure.
-/

namespace Nightstream.Implementation.Rust.CanonicalConformance.NativeStep

/-- The bounded generated profile contains exactly eleven differential cases. -/
theorem generated_case_count :
    Generated.all.length = 11 := by
  rfl

/-- One bounded executable certificate checks all eleven generated records. -/
theorem generated_all_check :
    Generated.all.all check = true := by
  native_decide

/-- Every generated record agrees with both normalized oracle replays. -/
theorem oracleReplayConforms_of_mem_generated
    (receipt : Receipt)
    (member : receipt ∈ Generated.all) :
    OracleReplayConforms receipt := by
  have checked : check receipt = true :=
    (List.all_eq_true.mp generated_all_check) receipt member
  exact (check_eq_true_iff_oracleReplayConforms receipt).1 checked

/-- Every generated record conserves the reached native control flow and the
exact observed call arguments/results.  This says nothing about Poseidon2 or
NIFS primitive correctness. -/
theorem controlFlowAndCallConservation_of_mem_generated
    (receipt : Receipt)
    (member : receipt ∈ Generated.all) :
    ControlFlowAndCallConservation receipt := by
  have replay := oracleReplayConforms_of_mem_generated receipt member
  exact controlFlowAndCallConservation_of_wellFormed receipt replay.1

/-- The normalized oracle replay equals the recorded Rust result for every
member of the bounded generated profile.  The receipt supplies the primitive
results, so this is a control-flow differential rather than primitive
conformance. -/
theorem nativeOutcome_eq_recorded_of_mem_generated
    (receipt : Receipt)
    (member : receipt ∈ Generated.all) :
    nativeOutcome receipt = receipt.outcome :=
  (oracleReplayConforms_of_mem_generated receipt member).2.2

end Nightstream.Implementation.Rust.CanonicalConformance.NativeStep
