import Nightstream.Implementation.R1CS.Artifacts.PiCcsExecution

/-!
Generic soundness regression for per-execution Rust `Pi_CCS` assurance.

Assurance tier: model-level. The generated selected-profile receipt has a
separate Rust drift owner and a compact structural probe.
-/

set_option autoImplicit false

namespace NightstreamTests.PiCcsExecutionReceipt

open Nightstream.Implementation.Rust.PiCcsExecution

example
    (expectedRelationId : List Nat)
    (statement : PiCcsCanonicalStatement)
    (rustProof : PiCcsExecutionProof)
    (checked : checkReceipt expectedRelationId statement rustProof = true) :
    PaperPiCCS.Accepts expectedRelationId statement rustProof :=
  checkReceipt_sound expectedRelationId statement rustProof checked

end NightstreamTests.PiCcsExecutionReceipt
