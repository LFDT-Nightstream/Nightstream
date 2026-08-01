import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Program

/-!
Contract: compact, proof-free descriptor for the direct terminal R1CS.

Assurance tier: model-level.

Owns: the dimensions that reconstruct the statement-specialized terminal
relation and its exact derived physical cost.

Does not own: terminal statements, commitment keys, materialized rows,
Spartan, WHIR, JSON, Rust, or deployment suitability.

Emits constraints: none. A verifier combines this descriptor with its
authoritative key and terminal statements to construct the terminal rows.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Typed
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Static dimensions needed to reconstruct one direct terminal R1CS. -/
structure Descriptor where
  rowVariables : Nat
  logicalWidth : Nat
  recursiveRows : Nat
  matrixCount : Nat
  publicRingColumns : Nat
  verifierRows : Nat
deriving DecidableEq, Repr

namespace Descriptor

/-- Complete Phi81 carrier width used by each terminal claim. -/
def carrierWidth (descriptor : Descriptor) : Nat :=
  Phi81CarrierLayout.carrierWidth descriptor.logicalWidth

/-- Exact field width of the ring-aligned public projection. -/
def publicWidth (descriptor : Descriptor) : Nat :=
  ringDegree * descriptor.publicRingColumns

/-- Exact direct-R1CS cost derived from the descriptor dimensions. -/
def cost (descriptor : Descriptor) : Cost :=
  ⟨productionGlobalParams.k *
        (descriptor.verifierRows * ringDegree + descriptor.publicWidth +
          2 * descriptor.carrierWidth +
          2 * (descriptor.matrixCount * ringDegree)) +
      (descriptor.verifierRows * ringDegree + descriptor.publicWidth +
        2 * descriptor.carrierWidth + 2 * descriptor.recursiveRows),
    (productionGlobalParams.k + 1) * descriptor.carrierWidth,
    1 +
      productionGlobalParams.k *
        (descriptor.verifierRows * ringDegree + descriptor.publicWidth +
          2 * (descriptor.matrixCount * ringDegree)) +
      (descriptor.verifierRows * ringDegree + descriptor.publicWidth),
    productionGlobalParams.k * descriptor.carrierWidth +
      descriptor.carrierWidth + descriptor.recursiveRows⟩

/-- Descriptor projected from the Lean-owned native recursive program. -/
def ofProgram
    (program : NativeCcsProgram.Program)
    (rowVariables publicRingColumns verifierRows : Nat) : Descriptor where
  rowVariables := rowVariables
  logicalWidth := program.columnIds.length
  recursiveRows := program.rows.length
  matrixCount := NativeCcsSelector.matrixCount
  publicRingColumns := publicRingColumns
  verifierRows := verifierRows

/-- The descriptor cost is the exact terminal-program formula. -/
@[simp] theorem cost_ofProgram
    (program : NativeCcsProgram.Program)
    (rowVariables publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length) :
    cost (ofProgram program rowVariables publicRingColumns verifierRows) =
      TerminalR1cs.Program.cost program
        { rowVariables := rowVariables
          logicalWidth := program.columnIds.length
          matrixCount := NativeCcsSelector.matrixCount
          publicRingColumns := publicRingColumns
          publicFits := publicFits }
        verifierRows := by
  rfl

end Descriptor

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs
