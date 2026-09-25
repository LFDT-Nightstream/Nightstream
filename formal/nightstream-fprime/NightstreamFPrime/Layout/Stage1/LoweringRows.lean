import NightstreamFPrime.Layout.Stage1.Lowering
import NightstreamFPrime.Layout.Stage1.SpartanRows
import NightstreamFPrime.Layout.Stage1.AssemblerApplicationCompleteness

/-!
Owns the one physical Stage 1 row order for a verifier-selected application.

The validated prefix is first mapped to Spartan order. Application-private
columns then occupy the old constant/public boundary, so the old constant and
public suffix moves by one exact displacement. The selected Lean application
rows and the five NextPreimage rows follow. No file boundary adds a row.
-/

namespace NightstreamFPrime.Layout.Stage1.Lowering

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-! ## Complete physical layout -/

def physicalRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) : List R1CS.Row :=
  (shiftRows program (Spartan.remappedRows relation) ++
    applicationRows program) ++ nextPreimageRows

def physicalRowCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  (physicalRows relation program).length

def jointDomain
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  max (physicalRowCount relation program) (totalColumnCount program)

theorem physicalRowCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) :
    physicalRowCount relation program =
      29218024 + R1CS.totalRowCount (applicationConstraints program) + 5 := by
  have prefixLength : (Spartan.remappedRows relation).length = 29218024 := by
    unfold Spartan.remappedRows Spartan.remapRows
    rw [List.length_map]
    exact Spartan.sourceRowCount_eq relation
  unfold physicalRowCount physicalRows shiftRows
  rw [List.length_append, List.length_append, List.length_map,
    prefixLength, applicationRows_length,
    nextPreimageRows_length]

/-! ## Sole logical circuit instantiation -/

/-- The canonical layout instantiates the sole logical Stage 1 circuit at its
verifier-owned compact root. -/
noncomputable def logicalCircuit
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation)) : FormalCircuit :=
  Lifecycle.Stage1.circuit relation ajtai program
    (AssemblerInputs.interface relation program) template
    (AssemblerInputs.rootOffset program)
    (AssemblerApplicationCompleteness.rootCompleteness relation ajtai program
      template)

theorem logicalCircuit_coverage
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation)) :
    (Circuit.ops (logicalCircuit relation ajtai program template).main
      (AssemblerInputs.rootOffset program)).length = 8 := by
  exact Lifecycle.Stage1.circuit_coverage relation ajtai program
    (AssemblerInputs.interface relation program) template
    (AssemblerInputs.rootOffset program)
    (AssemblerApplicationCompleteness.rootCompleteness relation ajtai program
      template)

end NightstreamFPrime.Layout.Stage1.Lowering
