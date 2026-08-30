import NightstreamFPrime.Export.Package
import NightstreamFPrime.Lifecycle.Stage1.Terminal

/-!
Owns canonical package metadata for the outer Stage 1 terminal verifier.

The terminal verifier reuses the complete F′ relation. It checks 16 running
CE claims and one fresh CCS claim against the same 14 matrices, so this edge
adds no circuit row or column. The final package constructor must install this
metadata after the concrete application and complete relation exist.
-/

namespace NightstreamFPrime.Export.Stage1.TerminalPackage

open NightstreamFPrime.Export.Package
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- The terminal verifier uses every row of the final F′ relation. -/
def layoutFor (package : CircuitPackage) : TerminalLayout where
  rowStart := 0
  rowCount := package.relation.rowCount
  runningClaims := productionShape.runningCount
  freshClaims := productionShape.freshCount

/-- Install the canonical outer-terminal metadata without changing any
executable row, witness instruction, layout dimension, or relation field. -/
def install (package : CircuitPackage) : CircuitPackage :=
  { package with terminal := some (layoutFor package) }

@[simp] theorem install_terminal (package : CircuitPackage) :
    (install package).terminal = some (layoutFor package) := by
  rfl

@[simp] theorem install_layout (package : CircuitPackage) :
    (install package).layout = package.layout := by
  rfl

@[simp] theorem install_relation (package : CircuitPackage) :
    (install package).relation = package.relation := by
  rfl

@[simp] theorem install_witnessInstructions (package : CircuitPackage) :
    (install package).witnessInstructions = package.witnessInstructions := by
  rfl

@[simp] theorem install_assertionRows (package : CircuitPackage) :
    (install package).assertionRows = package.assertionRows := by
  rfl

@[simp] theorem layoutFor_rowStart (package : CircuitPackage) :
    (layoutFor package).rowStart = 0 := by
  rfl

@[simp] theorem layoutFor_rowCount (package : CircuitPackage) :
    (layoutFor package).rowCount = package.relation.rowCount := by
  rfl

theorem layoutFor_runningClaims (package : CircuitPackage) :
    (layoutFor package).runningClaims = 16 := by
  rfl

theorem layoutFor_freshClaims (package : CircuitPackage) :
    (layoutFor package).freshClaims = 1 := by
  rfl

end NightstreamFPrime.Export.Stage1.TerminalPackage
