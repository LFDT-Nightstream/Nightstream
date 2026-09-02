import NightstreamFPrime.Export.MatrixProgram.PlanBridge
import NightstreamFPrime.Export.Stage1.ApplicationDirectPlan
import NightstreamFPrime.Export.Stage1.PerApplicationSourceProjection

/-!
Owns the compact matrix program for one verifier-selected Stage 1
application. Lean fixes its four retained source intervals and the exact
application-row suffix of the canonical package.
-/

namespace NightstreamFPrime.Export.Stage1.ApplicationMatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open ApplicationRetainedBlocks
open ApplicationRetainedGeometry

abbrev ApplicationProgram := Lifecycle.Stage1.Application.Program
abbrev FitsTwoPow28 (application : ApplicationProgram) :=
  PerApplicationPackage.FitsTwoPow28 application

def inputRange (application : ApplicationProgram) : SourceRange :=
  SourceRange.ofSemantic (inputBlock application) (inputStart application)
    ApplicationInputs.currentWordStart Lifecycle.Stage1.Application.stateWordCount
    0

def witnessRange (application : ApplicationProgram) : SourceRange :=
  SourceRange.ofSemantic (witnessBlock application) (witnessStart application)
    ApplicationInputs.witnessStart application.witnessWordCount 0

def outputRange (application : ApplicationProgram) : SourceRange :=
  SourceRange.ofSemantic (outputBlock application) (outputStart application)
    49428 Lifecycle.Stage1.Application.stateWordCount 0

def localRange (application : ApplicationProgram) : SourceRange :=
  SourceRange.ofSemantic (localBlock application) (localStart application)
    (ApplicationInputs.localStart application) (localCount application) 0

/-- Complete fail-closed source substitution for one selected application. -/
def substitution (application : ApplicationProgram) : SourceSubstitution where
  ranges := [inputRange application, witnessRange application,
    outputRange application, localRange application]

def rowSchedule (application : ApplicationProgram) : IndexSchedule :=
  .rangeList [⟨PerApplicationPackage.basePackage.layout.rowCount,
    (PerApplicationPackage.applicationPlan application).rowCount⟩]

def directRowSchedule (application : ApplicationProgram) : IndexSchedule :=
  .rangeList [⟨29218024,
    (PerApplicationPackage.directApplicationPlan application).rowCount⟩]

theorem directRowSchedule_eq_rowSchedule (application : ApplicationProgram) :
    directRowSchedule application = rowSchedule application := by
  unfold directRowSchedule rowSchedule
  rw [PerApplicationPackage.directApplicationPlan_eq_applicationPlan,
    PerApplicationPackage.basePackage_rowCount_eq]

@[csimp] theorem rowSchedule_eq_directRowSchedule :
    @rowSchedule = @directRowSchedule := by
  funext application
  exact (directRowSchedule_eq_rowSchedule application).symm

@[simp] theorem rowSchedule_count (application : ApplicationProgram) :
    (rowSchedule application).count =
      (PerApplicationPackage.applicationPlan application).rowCount := by
  rfl

def ordinaryBlock {application : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry application logicalWidth) : Ordinary.Block where
  rows := rowSchedule application
  oneColumn := (oneColumn geometry).val
  substitution := substitution application
  projection := PerApplicationSourceProjection.application

def matrixProgram {application : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry application logicalWidth) : MatrixProgram.Program where
  blocks := [.ordinary (ordinaryBlock geometry)]

@[simp] theorem matrixProgram_rowCount
    {application : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry application logicalWidth) :
    (matrixProgram geometry).rowCount =
      (PerApplicationPackage.applicationPlan application).rowCount := by
  rfl

end NightstreamFPrime.Export.Stage1.ApplicationMatrixProgram
