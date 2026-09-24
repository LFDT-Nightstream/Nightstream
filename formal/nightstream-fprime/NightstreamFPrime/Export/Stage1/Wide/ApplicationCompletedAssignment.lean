import NightstreamFPrime.Export.Stage1.Wide.SourceAssignment
import NightstreamFPrime.Export.Stage1.ApplicationCompactWitness

/-! Construct the three application permutations, then preserve their retained
values while direct PiRLC fills its own block. The four advice words are exact. -/

namespace NightstreamFPrime.Export.Stage1.Wide.ApplicationCompletedAssignment

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open Poseidon2HashChainV1Package (application fits)

def suffix (env : Env) (message : Fin 4 → F) :=
  ApplicationCompactWitness.privateSuffix
    (ApplicationCompactWitness.priorValues (SourceAssignment.targetEnv env)) message

private theorem source (env : Env) (column : Nat) (before : column < SourceAssignment.prefixEnd) :
    SourceAssignment.targetEnv env (Layout.Stage1.Spartan.sourceToSpartan column) = env column := by
  rw [SourceAssignment.targetEnv_source env column (by
    change column < 19513117 at before
    rw [Layout.Stage1.Spartan.sourceColumnCount_eq]; omega)]
  exact SourceAssignment.sourceEnv_prefix env column before

theorem rowsZero (env : Env) (message : Fin 4 → F)
    (step : (List.ofFn fun lane : Fin 4 => env (Layout.Stage1.ApplicationInputs.outputSourceColumn lane)) =
      application.step (List.ofFn fun lane : Fin 4 => env (Layout.Stage1.ApplicationInputs.inputSourceColumn lane))
        (List.ofFn message)) :
    (Stage1Plan.application application fits.package).RowsZero
      (SourceAssignment.assignment application env (suffix env message)) := by
  have inputValues (lane : Fin 4) :
      ApplicationCompactWitness.priorValues (SourceAssignment.targetEnv env) lane =
        env (Layout.Stage1.ApplicationInputs.inputSourceColumn lane) := by
    apply source
    have bound : lane.val < 4 := lane.isLt
    change 35 + lane.val < 19513117
    omega
  have outputValues (lane : Fin 4) :
      SourceAssignment.targetEnv env (Layout.Stage1.ApplicationInputs.outputColumn lane) =
        env (Layout.Stage1.ApplicationInputs.outputSourceColumn lane) := by
    apply source
    have bound : lane.val < 4 := lane.isLt
    change 49698 + lane.val < 19513117
    omega
  apply (AssignmentProjection.copied_rowsZero_iff application
    (SourceAssignment.raw application env (suffix env message)).assignment
    _ (ReadSupport.application application fits.package _) _).mpr
  apply ApplicationCompactWitness.complete_of_base (SourceAssignment.targetEnv env) message
    (SourceAssignment.raw application env (suffix env message)) rfl
  exact (congrArg List.ofFn (funext outputValues)).trans
    (step.trans (congrArg (fun values => application.step values (List.ofFn message))
      (congrArg List.ofFn (funext inputValues)).symm))

theorem advice (env : Env) (message : Fin 4 → F) :
    Lifecycle.Stage1.Application.witnessValue (Layout.Stage1.ApplicationInputs.interface application)
      (Layout.Stage1.ApplicationInputs.localStart application)
      (ProductionRelation.SourceCompiler.sourceEnv (SourceAssignment.raw application env (suffix env message)).base) =
        List.ofFn message :=
  ApplicationCompactWitness.witnessValue (SourceAssignment.targetEnv env) message

end NightstreamFPrime.Export.Stage1.Wide.ApplicationCompletedAssignment
