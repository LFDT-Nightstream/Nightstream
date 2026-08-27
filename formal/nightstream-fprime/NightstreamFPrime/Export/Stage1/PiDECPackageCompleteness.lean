import NightstreamFPrime.Export.Stage1.PackageCompleteness

/-!
Owns constructive completeness for the canonical PiDEC package-row packet.

The semantic completion runs in Lean source-column order. One proved Spartan
copy then writes exactly the 3,618 PiDEC logical and R1CS-fresh columns into
the final package assignment. The compiled-row equality fixes the final row
order; no exporter or Rust code selects it.
-/

namespace NightstreamFPrime.Export.Stage1.PiDECPackageCompleteness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def phaseInterface :
    NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal.Interface
      Data.logicalWidth Data.publicFits :=
  PiDECArithmetic.phaseInterface Data.logicalWidth Data.publicFits

def piRlcInterface :
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.Interface
      Data.logicalWidth Data.publicFits :=
  PiRLCInputs.interface
    (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)

theorem targetStart_eq :
    Spartan.sourceToSpartan PiDECInputs.phaseOffset =
      Data.piDecWitnessStart := by
  rfl

theorem targetEnd_eq :
    Data.piDecWitnessStart + 3618 = Spartan.privateColumnCount := by
  rfl

/-- A valid semantic PiDEC phase constructs every canonical PiDEC package row
and changes only its declared final private-column interval. -/
theorem completeRows
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (env : Env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal.Assumptions relation
        phaseInterface PiDECInputs.phaseOffset (Spartan.pullback env))
    (phase : NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.PhaseHolds
      relation ajtai phaseInterface PiDECInputs.phaseOffset
        (Spartan.pullback env)) :
    ∃ completed,
      AgreesOutside env completed Data.piDecWitnessStart 3618 ∧
        PackageCompleteness.PiDECRowsHold completed := by
  rcases
      NightstreamFPrime.Layout.PiDEC.v1_1.physical_complete_production
        relation ajtai phaseInterface PiDECInputs.phaseOffset
        (Spartan.pullback env) (PiDECInputs.inputShapes relation)
        assumptions phase with
    ⟨source, sourceAgrees, sourceRows⟩
  let completed := Spartan.copyMappedInterval env source
    PiDECInputs.phaseOffset 3618
  have targetAgrees : AgreesOutside env completed
      Data.piDecWitnessStart 3618 := by
    rw [← targetStart_eq]
    exact Spartan.copyMappedInterval_agreesOutside env source
      PiDECInputs.phaseOffset 3618
  have startLocal : Spartan.piCcsPhaseOffset ≤ PiDECInputs.phaseOffset := by
    norm_num [Spartan.piCcsPhaseOffset, PiDECInputs.phaseOffset,
      PiDECInputs.proofInputStart, PiDECInputs.proofInputColumnCount,
      PiDECInputs.childCount, PiDECInputs.commitmentWordsPerChild,
      PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
      PiDECInputs.publicInputWordsPerChild]
  have targetPrivate :
      Spartan.sourceToSpartan PiDECInputs.phaseOffset + 3618 ≤
        Spartan.privateColumnCount := by
    rw [targetStart_eq, targetEnd_eq]
  have remappedRows : R1CS.RowsHold completed
      (Spartan.remapRows
        (NightstreamFPrime.Layout.PiDEC.v1_1.physicalRows relation
          phaseInterface PiDECInputs.phaseOffset)) := by
    exact Spartan.remapRows_hold_copyMappedInterval
      (NightstreamFPrime.Layout.PiDEC.v1_1.physicalRows relation
        phaseInterface PiDECInputs.phaseOffset)
      env source PiDECInputs.phaseOffset 3618 startLocal targetPrivate
      sourceAgrees sourceRows
  have exactRows := PiDECArithmetic.Plan.rows_to_layout
    (PiDECArithmetic.canonicalPlan Data.logicalWidth Data.publicFits)
    (PiDECArithmetic.canonicalLayoutPlan relation)
    (PiDECArithmetic.canonicalPlan_matches relation)
  refine ⟨completed, targetAgrees, ⟨?_⟩⟩
  rw [exactRows]
  exact remappedRows

private theorem agreesOutside_widen
    {before after : Env} {start length innerStart innerLength : Nat}
    (inner : AgreesOutside before after innerStart innerLength)
    (starts : start ≤ innerStart)
    (ends : innerStart + innerLength ≤ start + length) :
    AgreesOutside before after start length := by
  intro index outside
  apply inner index
  rcases outside with beforeStart | afterEnd
  · exact Or.inl (lt_of_lt_of_le beforeStart starts)
  · exact Or.inr (Nat.le_trans ends afterEnd)

theorem targetAgrees_implies_phaseSuffix
    (before after : Env)
    (agrees : AgreesOutside before after Data.piDecWitnessStart 3618) :
    AgreesOutside before after PackageCompleteness.phaseSuffixStart
      PackageCompleteness.phaseSuffixLength := by
  apply agreesOutside_widen agrees
  · norm_num [PackageCompleteness.phaseSuffixStart,
      Data.piDecWitnessStart, PiDECInputs.phaseOffset,
      PiDECInputs.proofInputStart, PiDECInputs.proofInputColumnCount,
      PiDECInputs.childCount, PiDECInputs.commitmentWordsPerChild,
      PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
      PiDECInputs.publicInputWordsPerChild,
      Spartan.sourceToSpartan, Spartan.pilotSourceColumnCount,
      Spartan.proofInputSourceStart, Spartan.piCcsPhaseOffset,
      Spartan.piCcsLocalStart, PiRLCInputs.phaseOffset]
  · rw [targetEnd_eq, PackageCompleteness.phaseSuffixEnd_eq]

private theorem pullback_agreesBelow_piDec
    (before after : Env)
    (agrees : AgreesOutside before after Data.piDecWitnessStart 3618) :
    ∀ index, index < PiDECInputs.phaseOffset →
      Spartan.pullback after index = Spartan.pullback before index := by
  intro index below
  unfold Spartan.pullback
  apply agrees
  rcases Spartan.sourceToSpartan_before_piCcsLocal index
      PiDECInputs.phaseOffset (by
        norm_num [Spartan.piCcsPhaseOffset, PiDECInputs.phaseOffset,
          PiDECInputs.proofInputStart, PiDECInputs.proofInputColumnCount,
          PiDECInputs.childCount, PiDECInputs.commitmentWordsPerChild,
          PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
          PiDECInputs.publicInputWordsPerChild]) below with
    mappedBefore | mappedPublic
  · apply Or.inl
    rw [← targetStart_eq]
    exact mappedBefore
  · apply Or.inr
    rw [targetEnd_eq]
    exact mappedPublic.le

theorem piRlcPhysicalRows_varsBelow
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (env : Env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.Assumptions relation
        piRlcInterface PiRLCInputs.phaseOffset (Spartan.pullback env))
    (phase : NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.PhaseHolds
      relation ajtai piRlcInterface PiRLCInputs.phaseOffset
        (Spartan.pullback env)) :
    ∀ row ∈ NightstreamFPrime.Layout.PiRLC.v1_1.physicalRows relation
        piRlcInterface PiRLCInputs.phaseOffset,
      row.VarsBelow PiDECInputs.phaseOffset := by
  have physicalEnd :
      NightstreamFPrime.Layout.PiRLC.v1_1.physicalColumnCount relation
          piRlcInterface PiRLCInputs.phaseOffset ≤
        PiDECInputs.phaseOffset := by
    rw [NightstreamFPrime.Layout.PiRLC.v1_1.physicalColumnCount_eq_production
      relation piRlcInterface PiRLCInputs.phaseOffset
      (PiRLCInputs.inputShapes relation)]
    norm_num [PiRLCInputs.phaseOffset, PiDECInputs.phaseOffset,
      PiDECInputs.proofInputStart, PiDECInputs.proofInputColumnCount,
      PiDECInputs.childCount, PiDECInputs.commitmentWordsPerChild,
      PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
      PiDECInputs.publicInputWordsPerChild]
  intro row member
  exact (NightstreamFPrime.Layout.PiRLC.v1_1.physicalRows_varsBelow_of_phase
    relation ajtai piRlcInterface PiRLCInputs.phaseOffset
    (Spartan.pullback env) assumptions phase row member).mono row physicalEnd

theorem piRlcPhysicalRows_of_piDecAgreesOutside
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (before after : Env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.Assumptions relation
        piRlcInterface PiRLCInputs.phaseOffset (Spartan.pullback before))
    (phase : NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.PhaseHolds
      relation ajtai piRlcInterface PiRLCInputs.phaseOffset
        (Spartan.pullback before))
    (agrees : AgreesOutside before after Data.piDecWitnessStart 3618)
    (holds : R1CS.RowsHold before (Spartan.remapRows
      (NightstreamFPrime.Layout.PiRLC.v1_1.physicalRows relation
        piRlcInterface PiRLCInputs.phaseOffset))) :
    R1CS.RowsHold after (Spartan.remapRows
      (NightstreamFPrime.Layout.PiRLC.v1_1.physicalRows relation
        piRlcInterface PiRLCInputs.phaseOffset)) := by
  have sourceHolds := (Spartan.remapRows_hold before _).mp holds
  have sourceAfter := R1CS.rowsHold_of_agree_below
    (NightstreamFPrime.Layout.PiRLC.v1_1.physicalRows relation
      piRlcInterface PiRLCInputs.phaseOffset)
    PiDECInputs.phaseOffset (Spartan.pullback before)
    (Spartan.pullback after)
    (piRlcPhysicalRows_varsBelow relation ajtai before assumptions phase)
    (pullback_agreesBelow_piDec before after agrees) sourceHolds
  exact (Spartan.remapRows_hold after _).mpr sourceAfter

/-- Completing PiDEC after a valid Pilot/PiCCS/PiRLC prefix produces one
assignment satisfying every row of the current canonical package. -/
theorem completePackageRows
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (env : Env)
    (piRlcAssumptions :
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.Assumptions relation
        piRlcInterface PiRLCInputs.phaseOffset (Spartan.pullback env))
    (piRlcPhase : NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.PhaseHolds
      relation ajtai piRlcInterface PiRLCInputs.phaseOffset
        (Spartan.pullback env))
    (piDecAssumptions :
      NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal.Assumptions relation
        phaseInterface PiDECInputs.phaseOffset (Spartan.pullback env))
    (piDecPhase : NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.PhaseHolds
      relation ajtai phaseInterface PiDECInputs.phaseOffset
        (Spartan.pullback env))
    (pilotChains : ∀ chain ∈ [Data.priorChain, Data.outputChain],
      NightstreamFPrime.Export.Package.HashChainHolds
        (Data.circuitPackage ()) chain env)
    (pilotAssertions : ∀ row ∈
      Data.liftPilotRows (NightstreamFPrime.Export.PilotData.assertionRows ()),
        row.Holds env)
    (piCcs : PackageCompleteness.PiCCSRowsHold env)
    (piRlcPhysical : R1CS.RowsHold env (Spartan.remapRows
      (NightstreamFPrime.Layout.PiRLC.v1_1.physicalRows relation
        piRlcInterface PiRLCInputs.phaseOffset))) :
    ∃ completed,
      AgreesOutside env completed Data.piDecWitnessStart 3618 ∧
        (Data.circuitPackage ()).RowsHold completed := by
  rcases completeRows relation ajtai env piDecAssumptions piDecPhase with
    ⟨completed, agrees, piDec⟩
  have suffixAgrees := targetAgrees_implies_phaseSuffix env completed agrees
  have piRlcPhysicalAfter := piRlcPhysicalRows_of_piDecAgreesOutside
    relation ajtai env completed piRlcAssumptions piRlcPhase agrees
    piRlcPhysical
  have packets := PiRLCPackageCompleteness.remappedPhysicalRows_imply_packets
    relation completed piRlcPhysicalAfter
  have piRlc := PackageCompleteness.piRlcRowsHold_of_packets completed packets
  refine ⟨completed, agrees, PackageCompleteness.rowsHold_of_phaseRows
    completed ?_ ?_ ?_ piRlc piDec⟩
  · exact PackageCompleteness.pilotHashChains_of_piRlcAgreesOutside
      env completed suffixAgrees pilotChains
  · exact PackageCompleteness.pilotAssertionRows_of_piRlcAgreesOutside
      env completed suffixAgrees pilotAssertions
  · exact PackageCompleteness.piCcsRows_of_piRlcAgreesOutside relation
      env completed suffixAgrees piCcs

end NightstreamFPrime.Export.Stage1.PiDECPackageCompleteness
