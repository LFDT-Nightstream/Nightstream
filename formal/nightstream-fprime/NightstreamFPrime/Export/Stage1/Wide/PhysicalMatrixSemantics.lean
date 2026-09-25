import NightstreamFPrime.Export.Stage1.Wide.PhysicalMatrixSource

/-! Exact interpretation of the relocated matrix program. Source custody is
stated on the recovered reference rows; the emitted program reads the new
archive directly. Embedded sampler rows do not use the archive accessor. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PhysicalMatrixSemantics

open NightstreamFPrime.Layout NightstreamFPrime.Lifecycle
open Layout.MatrixProgram ProductionRelation
open PhysicalMatrixSource

def referenceSource (extra : Nat) (sourceRow : Nat → Option R1CS.Row) (index : Nat) : Option R1CS.Row := do
  let moved ← (PhysicalRelabel.row index).toOption
  let row ← sourceRow moved
  (inverse extra).row? row

theorem block_correct (extra : Nat) (before after : Block)
    (emitted : PhysicalMatrixSource.block extra before = .ok after) :
    after.rowCount = before.rowCount ∧ ∀ width sourceRow index,
      after.row? width sourceRow index =
        before.row? width (referenceSource extra sourceRow) index := by
  induction before generalizing after with
  | ordinary value =>
    cases moved : schedule value.rows with
    | error message => simp [PhysicalMatrixSource.block, moved] at emitted
    | ok rows =>
      simp [PhysicalMatrixSource.block, moved] at emitted
      subst after
      obtain ⟨count, indices⟩ := schedule_correct value.rows rows moved
      refine ⟨count, ?_⟩
      intro width sourceRow index
      simp only [Block.row?, Ordinary.Block.row?, indices, referenceSource,
        SourceProjection.compose_row (inverse extra) value.projection (inverse_unique extra)]
      simp only [Bind.bind, Option.bind_assoc]
  | mapped width projection inner ih =>
    cases moved : PhysicalMatrixSource.block extra inner with
    | error message => simp [PhysicalMatrixSource.block, moved] at emitted
    | ok relocated =>
      simp [PhysicalMatrixSource.block, moved] at emitted
      subst after
      obtain ⟨count, rows⟩ := ih relocated moved
      refine ⟨count, ?_⟩
      intro targetWidth sourceRow index
      simp only [Block.row?, rows]
  | multiplicationGrid value =>
    simp [PhysicalMatrixSource.block] at emitted
    subst after
    exact ⟨rfl, fun _ _ _ => rfl⟩
  | phi81Product value =>
    simp [PhysicalMatrixSource.block] at emitted
    subst after
    exact ⟨rfl, fun _ _ _ => rfl⟩
  | pin value =>
    simp [PhysicalMatrixSource.block] at emitted
    subst after
    exact ⟨rfl, fun _ _ _ => rfl⟩
  | poseidon value =>
    simp [PhysicalMatrixSource.block] at emitted
    subst after
    exact ⟨rfl, fun _ _ _ => rfl⟩
  | ordinaryTemplate value table =>
    simp [PhysicalMatrixSource.block] at emitted
    subst after
    exact ⟨rfl, fun _ _ _ => rfl⟩

private theorem blocks_correct (extra : Nat) (before after : List Block)
    (pairs : List.Forall₂ (fun a b => PhysicalMatrixSource.block extra a = .ok b) before after) :
    (after.map Block.rowCount).sum = (before.map Block.rowCount).sum ∧ ∀ width sourceRow index,
      Layout.MatrixProgram.Program.row?.select width sourceRow after index =
        Layout.MatrixProgram.Program.row?.select width (referenceSource extra sourceRow) before index := by
  induction pairs with
  | nil => exact ⟨rfl, fun _ _ _ => rfl⟩
  | @cons a b before after mapped pairs ih =>
    obtain ⟨count, rows⟩ := block_correct extra a b mapped
    refine ⟨by simp only [List.map_cons, List.sum_cons, count, ih.1], ?_⟩
    intro width sourceRow index
    simp only [Layout.MatrixProgram.Program.row?.select, count]
    split
    · exact rows width sourceRow index
    · exact ih.2 width sourceRow (index - a.rowCount)

theorem program_correct (extra : Nat) (before after : Layout.MatrixProgram.Program)
    (emitted : relocateProgram extra before = .ok after) :
    after.rowCount = before.rowCount ∧ ∀ width sourceRow index,
      after.row? width sourceRow index =
        before.row? width (referenceSource extra sourceRow) index := by
  change (Layout.MatrixProgram.Program.mk <$> before.blocks.mapM (PhysicalMatrixSource.block extra)) = .ok after at emitted
  cases moved : before.blocks.mapM (PhysicalMatrixSource.block extra) with
  | error message => simp [moved] at emitted
  | ok blocks =>
    simp only [moved, Except.map_ok, Except.ok.injEq] at emitted
    subst after
    exact blocks_correct extra before.blocks blocks
      (PhysicalRelabel.mapM_pairs (PhysicalMatrixSource.block extra) before.blocks blocks moved)

/-- Reusing the phase custody proofs yields exact correspondence for every
row and port of the relocated complete program. -/
theorem exact (application : RetainedLayout.Program) (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application) (after : Layout.MatrixProgram.Program)
    (emitted : PhysicalMatrixSource.program application compiled = .ok after)
    (sourceRow : Nat → Option R1CS.Row)
    (custody : ReusedMatrixPrograms.SourceCustody application (FixedPoint.relation application compiled fits)
      fits.package (referenceSource (PerApplicationPackage.directAddedPrivateColumnCount application) sourceRow)) :
    Exact after (FixedPoint.structuralPlan application compiled fits) sourceRow := by
  have before := MatrixProgram.exact application compiled (FixedPoint.relation application compiled fits) fits
    (referenceSource (PerApplicationPackage.directAddedPrivateColumnCount application) sourceRow) custody
  rw [FixedPoint.plan_fixedPoint] at before
  obtain ⟨count, rows⟩ := program_correct
    (PerApplicationPackage.directAddedPrivateColumnCount application)
    (MatrixProgram.program application compiled) after emitted
  refine ⟨count.trans before.rowCount, ?_⟩
  intro row
  rw [rows]
  exact before.row? row

end NightstreamFPrime.Export.Stage1.Wide.PhysicalMatrixSemantics
