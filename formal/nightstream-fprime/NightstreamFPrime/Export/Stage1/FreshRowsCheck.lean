import NightstreamFPrime.Export.Stage1.PiDECMatrixNumericRows
import NightstreamFPrime.Export.Stage1.PiDECCanonicalSourceCache
import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalAssignment
import NightstreamFPrime.Export.Stage1.PiCCSSourceImages

/-! Check the scalar production polynomial on every active canonical row.
The Poseidon path stores one invocation's existing 86 numeric rows. The
source-row cache is fixed by the canonical package theorem. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.FreshRowsCheck

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Layout.ProductionRelation

@[specialize] def allFrom (predicate : Nat → Bool) (first : Nat) : Nat → Bool
  | 0 => true
  | count + 1 => predicate first && allFrom predicate (first + 1) count

private theorem allFrom_sound (predicate : Nat → Bool) (first count : Nat)
    (checked : allFrom predicate first count = true)
    (index : Nat) (lower : first ≤ index) (upper : index < first + count) :
    predicate index = true := by
  induction count generalizing first with
  | zero => omega
  | succ count inductionHypothesis =>
      have both : predicate first = true ∧
          allFrom predicate (first + 1) count = true := by
        simpa only [allFrom, Bool.and_eq_true] using checked
      by_cases same : index = first
      · simpa only [same] using both.1
      · exact inductionHypothesis (first + 1) both.2 (by omega) (by omega)

def checkValues (values : Fin matrixCount → F) : Bool :=
  decide (evaluatePolynomial baseOps polynomial values = 0)

def checkRow : Option (Vector F matrixCount) → Bool
  | none => false
  | some values => checkValues values.get

@[specialize] def checkInvocation (block : Poseidon.Block) {columns : Nat}
    (read : Fin columns → F) (invocation : Fin block.invocationCount) : Bool :=
  match PiDECPoseidonNumericBlock.loadInvocation? block columns invocation with
  | none => false
  | some interface =>
      let values := PiDECPoseidonNumericRows.stored read interface
      allFrom (fun row =>
        if bound : row < 86 then checkValues ((values.get ⟨row, bound⟩).get)
        else false) 0 86

@[specialize] def checkBlock (block : MatrixProgram.Block) {columns : Nat}
    (sourceRow : Nat → Option R1CS.Row) (read : Fin columns → F) : Bool :=
  match block with
  | .poseidon poseidon =>
      allFrom (fun invocation =>
        if bound : invocation < poseidon.invocationCount then
          checkInvocation poseidon read ⟨invocation, bound⟩
        else false) 0 poseidon.invocationCount
  | other =>
      allFrom (fun row => checkRow
        (PiDECMatrixNumericRows.blockRow? other sourceRow read row)) 0 other.rowCount

def checkProgram (program : MatrixProgram.Program) {columns : Nat}
    (sourceRow : Nat → Option R1CS.Row) (read : Fin columns → F) : Bool :=
  program.blocks.all (fun block => checkBlock block sourceRow read)


/-- Poseidon work stays invocation-local, so one unit builds the same existing
86-row table. Other block kinds use their existing scalar rows as units. -/
def checkBlockUnits : MatrixProgram.Block → Nat
  | .poseidon block => block.invocationCount
  | other => other.rowCount

@[specialize] def checkBlockUnit (block : MatrixProgram.Block) {columns : Nat}
    (sourceRow : Nat → Option R1CS.Row) (read : Fin columns → F) (unit : Nat) : Bool :=
  match block with
  | .poseidon poseidon =>
      if bound : unit < poseidon.invocationCount then
        checkInvocation poseidon read ⟨unit, bound⟩
      else false
  | other =>
      if unit < other.rowCount then
        checkRow (PiDECMatrixNumericRows.blockRow? other sourceRow read unit)
      else false

/-- A task checks exactly this half-open interval of the unchanged block
units. Empty ranges succeed; an out-of-bounds visited unit rejects. -/
@[specialize] def checkBlockRange (block : MatrixProgram.Block) {columns : Nat}
    (sourceRow : Nat → Option R1CS.Row) (read : Fin columns → F)
    (first count : Nat) : Bool :=
  allFrom (checkBlockUnit block sourceRow read) first count

private theorem allFrom_complete (predicate : Nat → Bool) (first count : Nat)
    (checked : ∀ index, first ≤ index → index < first + count → predicate index = true) :
    allFrom predicate first count = true := by
  induction count generalizing first with
  | zero => rfl
  | succ count inductionHypothesis =>
      have head := checked first (Nat.le_refl _) (by omega)
      have tail := inductionHypothesis (first + 1) (fun index lower upper =>
        checked index (by omega) (by omega))
      change (predicate first && allFrom predicate (first + 1) count) = true
      rw [head, tail]
      rfl

theorem checkBlockRange_unit (block : MatrixProgram.Block) {columns : Nat}
    (sourceRow : Nat → Option R1CS.Row) (read : Fin columns → F)
    (first count : Nat) (checked : checkBlockRange block sourceRow read first count = true)
    (unit : Nat) (lower : first ≤ unit) (upper : unit < first + count) :
    checkBlockUnit block sourceRow read unit = true :=
  allFrom_sound (checkBlockUnit block sourceRow read) first count checked unit lower upper

/-- Checking every active unit recovers the original block result, including
zero-sized blocks. No row arithmetic or result predicate changes. -/
theorem checkBlock_of_units (block : MatrixProgram.Block) {columns : Nat}
    (sourceRow : Nat → Option R1CS.Row) (read : Fin columns → F)
    (checked : ∀ unit, unit < checkBlockUnits block →
      checkBlockUnit block sourceRow read unit = true) :
    checkBlock block sourceRow read = true := by
  cases block with
  | poseidon block =>
      apply allFrom_complete
      intro unit _ upper
      exact checked unit (by simpa only [Nat.zero_add] using upper)
  | ordinaryTemplate block rows =>
      apply allFrom_complete
      intro unit _ upper
      have bounded : unit < (MatrixProgram.Block.ordinaryTemplate block rows).rowCount := by
        simpa only [Nat.zero_add] using upper
      simpa only [checkBlockUnit, if_pos bounded] using checked unit bounded
  | mapped width projection block =>
      apply allFrom_complete
      intro unit _ upper
      have bounded : unit < (MatrixProgram.Block.mapped width projection block).rowCount := by
        simpa only [Nat.zero_add] using upper
      simpa only [checkBlockUnit, if_pos bounded] using checked unit bounded
  | ordinary block =>
      apply allFrom_complete
      intro unit _ upper
      have bounded : unit < (MatrixProgram.Block.ordinary block).rowCount := by
        simpa only [Nat.zero_add] using upper
      simpa only [checkBlockUnit, if_pos bounded] using checked unit bounded
  | multiplicationGrid block =>
      apply allFrom_complete
      intro unit _ upper
      have bounded : unit < (MatrixProgram.Block.multiplicationGrid block).rowCount := by
        simpa only [Nat.zero_add] using upper
      simpa only [checkBlockUnit, if_pos bounded] using checked unit bounded
  | phi81Product block =>
      apply allFrom_complete
      intro unit _ upper
      have bounded : unit < (MatrixProgram.Block.phi81Product block).rowCount := by
        simpa only [Nat.zero_add] using upper
      simpa only [checkBlockUnit, if_pos bounded] using checked unit bounded
  | pin block =>
      apply allFrom_complete
      intro unit _ upper
      have bounded : unit < (MatrixProgram.Block.pin block).rowCount := by
        simpa only [Nat.zero_add] using upper
      simpa only [checkBlockUnit, if_pos bounded] using checked unit bounded

/-- Every active unit must occur in a returned successful task. The proof
allows repeated or empty ranges; they cannot hide an unchecked active unit. -/
theorem checkBlock_of_ranges (block : MatrixProgram.Block) {columns parts : Nat}
    (sourceRow : Nat → Option R1CS.Row) (read : Fin columns → F)
    (first count : Fin parts → Nat)
    (covered : ∀ unit, unit < checkBlockUnits block →
      ∃ part, first part ≤ unit ∧ unit < first part + count part)
    (checked : ∀ part, checkBlockRange block sourceRow read (first part) (count part) = true) :
    checkBlock block sourceRow read = true := by
  apply checkBlock_of_units block sourceRow read
  intro unit bounded
  obtain ⟨part, lower, upper⟩ := covered unit bounded
  exact checkBlockRange_unit block sourceRow read (first part) (count part)
    (checked part) unit lower upper

/-- The CLI's hardware-count ceiling chunks cover every active unit. This
includes a short final chunk and empty chunks when workers exceed the units. -/
theorem ceiling_ranges_cover (units workers : Nat) (positive : 0 < workers) :
    let chunk := (units + workers - 1) / workers
    ∀ unit, unit < units →
      ∃ part : Fin workers,
        part.val * chunk ≤ unit ∧
        unit < part.val * chunk + min chunk (units - part.val * chunk) := by
  intro chunk unit bounded
  have covered : units ≤ workers * chunk := by
    dsimp only [chunk]
    have decomposition := Nat.mod_add_div (units + workers - 1) workers
    have remainder := Nat.mod_lt (units + workers - 1) positive
    omega
  have chunkPositive : 0 < chunk := by
    by_cases zero : chunk = 0
    · rw [zero, Nat.mul_zero] at covered
      omega
    · exact Nat.pos_of_ne_zero zero
  have lower : (unit / chunk) * chunk ≤ unit := Nat.div_mul_le_self unit chunk
  have partBound : unit / chunk < workers := by
    by_contra impossible
    have order : workers ≤ unit / chunk := by omega
    have product := Nat.mul_le_mul_right chunk order
    omega
  refine ⟨⟨unit / chunk, partBound⟩, lower, ?_⟩
  change unit < (unit / chunk) * chunk + min chunk (units - (unit / chunk) * chunk)
  have decomposition : unit % chunk + (unit / chunk) * chunk = unit := by
    simpa only [Nat.mul_comm] using Nat.mod_add_div unit chunk
  have remainder := Nat.mod_lt unit chunkPositive
  by_cases full : chunk ≤ units - (unit / chunk) * chunk
  · rw [Nat.min_eq_left full]
    omega
  · rw [Nat.min_eq_right (by omega)]
    omega

/-- The exact schedule used by the CLI needs only a positive worker count and
every task result. Coverage is proved here, not trusted as an IO assertion. -/
theorem checkBlock_of_workerRanges (block : MatrixProgram.Block) {columns : Nat}
    (sourceRow : Nat → Option R1CS.Row) (read : Fin columns → F)
    (workers : Nat) (positive : 0 < workers)
    (checked : ∀ part : Fin workers,
      let chunk := (checkBlockUnits block + workers - 1) / workers
      checkBlockRange block sourceRow read (part.val * chunk)
        (min chunk (checkBlockUnits block - part.val * chunk)) = true) :
    checkBlock block sourceRow read = true := by
  exact checkBlock_of_ranges block sourceRow read
    (fun part : Fin workers => part.val * ((checkBlockUnits block + workers - 1) / workers))
    (fun part : Fin workers => min ((checkBlockUnits block + workers - 1) / workers)
      (checkBlockUnits block - part.val * ((checkBlockUnits block + workers - 1) / workers)))
    (ceiling_ranges_cover (checkBlockUnits block) workers positive) checked

private theorem ofFn_get {count : Nat} (values : Fin count → F) :
    (Vector.ofFn values).get = values := by
  funext index
  change (Vector.ofFn values)[index.val] = values index
  rw [Vector.getElem_ofFn]

private theorem checkInvocation_sound (block : Poseidon.Block) {columns : Nat}
    (read : Fin columns → F) (invocation : Fin block.invocationCount)
    (checked : checkInvocation block read invocation = true) (row : Fin 86) :
    checkRow ((PiDECPoseidonNumericBlock.row? block read
      (Fin.encodeProd (invocation, row)).val).map
        (fun values => Vector.ofFn values.get)) = true := by
  cases loaded : PiDECPoseidonNumericBlock.loadInvocation? block columns invocation with
  | none =>
      simp only [checkInvocation, loaded] at checked
      cases checked
  | some interface =>
      have allRows : allFrom (fun index =>
          if bound : index < 86 then checkValues
            (((PiDECPoseidonNumericRows.stored read interface).get ⟨index, bound⟩).get)
          else false) 0 86 = true := by
        simpa only [checkInvocation, loaded] using checked
      have rowChecked := allFrom_sound _ 0 86 allRows row.val (by omega) (by
        simpa using row.isLt)
      simpa only [PiDECPoseidonNumericBlock.row?,
        PiDECPoseidonNumericBlock.loadRow?_encodeProd, loaded,
        Option.map_some, checkRow, ofFn_get, dif_pos row.isLt] using rowChecked

theorem checkBlock_sound (block : MatrixProgram.Block) {columns : Nat}
    (sourceRow : Nat → Option R1CS.Row) (read : Fin columns → F)
    (checked : checkBlock block sourceRow read = true)
    (row : Nat) (bounded : row < block.rowCount) :
    checkRow (PiDECMatrixNumericRows.blockRow? block sourceRow read row) = true := by
  cases block with
  | ordinaryTemplate block rows =>
      exact allFrom_sound _ 0 _ checked row (by omega) (by simpa using bounded)
  | mapped width projection block =>
      exact allFrom_sound _ 0 _ checked row (by omega) (by simpa using bounded)
  | ordinary block =>
      exact allFrom_sound _ 0 _ checked row (by omega) (by simpa using bounded)
  | multiplicationGrid block =>
      exact allFrom_sound _ 0 _ checked row (by omega) (by simpa using bounded)
  | phi81Product block =>
      exact allFrom_sound _ 0 _ checked row (by omega) (by simpa using bounded)
  | pin block =>
      exact allFrom_sound _ 0 _ checked row (by omega) (by simpa using bounded)
  | poseidon block =>
      let index : Fin (block.invocationCount * 86) := ⟨row, bounded⟩
      let pair := Fin.decodeProd index
      have invocationChecked := allFrom_sound _ 0 block.invocationCount checked
        pair.1.val (by omega) (by simpa using pair.1.isLt)
      have selected : checkInvocation block read pair.1 = true := by
        simpa only [dif_pos pair.1.isLt] using invocationChecked
      have rowChecked := checkInvocation_sound block read pair.1 selected pair.2
      have encoded : (Fin.encodeProd pair).val = row := by
        exact congrArg Fin.val (Fin.encodeProd_decodeProd index)
      simpa only [PiDECMatrixNumericRows.blockRow?, encoded] using rowChecked

private theorem select_checked {columns : Nat}
    (sourceRow : Nat → Option R1CS.Row) (read : Fin columns → F)
    (blocks : List MatrixProgram.Block)
    (checked : ∀ block ∈ blocks, checkBlock block sourceRow read = true)
    (row : Nat) (bounded : row < (blocks.map MatrixProgram.Block.rowCount).sum) :
    checkRow (PiDECMatrixNumericRows.row?.select sourceRow read blocks row) = true := by
  induction blocks generalizing row with
  | nil => simp at bounded
  | cons block rest inductionHypothesis =>
      by_cases selected : row < block.rowCount
      · simpa only [PiDECMatrixNumericRows.row?.select, if_pos selected] using
          checkBlock_sound block sourceRow read (checked block (by simp)) row selected
      · simp only [List.map_cons, List.sum_cons] at bounded
        have restChecked : ∀ next ∈ rest, checkBlock next sourceRow read = true := by
          intro next member
          exact checked next (List.mem_cons_of_mem block member)
        simpa only [PiDECMatrixNumericRows.row?.select, if_neg selected] using
          inductionHypothesis restChecked (row - block.rowCount) (by omega)

/-- Missing active rows reject. All fourteen returned matrix values, including
the empty fourteenth port, feed the exact production polynomial. -/
theorem rowsZero_of_blockChecks {columns : Nat}
    (program : MatrixProgram.Program) (plan : ProductionRelation.Plan columns)
    (sourceRow : Nat → Option R1CS.Row) (read : Fin columns → F)
    (exactRows : PerApplicationMatrixProgramSemantics.Exact program plan sourceRow)
    (checked : ∀ block ∈ program.blocks, checkBlock block sourceRow read = true) :
    plan.RowsZero read := by
  intro row
  have bounded : row.val < program.rowCount := by
    rw [exactRows.rowCount]
    exact row.isLt
  have rowChecked : checkRow
      (PiDECMatrixNumericRows.row? program sourceRow read row.val) = true :=
    select_checked sourceRow read program.blocks checked row.val bounded
  rw [PiDECMatrixNumericRows.row?_eq, exactRows.row? row] at rowChecked
  simp only [Option.map_some, checkRow, checkValues, decide_eq_true_eq] at rowChecked
  have values : (PiDECMatrixNumericRows.sparseValues read (plan.forms row)).get =
      fun port => (plan.portForm row port).eval read := by
    funext port
    change (Vector.ofFn _).get port = _
    rw [ofFn_get]
    exact SparseForm.evalSparse_eq_eval _ read
  rw [Plan.rowImage_toVertex, ← values]
  exact rowChecked

theorem canonical_blocks_rowsZero
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (read : Fin (PerApplicationFixedPoint.logicalWidth application) → F)
    (checked : ∀ block ∈ (PerApplicationMatrixProgram.matrixProgram application).blocks,
      checkBlock block (fun source => (PiDECCanonicalSourceCache.stored application)[source]?)
        read = true) :
    (PerApplicationFixedPoint.structuralPlan application fits).RowsZero read := by
  have sources : (fun source => (PiDECCanonicalSourceCache.stored application)[source]?) =
      PerApplicationCanonicalPackage.sourceRow application fits :=
    funext (PiDECCanonicalSourceCache.stored_value application fits)
  apply rowsZero_of_blockChecks (PerApplicationMatrixProgram.matrixProgram application)
    (PerApplicationFixedPoint.structuralPlan application fits)
    (fun source => (PiDECCanonicalSourceCache.stored application)[source]?) read
  · rw [sources]
    exact PerApplicationCanonicalPackage.matrixProgram_exact application fits
  · exact checked

theorem checkProgram_rowsZero
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (read : Fin (PerApplicationFixedPoint.logicalWidth application) → F)
    (checked : checkProgram (PerApplicationMatrixProgram.matrixProgram application)
      (fun source => (PiDECCanonicalSourceCache.stored application)[source]?) read = true) :
    (PerApplicationFixedPoint.structuralPlan application fits).RowsZero read := by
  apply canonical_blocks_rowsZero application fits read
  simpa only [checkProgram, List.all_eq_true] using checked

/-- No padding row is scanned or omitted from the semantic conclusion. -/
theorem rowsZero_allVertices {columns : Nat} (plan : ProductionRelation.Plan columns)
    (read : Fin columns → F) (checked : plan.RowsZero read)
    (vertex : BooleanVertex Lifecycle.cubeVariables) :
    evaluatePolynomial baseOps polynomial (plan.rowImage read vertex) = 0 := by
  cases decoded : plan.rowLayout.toColumn? vertex with
  | none =>
      have images : plan.rowImage read vertex = (fun _ => (0 : F)) := by
        funext port
        dsimp only [Plan.rowImage]
        rw [decoded]
      exact (congrArg
        (fun values : Fin matrixCount → F =>
          evaluatePolynomial baseOps polynomial values) images).trans polynomial_zeroImages
  | some row =>
      have images : plan.rowImage read vertex =
          plan.rowImage read (plan.rowLayout.toVertex row) := by
        funext port
        simp only [Plan.rowImage, decoded, plan.rowLayout.toColumn_toVertex]
      rw [images]
      exact checked row

def field (code : UInt8) : F :=
  if code == 1 then 1 else if code == 255 then -1 else 0

theorem field_bounded (code : UInt8) : centeredMagnitude (field code) < 2 := by
  unfold field
  split_ifs <;> decide

def carrierRead (bytes : ByteArray) : Phi81Relation.Assignment PiCCSSourceImages.shape :=
  fun column => field (bytes.get! column.val)

def logicalRead (bytes : ByteArray) : Fin PiCCSSourceImages.logicalWidth → F :=
  PiCCSSourceImages.plainRead (carrierRead bytes)

private theorem completeAssignment_embed
    (application : Lifecycle.Stage1.Application.Program)
    (raw : PerApplicationCanonicalAssignment.RawValues application)
    (column : Fin (PerApplicationFixedPoint.logicalWidth application)) :
    raw.completeAssignment (Phi81CarrierLayout.embedLogical column) = raw.assignment column := by
  dsimp only [PerApplicationCanonicalAssignment.RawValues.completeAssignment,
    Phi81CarrierLayout.embedLogical]
  exact dif_pos column.isLt

/-- Assignment transport supplies this exact value equality. It is not a
file-hash premise or a claim that file parsing has been proved in Lean. -/
theorem checked_raw (bytes : ByteArray)
    (raw : PerApplicationCanonicalAssignment.RawValues Poseidon2HashChainV1Package.application)
    (custody : carrierRead bytes = raw.completeAssignment)
    (checked : checkProgram
      (PerApplicationMatrixProgram.matrixProgram Poseidon2HashChainV1Package.application)
      (fun source => (PiDECCanonicalSourceCache.stored
        Poseidon2HashChainV1Package.application)[source]?) (logicalRead bytes) = true) :
    (PerApplicationFixedPoint.structuralPlan Poseidon2HashChainV1Package.application
      Poseidon2HashChainV1Package.fits).RowsZero raw.assignment ∧
      ∀ column, centeredMagnitude (raw.completeAssignment column) < 2 := by
  have reads : logicalRead bytes = raw.assignment := by
    funext column
    change carrierRead bytes (Phi81CarrierLayout.embedLogical column) = _
    rw [custody]
    exact completeAssignment_embed _ raw column
  constructor
  · rw [← reads]
    exact checkProgram_rowsZero _ Poseidon2HashChainV1Package.fits _ checked
  · intro column
    rw [← custody]
    exact field_bounded (bytes.get! column.val)

end NightstreamFPrime.Export.Stage1.FreshRowsCheck
