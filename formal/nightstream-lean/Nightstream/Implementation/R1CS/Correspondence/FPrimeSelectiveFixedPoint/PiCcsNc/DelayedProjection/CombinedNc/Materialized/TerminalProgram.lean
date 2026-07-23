import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Semantics

/-!
Exact straight-line program of the fixed production block-by-lane combined-NC
terminal.

Owns: the literal allocation and row order of the 19-block/6-lane terminal,
including the 64-entry lane table for each of fifteen outputs, the ordinary
cubic mix, the delayed radix-two running suffix, and the final equality with
the SumCheck claim.

Does not own: generated-row equality, padding truth, transcript scheduling,
raw-child authority, parent continuity, commitment binding, costs, or row
removal.

The terminal identity allocates one scalar column per source row.  Its 6,593
fresh columns therefore end at the generated `terminalRhsColumns`; this fixes
the first fresh column without trusting a stage label.  The two following
rows are assertions and allocate no columns.

Assurance tier: model-level until an exact generated-row certificate identifies
this program with the physical source interval.

Emits constraints: none; this module states the semantics of the existing terminal program.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.terminal_program` | Define the exact straight-line terminal formula and final equality checks. | computed semantics |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.TerminalProgram

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Semantics

local instance : Std.Associative (fun (left right : F) => left + right) := ⟨fadd_assoc⟩
local instance : Std.Commutative (fun (left right : F) => left + right) := ⟨fadd_comm⟩

abbrev rawBoundary : RawBoundaryMap := Metadata.boundary

def terminalIdentityRowCount : Nat :=
  rawBoundary.terminalIdentityRows.stop - rawBoundary.terminalIdentityRows.start

def terminalFinalEqualityRowCount : Nat :=
  rawBoundary.terminalFinalEqualityRows.stop -
    rawBoundary.terminalFinalEqualityRows.start

def terminalRhsColumns : KColumns :=
  rawKColumnsToColumns rawBoundary.terminalRhsColumns

def finalSumColumns : KColumns :=
  rawKColumnsToColumns rawBoundary.finalSumColumns

/-- The exact first fresh terminal column: `4084564 - (6593 - 2)`. -/
def firstAllocatedColumn : Nat :=
  terminalRhsColumns.c0 - (terminalIdentityRowCount - 2)

def gammaColumns : KColumns :=
  rawKColumnsToColumns rawBoundary.gammaColumns

def betaLaneColumns : List KColumns :=
  rawBoundary.betaLaneColumns.map rawKColumnsToColumns

def betaBlockColumns : List KColumns :=
  rawBoundary.betaBlockColumns.map rawKColumnsToColumns

def producerBetaColumns : KColumns :=
  rawKColumnsToColumns rawBoundary.producerBetaColumns

def batchWeightColumns : KColumns :=
  rawKColumnsToColumns rawBoundary.batchWeightColumns

def pendingOldBlockColumns : List KColumns :=
  rawBoundary.pendingOldBlockColumns.map rawKColumnsToColumns

def outputYZcolColumns : List (List KColumns) :=
  rawBoundary.outputYZcolColumns.map fun output =>
    output.map rawKColumnsToColumns

def blockPointColumns : List KColumns :=
  rawBoundary.blockPointColumns.map rawKColumnsToColumns

def lanePointColumns : List KColumns :=
  rawBoundary.lanePointColumns.map rawKColumnsToColumns

/-! ## Exact primitive traces -/

/-- One `alloc_klc`: two adjacent definitions and no multiplication. -/
structure KLinearTrace where
  input : KTerms
  output : KColumns
deriving DecidableEq, Repr

def KLinearTrace.at (base : Nat) (input : KTerms) : KLinearTrace :=
  { input, output := ⟨base, base + 1⟩ }

def KLinearTrace.definitions (trace : KLinearTrace) : List Definition :=
  [⟨trace.output.c0, .linear trace.input.c0⟩,
   ⟨trace.output.c1, .linear trace.input.c1⟩]

def KLinearTrace.next (trace : KLinearTrace) : Nat :=
  trace.output.c1 + 1

def oneTerms : KTerms := ⟨[(0, 1)], []⟩

def baseTwoTerms : KTerms := ⟨[(0, 2)], []⟩

def columnsTerms (columns : KColumns) : KTerms :=
  KTerms.ofColumns columns

def addTerms (left right : KColumns) : KTerms :=
  ⟨[(left.c0, 1), (right.c0, 1)],
   [(left.c1, 1), (right.c1, 1)]⟩

def negateTerms (value : KColumns) : KTerms :=
  ⟨[(value.c0, goldilocksP - 1)],
   [(value.c1, goldilocksP - 1)]⟩

def subtractTerms (left right : KColumns) : KTerms :=
  ⟨[(left.c0, 1), (right.c0, goldilocksP - 1)],
   [(left.c1, 1), (right.c1, goldilocksP - 1)]⟩

def oneMinusTerms (value : KColumns) : KTerms :=
  ⟨[(0, 1), (value.c0, goldilocksP - 1)],
   [(value.c1, goldilocksP - 1)]⟩

def eqFactorTerms (product left right : KColumns) : KTerms :=
  ⟨[(product.c0, 2), (0, 1),
      (left.c0, goldilocksP - 1), (right.c0, goldilocksP - 1)],
   [(product.c1, 2),
      (left.c1, goldilocksP - 1), (right.c1, goldilocksP - 1)]⟩

def selectorFactorTerms (selected coordinate : KColumns) : KTerms :=
  ⟨[(0, 1), (coordinate.c0, goldilocksP - 1), (selected.c0, 1)],
   [(coordinate.c1, goldilocksP - 1), (selected.c1, 1)]⟩

/-- One exact five-row Karatsuba multiplication at the current frontier. -/
def mulAt (base : Nat) (left right : KTerms) : KMulTrace where
  left
  right
  sumLeft := left.c0 ++ left.c1
  sumRight := right.c0 ++ right.c1
  productC0 := base
  productC1 := base + 1
  productSum := base + 2
  output := ⟨base + 3, base + 4⟩

def mulColumnsAt (base : Nat) (left right : KColumns) : KMulTrace :=
  mulAt base (columnsTerms left) (columnsTerms right)

def mulNext (trace : KMulTrace) : Nat :=
  trace.output.c1 + 1

private theorem mulAt_layout (base : Nat) (left right : KTerms) :
    (mulAt base left right).SumLayoutValid := by
  simp [mulAt, KMulTrace.SumLayoutValid]

/-- Exact extension-field meaning of one two-row linear allocation. -/
theorem KLinearTrace.sound (trace : KLinearTrace)
    (assignment : Nat → Nat)
    (definitionsHold : DefinitionsHold assignment trace.definitions) :
    trace.output.value assignment = trace.input.value assignment := by
  have low := definitionsHold
    ⟨trace.output.c0, .linear trace.input.c0⟩ (by
      simp [KLinearTrace.definitions])
  have high := definitionsHold
    ⟨trace.output.c1, .linear trace.input.c1⟩ (by
      simp [KLinearTrace.definitions])
  simp only [KColumns.value, KTerms.value, K.mk.injEq]
  constructor
  · apply Fin.ext
    simpa [Definition.Holds, Rhs.eval, baseAt, residue] using
      congrArg (fun value => value % goldilocksP) low
  · apply Fin.ext
    simpa [Definition.Holds, Rhs.eval, baseAt, residue] using
      congrArg (fun value => value % goldilocksP) high

/-- Exact extension-field meaning of one five-row multiplication. -/
theorem mulAt_sound (base : Nat) (left right : KTerms)
    (assignment : Nat → Nat)
    (definitionsHold : DefinitionsHold assignment
      (mulAt base left right).definitions) :
    (mulAt base left right).output.value assignment =
      K.mul (left.value assignment) (right.value assignment) := by
  exact (mulAt base left right).sound assignment
    (mulAt_layout base left right) definitionsHold

theorem mulColumnsAt_sound (base : Nat) (left right : KColumns)
    (assignment : Nat → Nat)
    (definitionsHold : DefinitionsHold assignment
      (mulColumnsAt base left right).definitions) :
    (mulColumnsAt base left right).output.value assignment =
      K.mul (left.value assignment) (right.value assignment) := by
  simpa [mulColumnsAt, columnsTerms, KTerms.ofColumns_value] using
    mulAt_sound base (columnsTerms left) (columnsTerms right) assignment
      definitionsHold

/-! ## Reusable ordered blocks -/

def pairMulTracesFrom : Nat → List KColumns → List KColumns →
    List KMulTrace
  | _, [], _ => []
  | _, _, [] => []
  | base, left :: lefts, right :: rights =>
      mulColumnsAt base left right ::
        pairMulTracesFrom (base + 5) lefts rights

def constantRightMulTracesFrom
    (base : Nat) (left : List KColumns) (right : KTerms) : List KMulTrace :=
  match left with
  | [] => []
  | head :: tail =>
      mulAt base (columnsTerms head) right ::
        constantRightMulTracesFrom (base + 5) tail right

def tracesDefinitions (traces : List KMulTrace) : List Definition :=
  traces.flatMap KMulTrace.definitions

def tracesOutputs (traces : List KMulTrace) : List KColumns :=
  traces.map KMulTrace.output

def sumK (values : List K) : K :=
  values.foldr K.add K.zero

def dotValue (left right : List KColumns) (assignment : Nat → Nat) : K :=
  sumK ((left.zip right).map fun pair =>
    K.mul (pair.1.value assignment) (pair.2.value assignment))

private theorem termsValue_columns (assignment : Nat → Nat)
    (columns : List Nat) :
    residue (lcEval assignment (columns.map fun column => (column, 1))) =
      columns.foldr (fun column suffix =>
        baseAt assignment column + suffix) 0 := by
  induction columns with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      apply Fin.ext
      simp only [List.map_cons, List.foldr_cons, Fin.val_add, residue]
      rw [Program.lcEval_eq_raw_mod]
      simp only [Program.rawLcEval, Nat.one_mul, Nat.mod_mod]
      have valueHypothesis := congrArg Fin.val inductionHypothesis
      simp only [residue] at valueHypothesis
      rw [Program.lcEval_eq_raw_mod] at valueHypothesis
      simp only [Nat.mod_mod] at valueHypothesis
      rw [← valueHypothesis]
      simp only [baseAt, residue]
      rw [← Nat.add_mod]

private theorem productDefinitionValue (assignment : Nat → Nat)
    (output : Nat) (left right : List (Nat × Nat))
    (holds : assignment output =
      lcEval assignment left * lcEval assignment right % goldilocksP) :
    baseAt assignment output = residue (lcEval assignment left) *
      residue (lcEval assignment right) := by
  apply Fin.ext
  simp only [baseAt, residue, Fin.val_mul]
  rw [holds]
  simp [Nat.mul_mod]

private theorem karatsubaProductSumTerminal (a0 a1 b0 b1 : F) :
    (a0 * b1 + a1 * b0) +
        (a0 * b0 + 7 * (a1 * b1)) +
        residue (goldilocksP - 6) * (a1 * b1) =
      (a0 + a1) * (b0 + b1) := by
  let rawLeft :=
    (a0.val * b1.val + a1.val * b0.val) +
      (a0.val * b0.val + 7 * (a1.val * b1.val)) +
      (goldilocksP - 6) * (a1.val * b1.val)
  let rawRight := (a0.val + a1.val) * (b0.val + b1.val)
  have rawEquality : rawLeft = rawRight +
      goldilocksP * (a1.val * b1.val) := by
    dsimp [rawLeft, rawRight]
    simp only [Nat.add_mul, Nat.mul_add]
    unfold goldilocksP
    omega
  apply Fin.ext
  simp only [Fin.val_add, Fin.val_mul, residue]
  have modularEquality : rawLeft % goldilocksP =
      rawRight % goldilocksP := by
    rw [rawEquality, Nat.add_mul_mod_self_left]
  dsimp [rawLeft, rawRight] at modularEquality
  simpa only [Nat.add_mod, Nat.mul_mod, Nat.mod_mod] using modularEquality

/-- Scalar consequences of one five-row multiplication used by `DotTrace`. -/
structure KMulTrace.Components (trace : KMulTrace)
    (left right : KColumns) (assignment : Nat → Nat) : Prop where
  productC1 : baseAt assignment trace.productC1 = baseAt assignment left.c1 *
    baseAt assignment right.c1
  productSum : baseAt assignment trace.productSum =
    (baseAt assignment left.c0 + baseAt assignment left.c1) *
      (baseAt assignment right.c0 + baseAt assignment right.c1)
  terminal :
    baseAt assignment trace.output.c1 +
        baseAt assignment trace.output.c0 +
        residue (goldilocksP - 6) *
          baseAt assignment trace.productC1 =
      baseAt assignment trace.productSum

/-- Component semantics of the exact `mulColumnsAt` program. -/
theorem mulColumnsAt_components (base : Nat) (left right : KColumns)
    (assignment : Nat → Nat)
    (definitionsHold : DefinitionsHold assignment
      (mulColumnsAt base left right).definitions) :
    KMulTrace.Components (mulColumnsAt base left right) left right
      assignment := by
  let trace := mulColumnsAt base left right
  have productC1Holds : assignment trace.productC1 =
      lcEval assignment trace.left.c1 * lcEval assignment trace.right.c1 %
        goldilocksP := by
    simpa [Definition.Holds, Rhs.eval] using
      definitionsHold
        ⟨trace.productC1, .product trace.left.c1 trace.right.c1⟩
        (by simp [trace, KMulTrace.definitions])
  have productSumHolds : assignment trace.productSum =
      lcEval assignment trace.sumLeft * lcEval assignment trace.sumRight %
        goldilocksP := by
    simpa [Definition.Holds, Rhs.eval] using
      definitionsHold
        ⟨trace.productSum, .product trace.sumLeft trace.sumRight⟩
        (by simp [trace, KMulTrace.definitions])
  have productC1Internal := productDefinitionValue assignment
    trace.productC1 trace.left.c1 trace.right.c1 productC1Holds
  have productSumInternal := productDefinitionValue assignment
    trace.productSum trace.sumLeft trace.sumRight productSumHolds
  have leftC1 : residue (lcEval assignment trace.left.c1) =
      baseAt assignment left.c1 := by
    simpa [trace, mulColumnsAt, columnsTerms, KTerms.value, KColumns.value] using
      congrArg K.c1 (KTerms.ofColumns_value left assignment)
  have rightC1 : residue (lcEval assignment trace.right.c1) =
      baseAt assignment right.c1 := by
    simpa [trace, mulColumnsAt, columnsTerms, KTerms.value, KColumns.value] using
      congrArg K.c1 (KTerms.ofColumns_value right assignment)
  have leftSum : residue (lcEval assignment trace.sumLeft) =
      baseAt assignment left.c0 + baseAt assignment left.c1 := by
    simpa [trace, mulColumnsAt, mulAt, columnsTerms, KTerms.ofColumns] using
      termsValue_columns assignment [left.c0, left.c1]
  have rightSum : residue (lcEval assignment trace.sumRight) =
      baseAt assignment right.c0 + baseAt assignment right.c1 := by
    simpa [trace, mulColumnsAt, mulAt, columnsTerms, KTerms.ofColumns] using
      termsValue_columns assignment [right.c0, right.c1]
  have productC1 : baseAt assignment trace.productC1 =
      baseAt assignment left.c1 * baseAt assignment right.c1 := by
    simpa [leftC1, rightC1] using productC1Internal
  have productSum : baseAt assignment trace.productSum =
      (baseAt assignment left.c0 + baseAt assignment left.c1) *
        (baseAt assignment right.c0 + baseAt assignment right.c1) := by
    simpa [leftSum, rightSum] using productSumInternal
  have outputValue :=
    mulColumnsAt_sound base left right assignment definitionsHold
  have outputC0 := congrArg K.c0 outputValue
  have outputC1 := congrArg K.c1 outputValue
  simp only [KColumns.value, K.mul] at outputC0 outputC1
  have terminal := karatsubaProductSumTerminal
    (baseAt assignment left.c0) (baseAt assignment left.c1)
    (baseAt assignment right.c0) (baseAt assignment right.c1)
  rw [← outputC1, ← outputC0, ← productC1, ← productSum] at terminal
  exact ⟨productC1, productSum, terminal⟩

private theorem foldKValues (values : List K) :
    values.foldr K.add K.zero =
      ⟨(values.map K.c0).foldr (fun left right => left + right) 0,
       (values.map K.c1).foldr (fun left right => left + right) 0⟩ := by
  induction values with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.foldr_cons, List.map_cons]
      rw [inductionHypothesis]
      rfl

private theorem sumColumnsTerms_value
    (columns : List KColumns) (assignment : Nat → Nat) :
    (KTerms.mk
      (columns.map fun value => (value.c0, 1))
      (columns.map fun value => (value.c1, 1))).value assignment =
      sumK (columns.map fun value => value.value assignment) := by
  unfold sumK
  rw [foldKValues]
  simp only [KTerms.value, KColumns.value, List.map_map,
    Function.comp_apply, K.mk.injEq]
  constructor
  · simpa only [List.map_map, Function.comp_apply, List.foldr_map,
      KColumns.value] using
      termsValue_columns assignment (columns.map KColumns.c0)
  · simpa only [List.map_map, Function.comp_apply, List.foldr_map,
      KColumns.value] using
      termsValue_columns assignment (columns.map KColumns.c1)

theorem addTerms_value (left right : KColumns)
    (assignment : Nat → Nat) :
    (addTerms left right).value assignment =
      K.add (left.value assignment) (right.value assignment) := by
  simpa [addTerms, sumK, K.add, K.zero] using
    sumColumnsTerms_value [left, right] assignment

private theorem pairMulTraces_sound
    (assignment : Nat → Nat) :
    ∀ (base : Nat) (left right : List KColumns),
      DefinitionsHold assignment
        (tracesDefinitions (pairMulTracesFrom base left right)) →
      (tracesOutputs (pairMulTracesFrom base left right)).map
          (fun output => output.value assignment) =
        (left.zip right).map fun pair =>
          K.mul (pair.1.value assignment) (pair.2.value assignment) := by
  intro base left
  induction left generalizing base with
  | nil => intro right _; rfl
  | cons leftHead leftTail inductionHypothesis =>
      intro right holds
      cases right with
      | nil => rfl
      | cons rightHead rightTail =>
          let multiplication := mulColumnsAt base leftHead rightHead
          have headHolds : DefinitionsHold assignment
              multiplication.definitions := by
            intro definition member
            apply holds definition
            change definition ∈ multiplication.definitions ++
              tracesDefinitions
                (pairMulTracesFrom (base + 5) leftTail rightTail)
            exact List.mem_append_left _ member
          have tailHolds : DefinitionsHold assignment
              (tracesDefinitions
                (pairMulTracesFrom (base + 5) leftTail rightTail)) := by
            intro definition member
            apply holds definition
            change definition ∈ multiplication.definitions ++
              tracesDefinitions
                (pairMulTracesFrom (base + 5) leftTail rightTail)
            exact List.mem_append_right multiplication.definitions member
          have head := mulColumnsAt_sound base leftHead rightHead assignment
            headHolds
          have tail := inductionHypothesis (base + 5) rightTail tailHolds
          change multiplication.output.value assignment ::
              (tracesOutputs
                (pairMulTracesFrom (base + 5) leftTail rightTail)).map
                  (fun output => output.value assignment) =
            K.mul (leftHead.value assignment) (rightHead.value assignment) ::
              (leftTail.zip rightTail).map (fun pair =>
                K.mul (pair.1.value assignment) (pair.2.value assignment))
          rw [head, tail]

private theorem constantRightMulTraces_sound
    (assignment : Nat → Nat) :
    ∀ (base : Nat) (left : List KColumns) (right : KTerms),
      DefinitionsHold assignment
        (tracesDefinitions (constantRightMulTracesFrom base left right)) →
      (tracesOutputs
          (constantRightMulTracesFrom base left right)).map
          (fun output => output.value assignment) =
        left.map fun value =>
          K.mul (value.value assignment) (right.value assignment) := by
  intro base left
  induction left generalizing base with
  | nil => intro right _; rfl
  | cons head tail inductionHypothesis =>
      intro right holds
      let multiplication := mulAt base (columnsTerms head) right
      have headHolds : DefinitionsHold assignment
          multiplication.definitions := by
        intro definition member
        apply holds definition
        change definition ∈ multiplication.definitions ++
          tracesDefinitions
            (constantRightMulTracesFrom (base + 5) tail right)
        exact List.mem_append_left _ member
      have tailHolds : DefinitionsHold assignment
          (tracesDefinitions
            (constantRightMulTracesFrom (base + 5) tail right)) := by
        intro definition member
        apply holds definition
        change definition ∈ multiplication.definitions ++
          tracesDefinitions
            (constantRightMulTracesFrom (base + 5) tail right)
        exact List.mem_append_right multiplication.definitions member
      have headValue := mulAt_sound base (columnsTerms head) right assignment
        headHolds
      have tailValues := inductionHypothesis (base + 5) right tailHolds
      change multiplication.output.value assignment ::
          (tracesOutputs
            (constantRightMulTracesFrom (base + 5) tail right)).map
              (fun output => output.value assignment) =
        K.mul (head.value assignment) (right.value assignment) ::
          tail.map (fun value =>
            K.mul (value.value assignment) (right.value assignment))
      rw [headValue, tailValues]
      simp [columnsTerms]
private def traceColumnSum (assignment : Nat → Nat) (traces : List KMulTrace)
    (column : KMulTrace → Nat) : F :=
  (traces.map fun trace => baseAt assignment (column trace)).foldr (fun x y => x + y) 0
private theorem pairMulTraces_componentFolds (assignment : Nat → Nat) :
    ∀ (base : Nat) (left right : List KColumns),
      DefinitionsHold assignment (tracesDefinitions
        (pairMulTracesFrom base left right)) →
      (traceColumnSum assignment (pairMulTracesFrom base left right)
          (fun multiplication => multiplication.output.c1) +
          traceColumnSum assignment (pairMulTracesFrom base left right)
          (fun multiplication => multiplication.output.c0) +
          residue (goldilocksP - 6) * traceColumnSum assignment
            (pairMulTracesFrom base left right) KMulTrace.productC1 =
        traceColumnSum assignment (pairMulTracesFrom base left right)
          KMulTrace.productSum) ∧
      (traceColumnSum assignment (pairMulTracesFrom base left right)
          KMulTrace.productSum =
        ((left.zip right).map fun pair =>
          (baseAt assignment pair.1.c0 + baseAt assignment pair.1.c1) *
            (baseAt assignment pair.2.c0 + baseAt assignment pair.2.c1)).foldr
          (fun x y => x + y) 0) := by
  intro base left
  induction left generalizing base with
  | nil =>
      intro right _
      constructor
      · simpa only [pairMulTracesFrom, traceColumnSum, List.map_nil,
          List.foldr_nil, Fin.zero_add] using Fin.mul_zero (residue (goldilocksP - 6))
      · rfl
  | cons leftHead leftTail inductionHypothesis =>
      intro right definitionsHold
      cases right with
      | nil =>
          constructor
          · simpa only [pairMulTracesFrom, traceColumnSum, List.map_nil,
              List.foldr_nil, Fin.zero_add] using Fin.mul_zero (residue (goldilocksP - 6))
          · rfl
      | cons rightHead rightTail =>
          let multiplication := mulColumnsAt base leftHead rightHead
          have headHolds : DefinitionsHold assignment multiplication.definitions := by
            intro definition member
            apply definitionsHold definition
            change definition ∈ multiplication.definitions ++ tracesDefinitions
              (pairMulTracesFrom (base + 5) leftTail rightTail)
            exact List.mem_append_left _ member
          have tailHolds : DefinitionsHold assignment (tracesDefinitions
              (pairMulTracesFrom (base + 5) leftTail rightTail)) := by
            intro definition member
            apply definitionsHold definition
            change definition ∈ multiplication.definitions ++ tracesDefinitions
              (pairMulTracesFrom (base + 5) leftTail rightTail)
            exact List.mem_append_right multiplication.definitions member
          have head := mulColumnsAt_components base leftHead rightHead
            assignment headHolds
          have tail := inductionHypothesis (base + 5) rightTail tailHolds
          let tailTraces := pairMulTracesFrom (base + 5) leftTail rightTail
          constructor
          · rcases tail with ⟨tailTerminal, _⟩
            change
              (baseAt assignment multiplication.output.c1 + traceColumnSum
                  assignment tailTraces (fun step => step.output.c1)) +
                (baseAt assignment multiplication.output.c0 + traceColumnSum
                  assignment tailTraces (fun step => step.output.c0)) +
                residue (goldilocksP - 6) *
                  (baseAt assignment multiplication.productC1 +
                    traceColumnSum assignment tailTraces KMulTrace.productC1) =
                baseAt assignment multiplication.productSum + traceColumnSum
                  assignment tailTraces KMulTrace.productSum
            rw [fmul_add]
            calc
              _ = (baseAt assignment multiplication.output.c1 +
                    baseAt assignment multiplication.output.c0 +
                    residue (goldilocksP - 6) *
                      baseAt assignment multiplication.productC1) +
                  (traceColumnSum assignment tailTraces (fun step => step.output.c1) +
                   traceColumnSum assignment tailTraces (fun step => step.output.c0) +
                   residue (goldilocksP - 6) *
                     traceColumnSum assignment tailTraces KMulTrace.productC1) := by ac_rfl
              _ = _ := by rw [head.terminal, tailTerminal]
          · rcases tail with ⟨_, tailProductSum⟩
            change baseAt assignment multiplication.productSum + traceColumnSum
              assignment tailTraces KMulTrace.productSum =
                (baseAt assignment leftHead.c0 + baseAt assignment leftHead.c1) *
                  (baseAt assignment rightHead.c0 + baseAt assignment rightHead.c1) +
                ((leftTail.zip rightTail).map fun pair =>
                  (baseAt assignment pair.1.c0 + baseAt assignment pair.1.c1) *
                    (baseAt assignment pair.2.c0 + baseAt assignment pair.2.c1)).foldr
                  (fun value suffix => value + suffix) 0
            rw [head.productSum, tailProductSum]

structure DotTrace where
  base : Nat
  left : List KColumns
  right : List KColumns
deriving DecidableEq, Repr

def DotTrace.multiplications (trace : DotTrace) : List KMulTrace :=
  pairMulTracesFrom trace.base trace.left trace.right

def DotTrace.qSumColumn (trace : DotTrace) : Nat :=
  trace.base + 5 * trace.left.length

def DotTrace.output (trace : DotTrace) : KColumns :=
  ⟨trace.qSumColumn + 1, trace.qSumColumn + 2⟩

def DotTrace.qSumDefinition (trace : DotTrace) : Definition :=
  ⟨trace.qSumColumn,
    .linear (trace.multiplications.map fun multiplication =>
      (multiplication.productC1, 1))⟩

def DotTrace.outputTrace (trace : DotTrace) : KLinearTrace :=
  { input :=
      ⟨trace.multiplications.map fun multiplication =>
          (multiplication.output.c0, 1),
       trace.multiplications.map fun multiplication =>
          (multiplication.output.c1, 1)⟩
    output := trace.output }

def DotTrace.definitions (trace : DotTrace) : List Definition :=
  tracesDefinitions trace.multiplications ++
    trace.qSumDefinition :: trace.outputTrace.definitions

def DotTrace.next (trace : DotTrace) : Nat :=
  trace.output.c1 + 1
/-- Scalar consequences retained from the allocated five-row gadgets. -/
structure DotTrace.Components (trace : DotTrace) (assignment : Nat → Nat) : Prop where
  qSum : baseAt assignment trace.qSumColumn =
    (trace.multiplications.map fun multiplication =>
      baseAt assignment multiplication.productC1).foldr (fun x y => x + y) 0
  outputC0 : baseAt assignment trace.output.c0 =
    (trace.multiplications.map fun multiplication =>
      baseAt assignment multiplication.output.c0).foldr (fun x y => x + y) 0
  productSumTerminal :
    baseAt assignment trace.output.c1 + baseAt assignment trace.output.c0 +
        residue (goldilocksP - 6) * baseAt assignment trace.qSumColumn =
      (trace.multiplications.map fun multiplication =>
        baseAt assignment multiplication.productSum).foldr (fun x y => x + y) 0
/-- The dot program implies all three combined-NC scalar equations. -/
theorem DotTrace.components_sound (trace : DotTrace)
    (assignment : Nat → Nat)
    (definitionsHold : DefinitionsHold assignment trace.definitions) :
    trace.Components assignment := by
  have multiplicationHolds : DefinitionsHold assignment (tracesDefinitions
      trace.multiplications) := by
    intro definition member
    apply definitionsHold definition
    simp [DotTrace.definitions, member]
  have qDefinitionHolds : trace.qSumDefinition.Holds assignment := by
    apply definitionsHold trace.qSumDefinition
    unfold DotTrace.definitions
    exact List.mem_append_right _ (List.mem_cons_self)
  have qRaw : assignment trace.qSumColumn = lcEval assignment
      (trace.multiplications.map fun multiplication =>
        (multiplication.productC1, 1)) := by
    simpa [DotTrace.qSumDefinition, Definition.Holds, Rhs.eval] using
      qDefinitionHolds
  have qResidue : baseAt assignment trace.qSumColumn = residue (lcEval assignment
      (trace.multiplications.map fun multiplication =>
          (multiplication.productC1, 1))) := by
    apply Fin.ext
    simpa [baseAt, residue] using
      congrArg (fun value => value % goldilocksP) qRaw
  have qTerms : residue (lcEval assignment (trace.multiplications.map
      fun multiplication => (multiplication.productC1, 1))) =
      (trace.multiplications.map fun multiplication =>
        baseAt assignment multiplication.productC1).foldr (fun x y => x + y) 0 := by
    simpa only [List.map_map, Function.comp_apply, List.foldr_map] using
      termsValue_columns assignment (trace.multiplications.map KMulTrace.productC1)
  have outputHolds : DefinitionsHold assignment trace.outputTrace.definitions := by
    intro definition member
    apply definitionsHold definition
    simp [DotTrace.definitions, member]
  have output : trace.output.value assignment = trace.outputTrace.input.value
      assignment := by
    simpa [DotTrace.outputTrace] using trace.outputTrace.sound assignment outputHolds
  have summed : trace.outputTrace.input.value assignment =
      sumK (trace.multiplications.map fun multiplication => multiplication.output.value
        assignment) := by
    simpa [DotTrace.outputTrace] using sumColumnsTerms_value
      (trace.multiplications.map KMulTrace.output) assignment
  have outputFolded : trace.output.value assignment =
      ⟨(trace.multiplications.map fun multiplication =>
          baseAt assignment multiplication.output.c0).foldr (fun x y => x + y) 0,
      (trace.multiplications.map fun multiplication =>
          baseAt assignment multiplication.output.c1).foldr (fun x y => x + y) 0⟩ := by
    rw [output, summed]
    unfold sumK
    simpa only [List.map_map, Function.comp_def, KColumns.value] using
      foldKValues (trace.multiplications.map fun m => m.output.value assignment)
  have outputC0 : baseAt assignment trace.output.c0 =
      (trace.multiplications.map fun multiplication =>
        baseAt assignment multiplication.output.c0).foldr (fun x y => x + y) 0 := by
    simpa only [KColumns.value] using congrArg K.c0 outputFolded
  have outputC1 : baseAt assignment trace.output.c1 =
      (trace.multiplications.map fun multiplication =>
        baseAt assignment multiplication.output.c1).foldr (fun x y => x + y) 0 := by
    simpa only [KColumns.value] using congrArg K.c1 outputFolded
  have componentFolds := pairMulTraces_componentFolds assignment
    trace.base trace.left trace.right multiplicationHolds
  exact
    { qSum := qResidue.trans qTerms
      outputC0 := outputC0
      productSumTerminal := by
        rw [qResidue.trans qTerms, outputC0, outputC1]
        exact componentFolds.1 }

/-- Every dot-product block computes the ordered extension-field dot product.
The extra `qSum` definition is included in the exact row stream but is not
needed as an assumption for this semantic conclusion. -/
theorem DotTrace.sound (trace : DotTrace) (assignment : Nat → Nat)
    (definitionsHold : DefinitionsHold assignment trace.definitions) :
    trace.output.value assignment =
      dotValue trace.left trace.right assignment := by
  have multiplicationHolds : DefinitionsHold assignment
      (tracesDefinitions trace.multiplications) := by
    intro definition member
    apply definitionsHold definition
    simp [DotTrace.definitions, member]
  have outputHolds : DefinitionsHold assignment
      trace.outputTrace.definitions := by
    intro definition member
    apply definitionsHold definition
    simp [DotTrace.definitions, member]
  have output : trace.output.value assignment =
      trace.outputTrace.input.value assignment := by
    simpa [DotTrace.outputTrace] using
      trace.outputTrace.sound assignment outputHolds
  have summed : trace.outputTrace.input.value assignment =
      sumK (trace.multiplications.map fun multiplication =>
        multiplication.output.value assignment) := by
    simpa [DotTrace.outputTrace] using
      sumColumnsTerms_value
        (trace.multiplications.map KMulTrace.output) assignment
  have products : trace.multiplications.map
      (fun multiplication => multiplication.output.value assignment) =
      (trace.left.zip trace.right).map fun pair =>
        K.mul (pair.1.value assignment) (pair.2.value assignment) := by
    simpa [tracesOutputs] using
      pairMulTraces_sound assignment trace.base trace.left trace.right
        multiplicationHolds
  rw [output, summed, products]
  rfl

/-- The terminal scalar is the sum of the literal left/right coordinate
products allocated by the same ordered five-row multiplication blocks. -/
theorem DotTrace.productSum_terminal (trace : DotTrace)
    (assignment : Nat → Nat)
    (definitionsHold : DefinitionsHold assignment trace.definitions) :
    baseAt assignment trace.output.c1 +
        baseAt assignment trace.output.c0 +
        residue (goldilocksP - 6) *
          baseAt assignment trace.qSumColumn =
      ((trace.left.zip trace.right).map fun pair =>
        (baseAt assignment pair.1.c0 + baseAt assignment pair.1.c1) *
          (baseAt assignment pair.2.c0 +
            baseAt assignment pair.2.c1)).foldr
        (fun value suffix => value + suffix) 0 := by
  have components := trace.components_sound assignment definitionsHold
  have multiplicationHolds : DefinitionsHold assignment
      (tracesDefinitions trace.multiplications) := by
    intro definition member
    apply definitionsHold definition
    simp [DotTrace.definitions, member]
  have componentFolds := pairMulTraces_componentFolds assignment
    trace.base trace.left trace.right multiplicationHolds
  exact components.productSumTerminal.trans componentFolds.2

structure ChiLayer where
  base : Nat
  input : List KColumns
  coordinate : KColumns
deriving DecidableEq, Repr

def ChiLayer.zeroTraces (layer : ChiLayer) : List KMulTrace :=
  constantRightMulTracesFrom layer.base layer.input
    (oneMinusTerms layer.coordinate)

def ChiLayer.oneTraces (layer : ChiLayer) : List KMulTrace :=
  constantRightMulTracesFrom (layer.base + 5 * layer.input.length)
    layer.input (columnsTerms layer.coordinate)

def ChiLayer.output (layer : ChiLayer) : List KColumns :=
  tracesOutputs layer.zeroTraces ++ tracesOutputs layer.oneTraces

def ChiLayer.definitions (layer : ChiLayer) : List Definition :=
  tracesDefinitions layer.zeroTraces ++ tracesDefinitions layer.oneTraces

def ChiLayer.next (layer : ChiLayer) : Nat :=
  layer.base + 10 * layer.input.length

def ChiLayer.expected (layer : ChiLayer) (assignment : Nat → Nat) :
    List K :=
  (layer.input.map fun value =>
      K.mul (value.value assignment)
        ((oneMinusTerms layer.coordinate).value assignment)) ++
    (layer.input.map fun value =>
      K.mul (value.value assignment) (layer.coordinate.value assignment))

theorem ChiLayer.sound (layer : ChiLayer) (assignment : Nat → Nat)
    (definitionsHold : DefinitionsHold assignment layer.definitions) :
    layer.output.map (fun output => output.value assignment) =
      layer.expected assignment := by
  have zeroHolds : DefinitionsHold assignment
      (tracesDefinitions layer.zeroTraces) := by
    intro definition member
    apply definitionsHold definition
    simp [ChiLayer.definitions, member]
  have oneHolds : DefinitionsHold assignment
      (tracesDefinitions layer.oneTraces) := by
    intro definition member
    apply definitionsHold definition
    simp [ChiLayer.definitions, member]
  have zeroValues : layer.zeroTraces.map
      (fun multiplication => multiplication.output.value assignment) =
      layer.input.map fun value =>
        K.mul (value.value assignment)
          ((oneMinusTerms layer.coordinate).value assignment) := by
    simpa [ChiLayer.zeroTraces, tracesOutputs] using
      constantRightMulTraces_sound assignment layer.base layer.input
        (oneMinusTerms layer.coordinate) zeroHolds
  have oneValues : layer.oneTraces.map
      (fun multiplication => multiplication.output.value assignment) =
      layer.input.map fun value =>
        K.mul (value.value assignment)
          ((columnsTerms layer.coordinate).value assignment) := by
    simpa [ChiLayer.oneTraces, tracesOutputs] using
      constantRightMulTraces_sound assignment
        (layer.base + 5 * layer.input.length) layer.input
        (columnsTerms layer.coordinate) oneHolds
  calc
    layer.output.map (fun output => output.value assignment) =
        (layer.zeroTraces.map fun multiplication : KMulTrace =>
          multiplication.output.value assignment) ++
        (layer.oneTraces.map fun multiplication : KMulTrace =>
          multiplication.output.value assignment) := by
      simp only [ChiLayer.output, tracesOutputs, List.map_append,
        List.map_map, Function.comp_def]
    _ = (layer.input.map fun value =>
          K.mul (value.value assignment)
            ((oneMinusTerms layer.coordinate).value assignment)) ++
        (layer.input.map fun value =>
          K.mul (value.value assignment)
            ((columnsTerms layer.coordinate).value assignment)) :=
      by rw [zeroValues, oneValues]
    _ = layer.expected assignment := by
      simp only [ChiLayer.expected, columnsTerms, KTerms.ofColumns_value]

def chiLayersFrom : Nat → List KColumns → List KColumns → List ChiLayer
  | _, _, [] => []
  | base, current, coordinate :: coordinates =>
      let layer : ChiLayer := { base, input := current, coordinate }
      layer :: chiLayersFrom layer.next layer.output coordinates

def chiInitial : KLinearTrace :=
  KLinearTrace.at firstAllocatedColumn oneTerms

def chiLayers : List ChiLayer :=
  chiLayersFrom chiInitial.next [chiInitial.output] lanePointColumns

def chiColumns : List KColumns :=
  match chiLayers.reverse with
  | [] => [chiInitial.output]
  | layer :: _ => layer.output

def chiDefinitions : List Definition :=
  chiInitial.definitions ++ chiLayers.flatMap ChiLayer.definitions

def chiNext : Nat :=
  match chiLayers.reverse with
  | [] => chiInitial.next
  | layer :: _ => layer.next

structure OutputTrace where
  base : Nat
  values : List KColumns
  chi : List KColumns
deriving DecidableEq, Repr

def OutputTrace.evaluation (trace : OutputTrace) : DotTrace :=
  { base := trace.base, left := trace.values, right := trace.chi }

def OutputTrace.square (trace : OutputTrace) : KMulTrace :=
  let value := trace.evaluation.output
  mulColumnsAt trace.evaluation.next value value

def OutputTrace.cube (trace : OutputTrace) : KMulTrace :=
  mulColumnsAt (mulNext trace.square) trace.square.output trace.evaluation.output

def OutputTrace.residual (trace : OutputTrace) : KLinearTrace :=
  KLinearTrace.at (mulNext trace.cube)
    (subtractTerms trace.cube.output trace.evaluation.output)

def OutputTrace.definitions (trace : OutputTrace) : List Definition :=
  trace.evaluation.definitions ++ trace.square.definitions ++
    trace.cube.definitions ++ trace.residual.definitions

def OutputTrace.next (trace : OutputTrace) : Nat :=
  trace.residual.next

structure OutputTrace.Computed (trace : OutputTrace)
    (assignment : Nat → Nat) : Prop where
  evaluation : trace.evaluation.output.value assignment =
    dotValue trace.values trace.chi assignment
  square : trace.square.output.value assignment =
    K.mul (trace.evaluation.output.value assignment)
      (trace.evaluation.output.value assignment)
  cube : trace.cube.output.value assignment =
    K.mul (trace.square.output.value assignment)
      (trace.evaluation.output.value assignment)
  residual : trace.residual.output.value assignment =
    (subtractTerms trace.cube.output trace.evaluation.output).value assignment

theorem OutputTrace.sound (trace : OutputTrace) (assignment : Nat → Nat)
    (definitionsHold : DefinitionsHold assignment trace.definitions) :
    trace.Computed assignment := by
  have evaluationHolds : DefinitionsHold assignment
      trace.evaluation.definitions := by
    intro definition member
    apply definitionsHold definition
    simp [OutputTrace.definitions, member]
  have squareHolds : DefinitionsHold assignment trace.square.definitions := by
    intro definition member
    apply definitionsHold definition
    simp [OutputTrace.definitions, member]
  have cubeHolds : DefinitionsHold assignment trace.cube.definitions := by
    intro definition member
    apply definitionsHold definition
    simp [OutputTrace.definitions, member]
  have residualHolds : DefinitionsHold assignment
      trace.residual.definitions := by
    intro definition member
    apply definitionsHold definition
    simp [OutputTrace.definitions, member]
  exact
    { evaluation := trace.evaluation.sound assignment evaluationHolds
      square := mulColumnsAt_sound trace.evaluation.next
        trace.evaluation.output trace.evaluation.output assignment squareHolds
      cube := mulColumnsAt_sound (mulNext trace.square) trace.square.output
        trace.evaluation.output assignment cubeHolds
      residual := trace.residual.sound assignment residualHolds }

def outputTracesFrom : Nat → List (List KColumns) → List KColumns →
    List OutputTrace
  | _, [], _ => []
  | base, values :: remaining, chi =>
      let trace : OutputTrace := { base, values, chi }
      trace :: outputTracesFrom trace.next remaining chi

private theorem outputTracesFrom_length
    (base : Nat) (outputs : List (List KColumns)) (chi : List KColumns) :
    (outputTracesFrom base outputs chi).length = outputs.length := by
  induction outputs generalizing base with
  | nil => rfl
  | cons values remaining inductionHypothesis =>
      simp only [outputTracesFrom, List.length_cons]
      rw [inductionHypothesis]

def outputTraces : List OutputTrace :=
  outputTracesFrom chiNext outputYZcolColumns chiColumns

theorem outputYZcolColumns_length : outputYZcolColumns.length = outputCount := by
  have valid := BoundaryArtifact.boundary_valid
  rcases valid with
    ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
      outputLength, _⟩
  simpa [outputYZcolColumns] using outputLength

theorem outputTraces_length : outputTraces.length = outputCount := by
  rw [outputTraces, outputTracesFrom_length, outputYZcolColumns_length]

def outputDefinitions : List Definition :=
  outputTraces.flatMap OutputTrace.definitions

def outputNext : Nat :=
  match outputTraces.reverse with
  | [] => chiNext
  | trace :: _ => trace.next

def outputEvaluations : List KColumns :=
  outputTraces.map fun trace => trace.evaluation.output

def outputResiduals : List KColumns :=
  outputTraces.map fun trace => trace.residual.output

/-- The fixed fixture has one fresh application relation; the remaining
fourteen outputs are the authoritative running suffix. -/
def freshOutputCount : Nat := Metadata.applicationRows

theorem freshOutputCount_eq_one : freshOutputCount = 1 := rfl

structure PowerTrace where
  base : Nat
  point : KColumns
  count : Nat
deriving DecidableEq, Repr

def powerMultiplicationsFrom : Nat → KColumns → KColumns → Nat →
    List KMulTrace
  | _, _, _, 0 => []
  | base, current, point, count + 1 =>
      let multiplication := mulColumnsAt base current point
      multiplication :: powerMultiplicationsFrom (mulNext multiplication)
        multiplication.output point count

private theorem powerMultiplicationsFrom_length
    (base : Nat) (current point : KColumns) (count : Nat) :
    (powerMultiplicationsFrom base current point count).length = count := by
  induction count generalizing base current with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp [powerMultiplicationsFrom, inductionHypothesis]

def PowerTrace.powers (trace : PowerTrace) : List KColumns :=
  if trace.count = 0 then [] else
    let initial : KColumns := ⟨trace.base, trace.base + 1⟩
    initial :: tracesOutputs
      (powerMultiplicationsFrom (trace.base + 2) initial trace.point
        (trace.count - 1))

def PowerTrace.ladder (trace : PowerTrace) : LadderTrace :=
  let initial : KColumns := ⟨trace.base, trace.base + 1⟩
  { beta := trace.point
    powers := trace.powers
    multiplications :=
      powerMultiplicationsFrom (trace.base + 2) initial trace.point
        (trace.count - 1) }

def PowerTrace.definitions (trace : PowerTrace) : List Definition :=
  trace.ladder.definitions

def PowerTrace.next (trace : PowerTrace) : Nat :=
  trace.base + 2 + 5 * (trace.count - 1)

private theorem powerMultiplications_linked
    (base : Nat) (current point : KColumns) (count : Nat) :
    LadderLinked point
      (current :: tracesOutputs
        (powerMultiplicationsFrom base current point count))
      (powerMultiplicationsFrom base current point count) := by
  induction count generalizing base current with
  | zero => simp [powerMultiplicationsFrom, tracesOutputs, LadderLinked]
  | succ count inductionHypothesis =>
      simp [powerMultiplicationsFrom, tracesOutputs, LadderLinked,
        mulColumnsAt, mulAt, columnsTerms, KMulTrace.SumLayoutValid]
      exact inductionHypothesis
        (mulNext (mulColumnsAt base current point))
        (mulColumnsAt base current point).output

theorem PowerTrace.layout (trace : PowerTrace)
    (countPositive : 0 < trace.count) : trace.ladder.LayoutValid := by
  have countNonzero : trace.count ≠ 0 := Nat.ne_of_gt countPositive
  unfold PowerTrace.ladder LadderTrace.LayoutValid
  simp only [PowerTrace.powers, countNonzero, if_false]
  exact powerMultiplications_linked (trace.base + 2)
    ⟨trace.base, trace.base + 1⟩ trace.point (trace.count - 1)

theorem PowerTrace.sound (trace : PowerTrace) (assignment : Nat → Nat)
    (constantOne : assignment 0 = 1)
    (countPositive : 0 < trace.count)
    (layout : trace.ladder.LayoutValid)
    (definitionsHold : DefinitionsHold assignment trace.definitions) :
    trace.powers.map (fun power => power.value assignment) =
      K.powersFrom (trace.point.value assignment) K.one trace.count := by
  have computed := trace.ladder.sound assignment constantOne layout
    definitionsHold
  have length : trace.powers.length = trace.count := by
    unfold PowerTrace.powers
    rw [if_neg (Nat.ne_of_gt countPositive)]
    simp only [List.length_cons, tracesOutputs, List.length_map,
      powerMultiplicationsFrom_length]
    omega
  change trace.powers.map (fun power => power.value assignment) =
      K.powersFrom (trace.point.value assignment) K.one
        trace.powers.length at computed
  rw [length] at computed
  exact computed

def gammaPowers : PowerTrace :=
  { base := outputNext, point := gammaColumns, count := outputCount }

def ordinarySum : DotTrace :=
  { base := gammaPowers.next
    left := gammaPowers.powers
    right := outputResiduals }

/-! ## Point equality -/

structure EqFirstTrace where
  base : Nat
  left : KColumns
  right : KColumns
deriving DecidableEq, Repr

def EqFirstTrace.product (trace : EqFirstTrace) : KMulTrace :=
  mulColumnsAt trace.base trace.left trace.right

def EqFirstTrace.factor (trace : EqFirstTrace) : KLinearTrace :=
  KLinearTrace.at (mulNext trace.product)
    (eqFactorTerms trace.product.output trace.left trace.right)

def EqFirstTrace.definitions (trace : EqFirstTrace) : List Definition :=
  trace.product.definitions ++ trace.factor.definitions

def EqFirstTrace.next (trace : EqFirstTrace) : Nat :=
  trace.factor.next

structure EqFirstTrace.Computed (trace : EqFirstTrace)
    (assignment : Nat → Nat) : Prop where
  product : trace.product.output.value assignment =
    K.mul (trace.left.value assignment) (trace.right.value assignment)
  factor : trace.factor.output.value assignment =
    (eqFactorTerms trace.product.output trace.left trace.right).value assignment

theorem EqFirstTrace.sound (trace : EqFirstTrace)
    (assignment : Nat → Nat)
    (definitionsHold : DefinitionsHold assignment trace.definitions) :
    trace.Computed assignment := by
  have productHolds : DefinitionsHold assignment trace.product.definitions := by
    intro definition member
    apply definitionsHold definition
    simp [EqFirstTrace.definitions, member]
  have factorHolds : DefinitionsHold assignment trace.factor.definitions := by
    intro definition member
    apply definitionsHold definition
    simp [EqFirstTrace.definitions, member]
  exact
    { product := mulColumnsAt_sound trace.base trace.left trace.right
        assignment productHolds
      factor := trace.factor.sound assignment factorHolds }

structure EqTailTrace where
  base : Nat
  accumulator : KColumns
  left : KColumns
  right : KColumns
deriving DecidableEq, Repr

def EqTailTrace.product (trace : EqTailTrace) : KMulTrace :=
  mulColumnsAt trace.base trace.left trace.right

def EqTailTrace.factor (trace : EqTailTrace) : KLinearTrace :=
  KLinearTrace.at (mulNext trace.product)
    (eqFactorTerms trace.product.output trace.left trace.right)

def EqTailTrace.fold (trace : EqTailTrace) : KMulTrace :=
  mulColumnsAt trace.factor.next trace.accumulator trace.factor.output

def EqTailTrace.definitions (trace : EqTailTrace) : List Definition :=
  trace.product.definitions ++ trace.factor.definitions ++
    trace.fold.definitions

def EqTailTrace.next (trace : EqTailTrace) : Nat :=
  mulNext trace.fold

structure EqTailTrace.Computed (trace : EqTailTrace)
    (assignment : Nat → Nat) : Prop where
  product : trace.product.output.value assignment =
    K.mul (trace.left.value assignment) (trace.right.value assignment)
  factor : trace.factor.output.value assignment =
    (eqFactorTerms trace.product.output trace.left trace.right).value assignment
  fold : trace.fold.output.value assignment =
    K.mul (trace.accumulator.value assignment)
      (trace.factor.output.value assignment)

theorem EqTailTrace.sound (trace : EqTailTrace)
    (assignment : Nat → Nat)
    (definitionsHold : DefinitionsHold assignment trace.definitions) :
    trace.Computed assignment := by
  have productHolds : DefinitionsHold assignment trace.product.definitions := by
    intro definition member
    apply definitionsHold definition
    simp [EqTailTrace.definitions, member]
  have factorHolds : DefinitionsHold assignment trace.factor.definitions := by
    intro definition member
    apply definitionsHold definition
    simp [EqTailTrace.definitions, member]
  have foldHolds : DefinitionsHold assignment trace.fold.definitions := by
    intro definition member
    apply definitionsHold definition
    simp [EqTailTrace.definitions, member]
  exact
    { product := mulColumnsAt_sound trace.base trace.left trace.right
        assignment productHolds
      factor := trace.factor.sound assignment factorHolds
      fold := mulColumnsAt_sound trace.factor.next trace.accumulator
        trace.factor.output assignment foldHolds }

def eqTailTracesFrom : Nat → KColumns → List KColumns →
    List KColumns → List EqTailTrace
  | _, _, [], _ => []
  | _, _, _, [] => []
  | base, accumulator, left :: lefts, right :: rights =>
      let trace : EqTailTrace := { base, accumulator, left, right }
      trace :: eqTailTracesFrom trace.next trace.fold.output lefts rights

structure EqTrace where
  base : Nat
  left : List KColumns
  right : List KColumns
deriving DecidableEq, Repr

def EqTrace.first? (trace : EqTrace) : Option EqFirstTrace :=
  match trace.left, trace.right with
  | left :: _, right :: _ => some { base := trace.base, left, right }
  | _, _ => none

def EqTrace.tail (trace : EqTrace) : List EqTailTrace :=
  match trace.left, trace.right, trace.first? with
  | _ :: lefts, _ :: rights, some first =>
      eqTailTracesFrom first.next first.factor.output lefts rights
  | _, _, _ => []

def EqTrace.output (trace : EqTrace) : KColumns :=
  match trace.tail.reverse, trace.first? with
  | tail :: _, _ => tail.fold.output
  | [], some first => first.factor.output
  | [], none => default

def EqTrace.definitions (trace : EqTrace) : List Definition :=
  match trace.first? with
  | none => []
  | some first => first.definitions ++ trace.tail.flatMap EqTailTrace.definitions

def EqTrace.next (trace : EqTrace) : Nat :=
  match trace.tail.reverse, trace.first? with
  | tail :: _, _ => tail.next
  | [], some first => first.next
  | [], none => trace.base

structure EqTrace.Computed (trace : EqTrace)
    (assignment : Nat → Nat) : Prop where
  first : ∀ first, trace.first? = some first → first.Computed assignment
  tail : ∀ step ∈ trace.tail, step.Computed assignment

theorem EqTrace.sound_of_first (trace : EqTrace) (assignment : Nat → Nat)
    (first : EqFirstTrace) (firstEq : trace.first? = some first)
    (definitionsHold : DefinitionsHold assignment trace.definitions) :
    trace.Computed assignment := by
  constructor
  · intro candidate candidateEq
    have equal : candidate = first := by
      exact Option.some.inj (candidateEq.symm.trans firstEq)
    subst candidate
    apply first.sound assignment
    intro definition member
    apply definitionsHold definition
    simp [EqTrace.definitions, firstEq, member]
  · intro step stepMember
    apply step.sound assignment
    intro definition member
    apply definitionsHold definition
    rw [EqTrace.definitions, firstEq]
    exact List.mem_append_right first.definitions
      (List.mem_flatMap.mpr ⟨step, stepMember, member⟩)

def blockEquality : EqTrace :=
  { base := ordinarySum.next
    left := blockPointColumns
    right := betaBlockColumns }

def laneEquality : EqTrace :=
  { base := blockEquality.next
    left := lanePointColumns
    right := betaLaneColumns }

def blockEqualityFirst : EqFirstTrace :=
  { base := blockEquality.base
    left := blockEquality.left.getD 0 default
    right := blockEquality.right.getD 0 default }

def laneEqualityFirst : EqFirstTrace :=
  { base := laneEquality.base
    left := laneEquality.left.getD 0 default
    right := laneEquality.right.getD 0 default }

set_option maxRecDepth 100000 in
theorem blockEquality_first :
    blockEquality.first? = some blockEqualityFirst := by
  native_decide

set_option maxRecDepth 100000 in
theorem laneEquality_first :
    laneEquality.first? = some laneEqualityFirst := by
  native_decide

def equalityProduct : KMulTrace :=
  mulColumnsAt laneEquality.next blockEquality.output laneEquality.output

def ordinaryProduct : KMulTrace :=
  mulColumnsAt (mulNext equalityProduct) equalityProduct.output ordinarySum.output

/-! ## Delayed running suffix and producer selector -/

def radixConstant : KLinearTrace :=
  KLinearTrace.at (mulNext ordinaryProduct) baseTwoTerms

def runningValues : List KColumns :=
  outputEvaluations.drop freshOutputCount

theorem outputEvaluations_length : outputEvaluations.length = outputCount := by
  simp [outputEvaluations, outputTraces_length]

theorem runningValues_length : runningValues.length = outputCount - 1 := by
  rw [runningValues, List.length_drop, outputEvaluations_length,
    freshOutputCount_eq_one]

def radixPowers : PowerTrace :=
  { base := radixConstant.next
    point := radixConstant.output
    count := runningValues.length }

def runningSum : DotTrace :=
  { base := radixPowers.next
    left := radixPowers.powers
    right := runningValues }

def oldBlockEquality : EqTrace :=
  { base := runningSum.next
    left := blockPointColumns
    right := pendingOldBlockColumns }

def oldBlockEqualityFirst : EqFirstTrace :=
  { base := oldBlockEquality.base
    left := oldBlockEquality.left.getD 0 default
    right := oldBlockEquality.right.getD 0 default }

set_option maxRecDepth 100000 in
theorem oldBlockEquality_first :
    oldBlockEquality.first? = some oldBlockEqualityFirst := by
  native_decide

structure SelectorStep where
  base : Nat
  accumulator : KColumns
  betaPower : KColumns
  coordinate : KColumns
deriving DecidableEq, Repr

def SelectorStep.selected (trace : SelectorStep) : KMulTrace :=
  mulColumnsAt trace.base trace.coordinate trace.betaPower

def SelectorStep.factor (trace : SelectorStep) : KLinearTrace :=
  KLinearTrace.at (mulNext trace.selected)
    (selectorFactorTerms trace.selected.output trace.coordinate)

def SelectorStep.fold (trace : SelectorStep) : KMulTrace :=
  mulColumnsAt trace.factor.next trace.accumulator trace.factor.output

def SelectorStep.squareBeta (trace : SelectorStep) : KMulTrace :=
  mulColumnsAt (mulNext trace.fold) trace.betaPower trace.betaPower

def SelectorStep.definitions (trace : SelectorStep) : List Definition :=
  trace.selected.definitions ++ trace.factor.definitions ++
    trace.fold.definitions ++ trace.squareBeta.definitions

def SelectorStep.next (trace : SelectorStep) : Nat :=
  mulNext trace.squareBeta

structure SelectorStep.Computed (trace : SelectorStep)
    (assignment : Nat → Nat) : Prop where
  selected : trace.selected.output.value assignment =
    K.mul (trace.coordinate.value assignment)
      (trace.betaPower.value assignment)
  factor : trace.factor.output.value assignment =
    (selectorFactorTerms trace.selected.output trace.coordinate).value
      assignment
  fold : trace.fold.output.value assignment =
    K.mul (trace.accumulator.value assignment)
      (trace.factor.output.value assignment)
  squareBeta : trace.squareBeta.output.value assignment =
    K.mul (trace.betaPower.value assignment)
      (trace.betaPower.value assignment)

theorem SelectorStep.sound (trace : SelectorStep)
    (assignment : Nat → Nat)
    (definitionsHold : DefinitionsHold assignment trace.definitions) :
    trace.Computed assignment := by
  have selectedHolds : DefinitionsHold assignment
      trace.selected.definitions := by
    intro definition member
    apply definitionsHold definition
    simp [SelectorStep.definitions, member]
  have factorHolds : DefinitionsHold assignment trace.factor.definitions := by
    intro definition member
    apply definitionsHold definition
    simp [SelectorStep.definitions, member]
  have foldHolds : DefinitionsHold assignment trace.fold.definitions := by
    intro definition member
    apply definitionsHold definition
    simp [SelectorStep.definitions, member]
  have squareHolds : DefinitionsHold assignment
      trace.squareBeta.definitions := by
    intro definition member
    apply definitionsHold definition
    simp [SelectorStep.definitions, member]
  exact
    { selected := mulColumnsAt_sound trace.base trace.coordinate
        trace.betaPower assignment selectedHolds
      factor := trace.factor.sound assignment factorHolds
      fold := mulColumnsAt_sound trace.factor.next trace.accumulator
        trace.factor.output assignment foldHolds
      squareBeta := mulColumnsAt_sound (mulNext trace.fold) trace.betaPower
        trace.betaPower assignment squareHolds }

def selectorStepsFrom : Nat → KColumns → KColumns →
    List KColumns → List SelectorStep
  | _, _, _, [] => []
  | base, accumulator, betaPower, coordinate :: coordinates =>
      let trace : SelectorStep :=
        { base, accumulator, betaPower, coordinate }
      trace :: selectorStepsFrom trace.next trace.fold.output
        trace.squareBeta.output coordinates

def selectorInitial : KLinearTrace :=
  KLinearTrace.at oldBlockEquality.next oneTerms

def selectorSteps : List SelectorStep :=
  selectorStepsFrom selectorInitial.next selectorInitial.output
    producerBetaColumns lanePointColumns

def selectorOutput : KColumns :=
  match selectorSteps.reverse with
  | [] => selectorInitial.output
  | trace :: _ => trace.fold.output

def selectorNext : Nat :=
  match selectorSteps.reverse with
  | [] => selectorInitial.next
  | trace :: _ => trace.next

def weightedOldEquality : KMulTrace :=
  mulColumnsAt selectorNext batchWeightColumns oldBlockEquality.output

def weightedSelector : KMulTrace :=
  mulColumnsAt (mulNext weightedOldEquality) weightedOldEquality.output
    selectorOutput

def delayedProduct : KMulTrace :=
  mulColumnsAt (mulNext weightedSelector) weightedSelector.output runningSum.output

def finalAddition : KLinearTrace :=
  { input := addTerms ordinaryProduct.output delayedProduct.output
    output := terminalRhsColumns }

/-! ## Whole program -/

/-- Lane interpolation, cubic residuals, gamma powers, and their dot product. -/
def laneComputationDefinitions : List Definition :=
  chiDefinitions ++ outputDefinitions ++ gammaPowers.definitions ++
    ordinarySum.definitions

/-- The two point equalities and ordinary equality-gated cubic mix. -/
def ordinaryDefinitions : List Definition :=
  blockEquality.definitions ++
    laneEquality.definitions ++ equalityProduct.definitions ++
    ordinaryProduct.definitions

/-- Delayed schedule through the selector's constant-one initializer. -/
def delayedPreSelectorDefinitions : List Definition :=
  radixConstant.definitions ++
    radixPowers.definitions ++ runningSum.definitions ++
    oldBlockEquality.definitions ++ selectorInitial.definitions

/-- Three terminal multiplications after the producer selector. -/
def delayedPostSelectorDefinitions : List Definition :=
  weightedOldEquality.definitions ++ weightedSelector.definitions ++
    delayedProduct.definitions

/-- Radix-two running suffix, old-block equality, producer selector, and
`batchWeight` multiplication. -/
def delayedDefinitions : List Definition :=
  delayedPreSelectorDefinitions ++
    selectorSteps.flatMap SelectorStep.definitions ++
    delayedPostSelectorDefinitions

theorem delayedDefinitions_eq_segments :
    delayedDefinitions = delayedPreSelectorDefinitions ++
      selectorSteps.flatMap SelectorStep.definitions ++
      delayedPostSelectorDefinitions := rfl

/-- Point-equality and delayed suffix of the terminal schedule. -/
def suffixDefinitions : List Definition :=
  ordinaryDefinitions ++ delayedDefinitions

/-- All exact Rust definitions before the final two-limb addition. -/
def prefixDefinitions : List Definition :=
  laneComputationDefinitions ++ suffixDefinitions

/-- Exact Rust definition order for `enforce_block_lane_terminal_identity`. -/
def definitions : List Definition :=
  prefixDefinitions ++ finalAddition.definitions

theorem definitions_eq_prefix_append_final :
    definitions = prefixDefinitions ++ finalAddition.definitions := rfl

theorem prefixDefinitions_eq_lane_append_suffix :
    prefixDefinitions = laneComputationDefinitions ++ suffixDefinitions := rfl

theorem suffixDefinitions_eq_ordinary_append_delayed :
    suffixDefinitions = ordinaryDefinitions ++ delayedDefinitions := rfl

def identityRows : List Row :=
  definitions.map Definition.builderRow

def finalEqualityRows : List Row :=
  [builderLinearRow finalSumColumns.c0 [(terminalRhsColumns.c0, 1)],
   builderLinearRow finalSumColumns.c1 [(terminalRhsColumns.c1, 1)]]

def rows : List Row := identityRows ++ finalEqualityRows
end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.TerminalProgram
