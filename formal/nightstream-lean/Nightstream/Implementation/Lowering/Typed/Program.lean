import Nightstream.Implementation.Lowering.Typed.Signature

/-!
Contract: artifact-independent execution and relational semantics for typed
lowering programs.

Owns:
- verifier-static literals, linear combinations, products, N-ary calls, and
  typed Boolean assertions;
- deterministic partial execution and an independently stated relation for
  every primitive;
- intrinsic base/recursive control flow whose two private arms expose one
  common joined schema;
- whole-program soundness and completeness.

Does not own: emission receipts, physical rows or columns, a concrete R1CS
encoding, Rust behavior, or caller-supplied propositions.

Runtime values come only from the program's static input schema.  `literal`
embeds verifier-owned constants in the fixed program; it is not a prover input.
Branching is a program node rather than an external Lean `if`, so the receipt
layer can account for both arms and the join.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.Typed

universe u

open TypeSystem

/-- Typed exports from a private context into one exact declared schema. -/
inductive Exports (types : TypeSystem.{u}) (context : Schema types) :
    Schema types -> Type (u + 1) where
  | nil : Exports types context []
  | cons {port : Port types} {tail : Schema types} :
      Ref types context port.kind ->
      Exports types context tail ->
      Exports types context (port :: tail)

namespace Exports

def get {types : TypeSystem.{u}} {context result : Schema types} :
    Exports types context result ->
    Schema.Values types context ->
    Schema.Values types result
  | .nil, _ => .nil
  | .cons reference tail, source =>
      .cons (reference.get source) (tail.get source)

end Exports

/-- Deterministic evaluation of an affine field expression. -/
def linearEval {types : TypeSystem.{u}} {context : Schema types}
    (constant : types.Field)
    (terms : List (types.Field × Ref types context .field))
    (values : Schema.Values types context) : types.Field :=
  terms.foldl
    (fun accumulator term =>
      types.add accumulator (types.mul term.1 (term.2.get values)))
    constant

/-- One closed typed operation.  Calls receive separately addressed operands
and return ordinary schema ports, so later calls can consume their results.
Assertions consume the type-system-owned bit carrier and introduce no
caller-selected proposition. -/
inductive Primitive (signature : Signature.{u}) :
    Schema signature.types -> Schema signature.types -> Type (u + 2) where
  | literal {context : Schema signature.types}
      (port : Port signature.types)
      (value : signature.types.Value port.kind) :
      Primitive signature context (port :: context)
  | linear {context : Schema signature.types}
      (layout : Layout)
      (constant : signature.types.Field)
      (terms :
        List (signature.types.Field ×
          Ref signature.types context .field)) :
      Primitive signature context
        ({ kind := .field, layout := layout } :: context)
  | product {context : Schema signature.types}
      (layout : Layout)
      (left right : Ref signature.types context .field) :
      Primitive signature context
        ({ kind := .field, layout := layout } :: context)
  | invoke {context : Schema signature.types}
      (call : signature.Call)
      (operands :
        Refs signature.types context (signature.callInputs call)) :
      Primitive signature context (signature.callOutputs call ++ context)
  | assertTrue {context : Schema signature.types}
      (condition : Ref signature.types context .bit) :
      Primitive signature context context

namespace Primitive

/-- Deterministic partial execution of one primitive. -/
def exec {signature : Signature.{u}} :
    {input output : Schema signature.types} ->
    Primitive signature input output ->
    Schema.Values signature.types input ->
    Option (Schema.Values signature.types output)
  | _, _, .literal _ value, source =>
      some (.cons value source)
  | _, _, .linear _ constant terms, source =>
      some (.cons (linearEval constant terms source) source)
  | _, _, .product _ left right, source =>
      some (.cons
        (signature.types.mul (left.get source) (right.get source)) source)
  | _, _, .invoke call operands, source =>
      match signature.callEval call (operands.get source) with
      | none => none
      | some outputs => some (outputs.append source)
  | _, _, .assertTrue condition, source =>
      match signature.types.bitValue (condition.get source) with
      | false => none
      | true => some source

/-- Reduce an invoked call from an exact failed call-evaluation fact.  The
constructor retains the dependent output-schema equality, so concrete callers
do not have to reconstruct its hidden transport. -/
theorem invoke_exec_of_eq_none
    {signature : Signature.{u}}
    {context : Schema signature.types}
    {call : signature.Call}
    {operands : Refs signature.types context (signature.callInputs call)}
    {source : Schema.Values signature.types context}
    (evaluated : signature.callEval call (operands.get source) = none) :
    (Primitive.invoke call operands).exec source = none := by
  simp only [Primitive.exec, evaluated]

/-- Reduce an invoked call from an exact successful call-evaluation fact. -/
theorem invoke_exec_of_eq_some
    {signature : Signature.{u}}
    {context : Schema signature.types}
    {call : signature.Call}
    {operands : Refs signature.types context (signature.callInputs call)}
    {source : Schema.Values signature.types context}
    {outputs :
      Schema.Values signature.types (signature.callOutputs call)}
    (evaluated : signature.callEval call (operands.get source) = some outputs) :
    (Primitive.invoke call operands).exec source =
      some (outputs.append source) := by
  simp only [Primitive.exec, evaluated]

/-- Exact head-assertion reduction without unfolding a complete dependent
program.  This is the common shape produced immediately after a Boolean call
and is intentionally available as a public rewrite rule. -/
@[simp] theorem assertTrue_here_exec
    {signature : Signature.{u}}
    {tail : Schema signature.types}
    (layout : Layout)
    (bit : signature.types.Value .bit)
    (values : Schema.Values signature.types tail) :
    (Primitive.assertTrue (signature := signature)
        (Ref.here { kind := .bit, layout := layout } :
          Ref signature.types
            ({ kind := .bit, layout := layout } :: tail) .bit)).exec
        (.cons bit values) =
      match signature.types.bitValue bit with
      | false => none
      | true => some (.cons bit values) :=
  rfl

/-- Artifact-independent relational semantics of one primitive.  This is
stated constructor-by-constructor rather than defined in terms of `exec`. -/
def Holds {signature : Signature.{u}} :
    {input output : Schema signature.types} ->
    Primitive signature input output ->
    Schema.Values signature.types input ->
    Schema.Values signature.types output -> Prop
  | _, _, .literal _ value, source, result =>
      result = .cons value source
  | _, _, .linear _ constant terms, source, result =>
      result = .cons (linearEval constant terms source) source
  | _, _, .product _ left right, source, result =>
      result = .cons
        (signature.types.mul (left.get source) (right.get source)) source
  | _, _, .invoke call operands, source, result =>
      ∃ outputs,
        signature.callEval call (operands.get source) = some outputs ∧
        result = outputs.append source
  | _, _, .assertTrue condition, source, result =>
      signature.types.bitValue (condition.get source) = true ∧ result = source

/-- Exact semantic characterization of executable primitive success. -/
theorem exec_eq_some_iff_holds
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    (primitive : Primitive signature input output)
    (source : Schema.Values signature.types input)
    (result : Schema.Values signature.types output) :
    primitive.exec source = some result ↔ primitive.Holds source result := by
  cases primitive with
  | literal =>
      simp [exec, Holds, eq_comm]
  | linear =>
      simp [exec, Holds, eq_comm]
  | product =>
      simp [exec, Holds, eq_comm]
  | invoke call operands =>
      cases evaluated :
        signature.callEval call (operands.get source) with
      | none =>
          simp [exec, Holds, evaluated]
      | some outputs =>
          simp [exec, Holds, evaluated, eq_comm]
  | assertTrue condition =>
      cases accepted : signature.types.bitValue (condition.get source) with
      | false =>
          simp [exec, Holds, accepted]
      | true =>
          simp [exec, Holds, accepted, eq_comm]

/-- Primitive execution is sound for the independent relation. -/
theorem sound
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    (primitive : Primitive signature input output)
    (source : Schema.Values signature.types input)
    (result : Schema.Values signature.types output)
    (executed : primitive.exec source = some result) :
    primitive.Holds source result :=
  (primitive.exec_eq_some_iff_holds source result).mp executed

/-- Every related primitive result is constructed by execution. -/
theorem complete
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    (primitive : Primitive signature input output)
    (source : Schema.Values signature.types input)
    (result : Schema.Values signature.types output)
    (holds : primitive.Holds source result) :
    primitive.exec source = some result :=
  (primitive.exec_eq_some_iff_holds source result).mpr holds

end Primitive

/-- A block starts from one static schema and exposes only its declared result
schema.  Private locals introduced by `step` disappear at `yield`.

`branch` runs one arm from the current schema, joins the selected arm into the
same `joined` schema, and prepends those exports to the unchanged input.  Its
continuation can inspect `joined ++ input`, but neither arm's private locals. -/
inductive Block (signature : Signature.{u}) :
    Schema signature.types -> Schema signature.types -> Type (u + 3) where
  | yield {context result : Schema signature.types}
      (values : Exports signature.types context result) :
      Block signature context result
  | step {input middle result : Schema signature.types}
      (primitive : Primitive signature input middle)
      (rest : Block signature middle result) :
      Block signature input result
  | branch {input joined result : Schema signature.types}
      (condition : Ref signature.types input .bit)
      (onTrue : Block signature input joined)
      (onFalse : Block signature input joined)
      (continuation : Block signature (joined ++ input) result) :
      Block signature input result

namespace Block

/-- Deterministic partial execution of an intrinsically branched block. -/
def exec {signature : Signature.{u}} :
    {input output : Schema signature.types} ->
    Block signature input output ->
    Schema.Values signature.types input ->
    Option (Schema.Values signature.types output)
  | _, _, .yield references, source =>
      some (references.get source)
  | _, _, .step primitive rest, source =>
      match primitive.exec source with
      | none => none
      | some middle => rest.exec middle
  | _, _, .branch condition onTrue onFalse continuation, source =>
      let selected :=
        match signature.types.bitValue (condition.get source) with
        | true => onTrue.exec source
        | false => onFalse.exec source
      match selected with
      | none => none
      | some joined => continuation.exec (joined.append source)

/-- Reduce one whole block step from an exact successful primitive execution.
This keeps concrete lowering proofs from unfolding a dependent continuation. -/
theorem step_exec_of_eq_some
    {signature : Signature.{u}}
    {input middle result : Schema signature.types}
    {primitive : Primitive signature input middle}
    {rest : Block signature middle result}
    {source : Schema.Values signature.types input}
    {values : Schema.Values signature.types middle}
    (executed : primitive.exec source = some values) :
    (Block.step primitive rest).exec source = rest.exec values := by
  simp only [Block.exec, executed]

/-- Reduce one whole block step from an exact failed primitive execution. -/
theorem step_exec_of_eq_none
    {signature : Signature.{u}}
    {input middle result : Schema signature.types}
    {primitive : Primitive signature input middle}
    {rest : Block signature middle result}
    {source : Schema.Values signature.types input}
    (executed : primitive.exec source = none) :
    (Block.step primitive rest).exec source = none := by
  simp only [Block.exec, executed]

/-- Select the true arm of a whole branch from an exact selector fact. -/
theorem branch_exec_of_selector_true
    {signature : Signature.{u}}
    {input joined result : Schema signature.types}
    {condition : Ref signature.types input .bit}
    {onTrue onFalse : Block signature input joined}
    {continuation : Block signature (joined ++ input) result}
    {source : Schema.Values signature.types input}
    (selected : signature.types.bitValue (condition.get source) = true) :
    (Block.branch condition onTrue onFalse continuation).exec source =
      match onTrue.exec source with
      | none => none
      | some exposed => continuation.exec (exposed.append source) := by
  simp only [Block.exec, selected]

/-- Select the false arm of a whole branch from an exact selector fact. -/
theorem branch_exec_of_selector_false
    {signature : Signature.{u}}
    {input joined result : Schema signature.types}
    {condition : Ref signature.types input .bit}
    {onTrue onFalse : Block signature input joined}
    {continuation : Block signature (joined ++ input) result}
    {source : Schema.Values signature.types input}
    (selected : signature.types.bitValue (condition.get source) = false) :
    (Block.branch condition onTrue onFalse continuation).exec source =
      match onFalse.exec source with
      | none => none
      | some exposed => continuation.exec (exposed.append source) := by
  simp only [Block.exec, selected]

/-- Exact reduction for a branch whose selector is the head of its input
context.  Concrete protocol programs use this form after computing a Boolean
selector, so callers need not unfold dependent reference lookup to expose the
selected arm. -/
@[simp] theorem branch_here_exec
    {signature : Signature.{u}}
    {tail joined result : Schema signature.types}
    (layout : Layout)
    (bit : signature.types.Value .bit)
    (values : Schema.Values signature.types tail)
    (onTrue onFalse :
      Block signature ({ kind := .bit, layout := layout } :: tail) joined)
    (continuation :
      Block signature
        (joined ++ ({ kind := .bit, layout := layout } :: tail)) result) :
    (Block.branch
        (Ref.here { kind := .bit, layout := layout })
        onTrue onFalse continuation).exec (.cons bit values) =
      match signature.types.bitValue bit with
      | true =>
          match onTrue.exec (.cons bit values) with
          | none => none
          | some exposed =>
              continuation.exec (exposed.append (.cons bit values))
      | false =>
          match onFalse.exec (.cons bit values) with
          | none => none
          | some exposed =>
              continuation.exec (exposed.append (.cons bit values)) :=
  by
    simp only [Block.exec, Ref.get_here]
    cases selected : signature.types.bitValue bit <;> simp [selected]

/-- Artifact-independent block relation.  Branch choice, both arm shapes, and
the join are explicit in the relation rather than delegated to a caller. -/
def Holds {signature : Signature.{u}} :
    {input output : Schema signature.types} ->
    Block signature input output ->
    Schema.Values signature.types input ->
    Schema.Values signature.types output -> Prop
  | _, _, .yield references, source, result =>
      result = references.get source
  | _, _, .step primitive rest, source, result =>
      ∃ middle, primitive.Holds source middle ∧ rest.Holds middle result
  | _, _, .branch condition onTrue onFalse continuation, source, result =>
      (signature.types.bitValue (condition.get source) = true ∧
        ∃ joined,
          onTrue.Holds source joined ∧
          continuation.Holds (joined.append source) result) ∨
      (signature.types.bitValue (condition.get source) = false ∧
        ∃ joined,
          onFalse.Holds source joined ∧
          continuation.Holds (joined.append source) result)

/-- Exact semantic characterization of executable block success. -/
theorem exec_eq_some_iff_holds
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    (block : Block signature input output)
    (source : Schema.Values signature.types input)
    (result : Schema.Values signature.types output) :
    block.exec source = some result ↔ block.Holds source result := by
  induction block with
  | yield references =>
      simp [exec, Holds, eq_comm]
  | step primitive rest inductionHypothesis =>
      constructor
      · intro executed
        simp only [exec] at executed
        cases primitiveResult : primitive.exec source with
        | none =>
            simp [primitiveResult] at executed
        | some middle =>
            rw [primitiveResult] at executed
            exact ⟨middle,
              primitive.sound source middle primitiveResult,
              (inductionHypothesis middle result).mp executed⟩
      · intro holds
        rcases holds with ⟨middle, primitiveHolds, restHolds⟩
        have primitiveExecuted :=
          primitive.complete source middle primitiveHolds
        simp only [exec, primitiveExecuted]
        exact (inductionHypothesis middle result).mpr restHolds
  | branch condition onTrue onFalse continuation
      onTrueHypothesis onFalseHypothesis continuationHypothesis =>
      cases selected :
        signature.types.bitValue (condition.get source) with
      | false =>
          constructor
          · intro executed
            simp only [exec, selected] at executed
            cases armResult : onFalse.exec source with
            | none =>
                simp [armResult] at executed
            | some joined =>
                rw [armResult] at executed
                exact Or.inr ⟨selected,
                  joined,
                  (onFalseHypothesis source joined).mp armResult,
                  (continuationHypothesis
                    (joined.append source) result).mp executed⟩
          · intro holds
            rcases holds with trueArm | falseArm
            · simp [selected] at trueArm
            · rcases falseArm with
                ⟨_, joined, armHolds, continuationHolds⟩
              have armExecuted :=
                (onFalseHypothesis source joined).mpr armHolds
              simp only [exec, selected, armExecuted]
              exact
                (continuationHypothesis
                  (joined.append source) result).mpr continuationHolds
      | true =>
          constructor
          · intro executed
            simp only [exec, selected] at executed
            cases armResult : onTrue.exec source with
            | none =>
                simp [armResult] at executed
            | some joined =>
                rw [armResult] at executed
                exact Or.inl ⟨selected,
                  joined,
                  (onTrueHypothesis source joined).mp armResult,
                  (continuationHypothesis
                    (joined.append source) result).mp executed⟩
          · intro holds
            rcases holds with trueArm | falseArm
            · rcases trueArm with
                ⟨_, joined, armHolds, continuationHolds⟩
              have armExecuted :=
                (onTrueHypothesis source joined).mpr armHolds
              simp only [exec, selected, armExecuted]
              exact
                (continuationHypothesis
                  (joined.append source) result).mpr continuationHolds
            · simp [selected] at falseArm

theorem sound
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    (block : Block signature input output)
    (source : Schema.Values signature.types input)
    (result : Schema.Values signature.types output)
    (executed : block.exec source = some result) :
    block.Holds source result :=
  (block.exec_eq_some_iff_holds source result).mp executed

theorem complete
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    (block : Block signature input output)
    (source : Schema.Values signature.types input)
    (result : Schema.Values signature.types output)
    (holds : block.Holds source result) :
    block.exec source = some result :=
  (block.exec_eq_some_iff_holds source result).mpr holds

end Block

/-- A complete program fixes its runtime input and exposed result schemas.
Input values are supplied only to `exec`; they are never embedded in a source
instruction. -/
structure Program (signature : Signature.{u})
    (input output : Schema signature.types) where
  body : Block signature input output

namespace Program

def exec {signature : Signature.{u}}
    {input output : Schema signature.types}
    (program : Program signature input output)
    (source : Schema.Values signature.types input) :
    Option (Schema.Values signature.types output) :=
  program.body.exec source

def Holds {signature : Signature.{u}}
    {input output : Schema signature.types}
    (program : Program signature input output)
    (source : Schema.Values signature.types input)
    (result : Schema.Values signature.types output) : Prop :=
  program.body.Holds source result

theorem exec_eq_some_iff_holds
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    (program : Program signature input output)
    (source : Schema.Values signature.types input)
    (result : Schema.Values signature.types output) :
    program.exec source = some result ↔ program.Holds source result :=
  program.body.exec_eq_some_iff_holds source result

theorem sound
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    (program : Program signature input output)
    (source : Schema.Values signature.types input)
    (result : Schema.Values signature.types output)
    (executed : program.exec source = some result) :
    program.Holds source result :=
  (program.exec_eq_some_iff_holds source result).mp executed

theorem complete
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    (program : Program signature input output)
    (source : Schema.Values signature.types input)
    (result : Schema.Values signature.types output)
    (holds : program.Holds source result) :
    program.exec source = some result :=
  (program.exec_eq_some_iff_holds source result).mpr holds

end Program

end Nightstream.Implementation.Lowering.Typed
