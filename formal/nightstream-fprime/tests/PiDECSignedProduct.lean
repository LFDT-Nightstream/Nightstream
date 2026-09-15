import NightstreamFPrime.Export.Stage1.PiDECNativeProduct

/-! Compare the signed native path with the retained general product. -/

namespace NightstreamFPrime.Tests.PiDECSignedProduct

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
open NightstreamFPrime.Export.Stage1.PiDECNativeProduct

private def check (label : String) (initial : Accumulator)
    (key digit : StoredRing) : IO Unit := do
  let prepared := prepareKey key
  let actual := (initial.addPreparedProduct prepared (prepareDigit digit)).finish
  let expected := (initial.addProduct prepared digit).finish
  unless actual == expected do
    throw (IO.userError s!"signed product mismatch: {label}")

private def regression : IO Unit := do
  for keyIndex in [:ringDegree] do
    let key : StoredRing := Vector.ofFn fun lane => if lane.val = keyIndex then 1 else 0
    for digitIndex in [:ringDegree] do
      for sign in [1, (-1 : F)] do
        let digit : StoredRing := Vector.ofFn fun lane =>
          if lane.val = digitIndex then sign else 0
        check s!"basis {keyIndex}/{digitIndex}/{sign.val}" Accumulator.zero key digit
  let key : StoredRing := Vector.ofFn fun lane =>
    match lane.val % 4 with
    | 0 => 0
    | 1 => 1
    | 2 => -1
    | _ => ⟨goldilocksModulus - 2, by decide⟩
  let signed : StoredRing := Vector.ofFn fun lane =>
    match lane.val % 3 with
    | 0 => 0
    | 1 => 1
    | _ => -1
  let initial := Accumulator.zero.addProduct (prepareKey key) signed
  check "mixed signs and nonzero initial" initial key signed
  check "zero" initial key (Vector.replicate ringDegree 0)
  check "cancellation" initial key (signed.map fun value => -value)
  for lane in [:ringDegree] do
    for unsupported in [2, (-2 : F), 65535, (-65535 : F)] do
      let digit : StoredRing := Vector.ofFn fun index =>
        if index.val = lane then unsupported else signed.get index
      check s!"fallback {lane}/{unsupported.val}" initial key digit

#eval regression

end NightstreamFPrime.Tests.PiDECSignedProduct
