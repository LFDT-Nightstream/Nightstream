import NightstreamFPrime.Export.Stage1.PiCCSOriginalMatrixSupported

/-! Compare prepared Poseidon weights with the numeric row evaluator, and
check that zero sources do not evaluate any invocation input forms. -/

namespace NightstreamFPrime.Tests.PoseidonColumnWeights

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Lifecycle (productionShape)

private def column (index : Nat) : Fin 32 := ⟨index % 32, Nat.mod_lt _ (by decide)⟩

private def form (seed : Nat) : SparseForm 32 :=
  if seed % 7 = 0 then .empty else
    ⟨[⟨column seed, 3⟩, ⟨column (seed + 1), Poseidon2.ofNat (goldilocksModulus - 1)⟩,
      ⟨column seed, Poseidon2.ofNat (goldilocksModulus - 3)⟩,
      ⟨column (seed + 9), Poseidon2.ofNat (2^32 + seed)⟩]⟩

private def interface (seed : Nat) : PoseidonSboxPlan.Interface 32 :=
  { oneColumn := column seed
    input := fun lane => form (seed + lane.val)
    sboxOutput := fun slot => form (seed + slot.val + 17)
    output := fun lane => form (seed + lane.val + 4) }

private def point : CubePoint K 10 :=
  ⟨List.ofFn (fun lane : Fin 10 =>
    ⟨Poseidon2.ofNat (goldilocksModulus - 1 - lane.val),
      Poseidon2.ofNat (2^32 + lane.val)⟩), by simp⟩

private def read (lane : Fin ringDegree) (index : Fin 32) : F :=
  Poseidon2.ofNat (goldilocksModulus - 1 - (lane.val * 37 + index.val * 13))

private def values : IO Unit := do
  -- Empty and repeated columns, cancelling coefficients, and nonzero row offsets.
  for (count, firstRow) in [(0, 0), (1, 0), (1, 211), (3, 211)] do
    let interfaces := Vector.ofFn fun invocation : Fin count => interface (invocation.val * 7)
    let actual := PiDECPoseidonColumnWeights.evaluate
      (PiDECPoseidonColumnWeights.prepare firstRow point interfaces) read
    let expected := PiDECMatrixInvocationRange.sum firstRow point read interfaces
    for port in List.finRange matrixCount do
      for lane in List.finRange ringDegree do
        unless (actual.get port).toRing lane == (expected.get port).toRing lane do
          throw (IO.userError s!"Poseidon mismatch: count={count} row={firstRow} port={port.val} lane={lane.val}")

private def selectedSources : IO Unit := do
  let interfaces := Vector.singleton (interface 1)
  let expected := PiDECMatrixInvocationRange.sum 211 point read interfaces
  for selected in [0, productionShape.sourceCount - 1] do
    let actual := PiCCSOriginalMatrixSupported.invocations
      (fun source => source.val != selected) 211 point
      (fun source => if source.val == selected then read else fun _ _ => 0) interfaces
    for source in List.finRange productionShape.sourceCount do
      for port in List.finRange matrixCount do
        for lane in List.finRange ringDegree do
          let wanted := if source.val == selected then (expected.get port).toRing lane else K.zero
          unless (actual.get (Fin.encodeProd (source, port))).toRing lane == wanted do
            throw (IO.userError "source selection changed the Poseidon matrix result")

private def zeroPreparation : IO Unit := do
  let selected := interface 0
  let traced : PoseidonSboxPlan.Interface 32 :=
    { selected with input := fun lane => dbg_trace "invocation input evaluated"; selected.input lane }
  -- Read flags inside the capture so the compiler cannot specialize an all-true predicate.
  let flags ← IO.mkRef (Vector.replicate productionShape.sourceCount true)
  let (trace, actual) ← IO.FS.withIsolatedStreams do
    let zeroSource ← flags.get
    pure (PiCCSOriginalMatrixSupported.invocations zeroSource.get 0 point
      (fun _ _ _ => 0) (Vector.singleton traced))
  unless trace.isEmpty do throw (IO.userError "zero sources prepared invocation column weights")
  for code in List.finRange (productionShape.sourceCount * matrixCount) do
    for lane in List.finRange ringDegree do
      unless (actual.get code).toRing lane == K.zero do
        throw (IO.userError "zero sources produced a nonzero matrix result")

#eval values
#eval selectedSources
#eval zeroPreparation

end NightstreamFPrime.Tests.PoseidonColumnWeights
