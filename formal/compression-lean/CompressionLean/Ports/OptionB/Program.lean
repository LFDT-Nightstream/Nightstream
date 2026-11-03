import CompressionLean.Ports.Terminal.Program
import CompressionLean.Ports.Circuit.Verifier
import NightstreamFPrime.Export.OptionB.StepPackage

/-! `lifecycle/option_b/final_program.rs::FinalProgram::run` (CW-6.5): the statement and a positive
iteration; the finish package's columns below its S-box columns, with column 0 one and the
statement and the state keys bound to their shared words; the package's rows other than the
blocks' S-box and binding rows as one R1CS block, each row with its constant on column 0
(`Terminal.Package.row`); each Poseidon2 block as the permutation of its input lanes into its
output locals; then, for each slot, the layer-1 verifier of the next slot's claim words, seeded
with the state hash that the fresh public input's bits encode and the slot. The word positions are
those of `RefreshAssembler.interface` from the public columns. -/

namespace CompressionLean.OptionB

open CompressionLean.Circuit
open CompressionLean.Replay (Word)
open NightstreamFPrime.Export.OptionB.StepPackage (Package PackagePiece)

/-- `final_program.rs::PUBLIC_WORDS`, where the shared words start. -/
def publicWords : ℕ := 270

/-- `final_program.rs::ITERATION`. -/
def iterationWord : ℕ := publicWords

/-- `final_program.rs::KEYS`: both circuits' key digests. -/
def keysWord : ℕ := publicWords + 5

/-- `final_program.rs::Z0`. -/
def z0Word : ℕ := keysWord + 8

/-- `final_program.rs::CURRENT`. -/
def currentWord : ℕ := z0Word + 4

/-- `final_program.rs::WITNESS`: the application witness, after the current and next states. -/
def witnessWord : ℕ := currentWord + 8

/-- `final_program.rs::SLOT_WORDS`. -/
def slotWords : ℕ := 2055

/-- `final_program.rs::FinalProgram::slot_start`: slot `slot` of the step's seven. -/
def slotStart (witnessWords slot : ℕ) : ℕ := witnessWord + witnessWords + slot * slotWords

/-- The fresh public input's first bit (`FinalProgram::seed`): after the slot and output digests,
the fresh commitment and the public input's constant one. -/
def bitsStart (witnessWords : ℕ) : ℕ := slotStart witnessWords 7 + 20 + 1188 + 1

/-- `option_b.rs::Linear::scale`. -/
def scaleForm (factor : NightstreamFPrime.Spec.F)
    (form : NightstreamFPrime.Spec.F × List (ℕ × NightstreamFPrime.Spec.F)) :
    NightstreamFPrime.Spec.F × List (ℕ × NightstreamFPrime.Spec.F) :=
  (form.1 * factor, form.2.map fun term => (term.1, term.2 * factor))

/-- `option_b.rs::decode_linear`: a lane's constant and terms; `none` for a product of two
non-constant expressions. -/
def linearForm :
    NightstreamFPrime.Circuit.Expr →
      Option (NightstreamFPrime.Spec.F × List (ℕ × NightstreamFPrime.Spec.F))
  | .var column => some (0, [(column, 1)])
  | .const value => some (value, [])
  | .add left right => do
      let (a, s) ← linearForm left
      let (b, t) ← linearForm right
      pure (a + b, s ++ t)
  | .mul (.const factor) right => scaleForm factor <$> linearForm right
  | .mul left (.const factor) => scaleForm factor <$> linearForm left
  | .mul _ _ => none

/-- `final_program.rs::Finish::new`: the rows of the package's pieces, before the binding rows. -/
def finishRows (finish : Package) : List (Shrink.R1csRow Word) :=
  (finish.pieces.flatMap PackagePiece.relationRows).map (Terminal.Package.row 0)

/-- One input lane of a block (`FinalProgram::run`): its constant, plus each scaled term. -/
def laneValue (column : ℕ → Var) (lane : NightstreamFPrime.Circuit.Expr) : Build Var := do
  let form := (linearForm lane).getD (0, [])
  let start ← Build.const (Word.ofF form.1)
  form.2.foldlM (fun sum term => do
    let scaled ← Build.scale (column term.1) (Word.ofF term.2)
    Build.add sum scaled) start

/-- A Poseidon2 block (`FinalProgram::run`): its output locals are the permutation of its lanes. -/
def checkBlock (column : ℕ → Var) (block : NightstreamFPrime.Export.Generic.Frame.Block) :
    Build Unit := do
  let lanes ← block.lanes.mapM (laneValue column)
  let outputs ← Build.permute fun lane => lanes.getD lane 0
  (List.finRange 16).forM fun lane =>
    Build.assertEqual (column (block.start + 1080 + lane.val)) (outputs lane)

/-- `compression.rs::SEED_DOMAIN`. -/
def seedDomain : String := "Nightstream/OptionB/compression/v1"

/-- `Poseidon2Transcript::new_v1_2` after it absorbs `domain_chunk_v1_2(SEED_DOMAIN)`. -/
def seedDomainState : Array Word :=
  Replay.permute (Array.ofFn fun lane : Fin 16 => (domainChunk seedDomain).getD lane 0)

/-- `FinalProgram::seed`: the domain's state, plus the four state-hash words that the bits encode
and the slot, permuted (`OptionB::layer1_seed`). -/
def seed (column : ℕ → Var) (bits slot : ℕ) : Build (Fin 16 → Var) := do
  let mut state ← (Array.range 16).mapM fun lane => Build.const seedDomainState[lane]!
  for word in List.range 4 do
    let mut value ← Build.const 0
    for bit in List.range 64 do
      let scaled ← Build.scale (column (bits + 64 * word + bit)) (Word.ofNat (2 ^ bit))
      value ← Build.add value scaled
    state := state.set! word (← Build.add state[word]! value)
  let index ← Build.const (Word.ofNat slot)
  state := state.set! 4 (← Build.add state[4]! index)
  Build.permute fun lane => state[lane]!

/-- `FinalProgram::claim`: a slot's claim words from `start` (`LightAssembler.slotAt`). -/
def claimAt (column : ℕ → Var) (start : ℕ) : ClaimView where
  commitment := (Array.range 22).map fun row => (Array.range 54).map fun lane => column (start + row * 54 + lane)
  publicInput := (Array.range 5).map fun block =>
    (Array.range 54).map fun lane => column (start + 1784 + block * 54 + lane)
  point := (Array.range 28).map fun k => (column (start + 1728 + 2 * k), column (start + 1728 + 2 * k + 1))
  evalK := (Array.range 54).map fun k => (column (start + 1188 + 2 * k), column (start + 1188 + 2 * k + 1))
  evalA := (Array.range 4).map fun family => (Array.range 54).map fun k =>
    (column (start + 1188 + (family + 1) * 108 + 2 * k), column (start + 1188 + (family + 1) * 108 + 2 * k + 1))

/-- `FinalProgram::run` before layer 1: the statement handles, the finish columns' first handle. -/
def finishPrefix (finish : Package) (stateKeys : List Word) : Build (Array Var × Var) := do
  let statement ← (Array.range Terminal.Program.statementWords).mapM fun _ => Build.pub
  let inverse ← Build.emit (.hint 1)
  let product ← Build.mul (statement.getD 0 0) inverse
  let one ← Build.const 1
  Build.assertEqual product one
  let first ← Build.emit (.hint finish.sboxStart)
  let column : ℕ → Var := fun k => first + k
  Build.assertEqual (column 0) one
  Build.assertEqual (column iterationWord) (statement.getD 0 0)
  (List.range 4).forM fun lane => do
    Build.assertEqual (column (z0Word + lane)) (statement.getD (1 + lane) 0)
    Build.assertEqual (column (currentWord + lane)) (statement.getD (5 + lane) 0)
  stateKeys.zipIdx.forM fun (word, index) => do
    let key ← Build.const word
    Build.assertEqual (column (keysWord + index)) key
  Build.r1cs (finishRows finish) ((List.range finish.sboxStart).map column)
  finish.blocks.forM (checkBlock column)
  pure (statement, first)

/-- `FinalProgram::run`: the prefix, then each slot's layer-1 verifier. -/
def finalProgram (finish : Package) (stateKeys : List Word) (keys : Fin 2 → Layer1Key) : Build Unit := do
  let (_, first) ← finishPrefix finish stateKeys
  let column : ℕ → Var := fun k => first + k
  let witnessWords := finish.application.1
  (List.finRange 2).forM fun slot => do
    let state ← seed column (bitsStart witnessWords) slot
    verifyWith (keys slot) state (claimAt column (slotStart witnessWords (5 + slot)))

end CompressionLean.OptionB
