import CompressionLean.Ports.Terminal.Program
import CompressionLean.Ports.Circuit.Verifier
import NightstreamFPrime.Export.BoundedDepth.StepPackage
import NightstreamFPrime.Lifecycle.Nebula.FactorWords
import NightstreamFPrime.Spec
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.GoldilocksCausal
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetConvention
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork
import NightstreamFPrime.Spec.Nebula

/-! `lifecycle/bounded_depth/final_program.rs::FinalProgram::run` (CW-6.5): the statement and a positive
iteration; the finish package's columns below its S-box columns, a cell for each column that the
program reads (`live`) and the constant 0 for each other column, with column 0 one and the
statement and the state keys bound to their shared words; the package's rows other than the
blocks' S-box and binding rows as one R1CS block, each row with its constant on column 0
(`Terminal.Package.row`); each Poseidon2 block as the permutation of its input lanes into its
output locals; then, for each slot, the layer-1 verifier of the next slot's claim words, seeded
with the state hash that the fresh public input's bits encode and the slot. The word positions are
those of `RefreshAssembler.interface` from the public columns. -/

namespace CompressionLean.BoundedDepth

open CompressionLean.Circuit
open CompressionLean.Replay (Word)
open NightstreamFPrime.Export.BoundedDepth.StepPackage (Package PackagePiece)

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

/-- `final_program.rs::Finish::slot_start`: slot `slot` of the step's seven. -/
def slotStart (witnessWords slot : ℕ) : ℕ := witnessWord + witnessWords + slot * slotWords

/-- The fresh public input's first bit (`Finish::fresh_public_start`, `FinalProgram::seed`): after
the slot and output digests, the fresh commitment and the public input's constant one. -/
def bitsStart (witnessWords : ℕ) : ℕ := slotStart witnessWords 7 + 20 + 1188 + 1

/-- `bounded_depth.rs::Linear::scale`. -/
def scaleForm (factor : NightstreamFPrime.Spec.F)
    (form : NightstreamFPrime.Spec.F × List (ℕ × NightstreamFPrime.Spec.F)) :
    NightstreamFPrime.Spec.F × List (ℕ × NightstreamFPrime.Spec.F) :=
  (form.1 * factor, form.2.map fun term => (term.1, term.2 * factor))

/-- `bounded_depth.rs::decode_linear`: a lane's constant and terms; `none` for a product of two
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

/-- `final_program.rs::Finish::program_reads`: the columns that `FinalProgram::run` reads outside the
rows and the blocks. They are the constant, the statement and key words, the two slot claims of
layer 1 and the bits of the state hash. -/
def programReads (witnessWords : ℕ) : List ℕ :=
  ([(0, 1), (iterationWord, 1), (keysWord, 8), (z0Word, 4), (currentWord, 4),
      (slotStart witnessWords 5, slotWords - 1), (slotStart witnessWords 6, slotWords - 1),
      (bitsStart witnessWords, 4 * 64)] : List (ℕ × ℕ)).flatMap fun range => List.range' range.1 range.2

/-- `Finish::new`: `live` with the column of each term set. -/
def mark {α : Type} (column : α → ℕ) (live : Array Bool) (terms : List α) : Array Bool :=
  terms.foldl (fun live term => live.set! (column term) true) live

/-- `Finish::new`: the columns below the S-box columns that the final program reads (`live`). They
are the columns of the rows, of each block's lanes and outputs, and `programReads`. -/
def liveColumns (rows : List (Shrink.R1csRow Word)) (blocks : List NightstreamFPrime.Export.Generic.Frame.Block)
    (witnessWords sboxStart : ℕ) : Array Bool :=
  let live := rows.foldl (fun live row => mark Prod.fst (mark Prod.fst (mark Prod.fst live row.a) row.b) row.c)
    (Array.replicate sboxStart false)
  let live := blocks.foldl (fun live block =>
    mark id (block.lanes.foldl (fun live lane => mark Prod.fst live ((linearForm lane).getD (0, [])).2) live)
      (List.range' (block.start + 1080) 16)) live
  mark id live (programReads witnessWords)

/-- `Finish::live`: the live columns of the finish package. -/
def live (finish : Package) : Array Bool :=
  liveColumns (finishRows finish) finish.blocks finish.application.1 finish.sboxStart

/-- `FinalProgram::run`, the forms of `Columns`: the next cell from `first` for each live column,
`zeroForm` for each other column. -/
def columnForms (live : Array Bool) (first zeroForm : Var) : Array Var :=
  (live.foldl (fun (state : Array Var × Var) flag =>
    if flag then (state.1.push state.2, state.2 + 1) else (state.1.push zeroForm, state.2)) (#[], first)).1

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
@[irreducible] def seedDomainState : Array Word :=
  Replay.permute (Array.ofFn fun lane : Fin 16 => (domainChunk seedDomain).getD lane 0)

/-- `FinalProgram::seed` before its permutation: the domain's state, plus the four state-hash words
that the bits encode and the slot. -/
def seedState (column : ℕ → Var) (bits slot : ℕ) : Build (Array Var) := do
  let mut state ← (Array.range 16).mapM fun lane => Build.const seedDomainState[lane]!
  for word in List.range 4 do
    let mut value ← Build.const 0
    for bit in List.range 64 do
      let scaled ← Build.scale (column (bits + 64 * word + bit)) (Word.ofNat (2 ^ bit))
      value ← Build.add value scaled
    state := state.set! word (← Build.add state[word]! value)
  let index ← Build.const (Word.ofNat slot)
  pure (state.set! 4 (← Build.add state[4]! index))

/-- `FinalProgram::seed`: that state permuted (`BoundedDepth::layer1_seed`). -/
def seed (column : ℕ → Var) (bits slot : ℕ) : Build (Fin 16 → Var) := do
  let state ← seedState column bits slot
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

/-- `FinalProgram::run`: column 0 is one, and the statement and the state keys are on their words. -/
def bindWords (column : ℕ → Var) (statement : Array Var) (one : Var) (stateKeys : List Word) : Build Unit := do
  Build.assertEqual (column 0) one
  Build.assertEqual (column iterationWord) (statement.getD 0 0)
  (List.range 4).forM fun lane => do
    Build.assertEqual (column (z0Word + lane)) (statement.getD (1 + lane) 0)
    Build.assertEqual (column (currentWord + lane)) (statement.getD (5 + lane) 0)
  stateKeys.zipIdx.forM fun (word, index) => do
    let key ← Build.const word
    Build.assertEqual (column (keysWord + index)) key

/-- `FinalProgram::run`: each Poseidon2 block. -/
def checkBlocks (column : ℕ → Var) (blocks : List NightstreamFPrime.Export.Generic.Frame.Block) : Build Unit :=
  blocks.forM (checkBlock column)

/-- `FinalProgram::run` from the hint of the finish columns to the blocks. A live column has its
cell of the hint, and each other column has the constant 0. The result is the handle of each
column (`Columns::at`). -/
def finishColumns (rows : List (Shrink.R1csRow Word)) (blocks : List NightstreamFPrime.Export.Generic.Frame.Block)
    (live : Array Bool) (statement : Array Var) (one : Var) (stateKeys : List Word) : Build (ℕ → Var) := do
  let first ← Build.emit (.hint (live.count true))
  let zeroForm ← Build.const 0
  let forms := columnForms live first zeroForm
  let column : ℕ → Var := fun k => forms.getD k zeroForm
  bindWords column statement one stateKeys
  Build.r1cs rows forms.toList
  checkBlocks column blocks
  pure column

/-- `FinalProgram::run` before layer 1: the statement handles and the handle of each finish column. -/
def finishPrefix (finish : Package) (stateKeys : List Word) : Build (Array Var × (ℕ → Var)) := do
  let statement ← (Array.range Terminal.Program.statementWords).mapM fun _ => Build.pub
  let inverse ← Build.emit (.hint 1)
  let product ← Build.mul (statement.getD 0 0) inverse
  let one ← Build.const 1
  Build.assertEqual product one
  let rows := finishRows finish
  let column ← finishColumns rows finish.blocks
    (liveColumns rows finish.blocks finish.application.1 finish.sboxStart) statement one stateKeys
  pure (statement, column)

/-- `FinalProgram::run` for one slot: its seed, then the layer-1 verifier of its next claim. -/
def slotLayer1 (finish : Package) (key : Layer1Key) (column : ℕ → Var) (slot : ℕ) : Build Unit := do
  let state ← seed column (bitsStart finish.application.1) slot
  verifyWith key state (claimAt column (slotStart finish.application.1 (5 + slot)))

/-- `FinalProgram::run`: the prefix, then each slot's layer-1 verifier. -/
def finalProgram (finish : Package) (stateKeys : List Word) (keys : Fin 2 → Layer1Key) : Build Unit := do
  let start ← finishPrefix finish stateKeys
  slotLayer1 finish (keys 0) start.2 0
  slotLayer1 finish (keys 1) start.2 1

end CompressionLean.BoundedDepth
