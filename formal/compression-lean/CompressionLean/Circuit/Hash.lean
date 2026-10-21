import CompressionLean.Circuit.Algebra

/-!
Owns the hashes of `crates/neo-spartan/src/circuit/hash.rs` as builder
programs: the Plonky3 duplex challenger (`Duplex`, `Transcript`), the leaf
sponge (`hashLeaf`), node compression (`compress`) and Merkle paths
(`merkleRoot`). Each one calls `Build` in the order of its Rust counterpart.

A `Transcript` program owns the duplex, as Rust's `&mut Duplex` does; the
builder's calls lift into it.
-/

namespace CompressionLean.Circuit

open CompressionLean.Replay (Gl)

/-- The sponge rate (`RATE`); the width is 16. -/
def rate : ℕ := 12

/-- A digest: four handles (`hash::Digest`). -/
abbrev DigestVar := Fin 4 → Var

/-- The chunks of `RATE` words, the last one partial (`<[_]>::chunks`). -/
def chunks (values : Array Var) : Array (Array Var) :=
  (Array.range ((values.size + rate - 1) / rate)).map fun c => values.extract (rate * c) (rate * c + rate)

/-- The duplex state, the pending input and the unread output
(`hash.rs:15-19`). The state has 16 lanes. -/
structure Duplex where
  state : Array Var
  input : Array Var
  output : Array Var

/-- A program that reads and writes the verifier's duplex. -/
abbrev Transcript := StateT Duplex Build

namespace Duplex

/-- The zero state (`hash.rs:22-29`). -/
def new : Build Duplex := do
  let zeroWord ← Build.const 0
  pure ⟨Array.replicate 16 zeroWord, #[], #[]⟩

/-- Overwrite the absorbed lanes; after any input, zero the rest of the rate
and add the length tag to lane `RATE`; permute and refill the output
(`hash.rs:75-89`). -/
def duplexing : Transcript Unit := do
  let d ← get
  let absorbed := d.input.size
  let mut state := d.state
  for (value, lane) in d.input.zipIdx do
    state := state.set! lane value
  if absorbed > 0 then
    let zeroWord ← Build.const 0
    for lane in [absorbed:rate] do
      state := state.set! lane zeroWord
    let tag ← Build.const (Gl.ofNat absorbed)
    state := state.set! rate (← Build.add state[rate]! tag)
  let next := Array.ofFn (← Build.permute fun lane => state[lane]!)
  set ({ state := next, input := #[], output := next.extract 0 rate } : Duplex)

/-- `hash.rs:31-37`. -/
def observe (value : Var) : Transcript Unit := do
  modify fun d => { d with output := #[], input := d.input.push value }
  if (← get).input.size = rate then duplexing

/-- `hash.rs:39-43`. -/
def observeSlice (values : Array Var) : Transcript Unit :=
  values.forM observe

/-- The coordinates in order (`hash.rs:46-50`). -/
def observeExt (value : ExtVar) : Transcript Unit := do
  observe (value 0)
  observe (value 1)
  observe (value 2)

/-- The last unread output lane, after a duplexing when there is input or no
output (`hash.rs:52-59`). -/
def sample : Transcript Var := do
  let d ← get
  if !d.input.isEmpty || d.output.isEmpty then duplexing
  let d ← get
  set { d with output := d.output.pop }
  pure d.output.back!

/-- Coordinate `i` is the `i`-th sample (`hash.rs:62-65`). -/
def sampleExt : Transcript ExtVar := do
  let x ← sample
  let y ← sample
  let z ← sample
  pure ![x, y, z]

/-- The low `bits` of the canonical sample's 64 bits (`hash.rs:68-73`). -/
def sampleBits (bits : ℕ) : Transcript (Array Var) := do
  let word ← sample
  let all ← Build.bits word 64
  pure ((Array.ofFn all).extract 0 bits)

end Duplex

/-- The overwrite sponge of Plonky3's `PaddingFreeSponge` (`hash.rs:93-101`). -/
def hashLeaf (values : Array Var) : Build DigestVar := do
  let zeroWord ← Build.const 0
  let mut state := Array.replicate 16 zeroWord
  for chunk in chunks values do
    for (value, lane) in chunk.zipIdx do
      state := state.set! lane value
    state := Array.ofFn (← Build.permute fun lane => state[lane]!)
  pure fun lane => state[lane]!

/-- `[left, right, 0…]`, permute, the first four lanes (`hash.rs:104-111`). -/
def compress (left right : DigestVar) : Build DigestVar := do
  let zeroWord ← Build.const 0
  let lanes ← Build.permute fun lane =>
    if h : lane.val < 4 then left ⟨lane, h⟩
    else if h' : lane.val < 8 then right ⟨lane - 4, by omega⟩
    else zeroWord
  pure fun lane => lanes ⟨lane, by omega⟩

/-- The root of `leaf` along `path`, bottom first; bit zero puts the node on
the left (`hash.rs:115-129`). -/
def merkleRoot (leaf : DigestVar) (bits : Array Var) (path : Array DigestVar) : Build DigestVar := do
  let mut node := leaf
  for (bit, sibling) in bits.zip path do
    let mut left := Array.ofFn node
    let mut right := Array.ofFn node
    for lane in List.finRange 4 do
      left := left.set! lane (← Build.select bit (node lane) (sibling lane))
      let both ← Build.add (node lane) (sibling lane)
      right := right.set! lane (← Build.sub both left[lane]!)
    node ← compress (fun lane => left[lane]!) (fun lane => right[lane]!)
  pure node

end CompressionLean.Circuit
