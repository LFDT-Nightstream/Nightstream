import CompressionLean.Ports.Circuit.Algebra

namespace CompressionLean.Circuit

open CompressionLean.Replay (Word)

def rate : ℕ := 12

abbrev DigestVar := Fin 4 → Var

def chunks (values : Array Var) : Array (Array Var) :=
  (Array.range ((values.size + rate - 1) / rate)).map fun c => values.extract (rate * c) (rate * c + rate)

structure Duplex where
  state : Array Var
  input : Array Var
  output : Array Var

abbrev Transcript := StateT Duplex Build

namespace Duplex

def new : Build Duplex := do
  let zeroWord ← Build.const 0
  pure ⟨Array.replicate 16 zeroWord, #[], #[]⟩

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
    let tag ← Build.const (Word.ofNat absorbed)
    state := state.set! rate (← Build.add state[rate]! tag)
  let next := Array.ofFn (← Build.permute fun lane => state[lane]!)
  set ({ state := next, input := #[], output := next.extract 0 rate } : Duplex)

def observe (value : Var) : Transcript Unit := do
  modify fun d => { d with output := #[], input := d.input.push value }
  if (← get).input.size = rate then duplexing

def observeSlice (values : Array Var) : Transcript Unit :=
  values.forM observe

def observeExt (value : ExtVar) : Transcript Unit := do
  observe (value 0)
  observe (value 1)
  observe (value 2)

def sample : Transcript Var := do
  let d ← get
  if !d.input.isEmpty || d.output.isEmpty then duplexing
  let d ← get
  set { d with output := d.output.pop }
  pure d.output.back!

def sampleExt : Transcript ExtVar := do
  let x ← sample
  let y ← sample
  let z ← sample
  pure ![x, y, z]

def sampleBits (bits : ℕ) : Transcript (Array Var) := do
  let word ← sample
  let all ← Build.bits word 64
  pure ((Array.ofFn all).extract 0 bits)

end Duplex

def hashLeaf (values : Array Var) : Build DigestVar := do
  let zeroWord ← Build.const 0
  let mut state := Array.replicate 16 zeroWord
  for chunk in chunks values do
    for (value, lane) in chunk.zipIdx do
      state := state.set! lane value
    state := Array.ofFn (← Build.permute fun lane => state[lane]!)
  pure fun lane => state[lane]!

def compress (left right : DigestVar) : Build DigestVar := do
  let zeroWord ← Build.const 0
  let lanes ← Build.permute fun lane =>
    if h : lane.val < 4 then left ⟨lane, h⟩
    else if h' : lane.val < 8 then right ⟨lane - 4, by omega⟩
    else zeroWord
  pure fun lane => lanes ⟨lane, by omega⟩

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
