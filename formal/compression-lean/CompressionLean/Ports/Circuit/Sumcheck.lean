import CompressionLean.Ports.Circuit.Hash

/-!
Owns the sum-check replay of `crates/neo-spartan/src/sumcheck.rs`
(`replay`) as a builder program. A round message is
`h(0), h(2), …, h(degree)`; `h(1)` is the claim minus `h(0)`. The caller
fixes the shape.
-/

namespace CompressionLean.Circuit

/-- Replay `rounds` against `claim`; return the point and the final claim
(`sumcheck.rs::replay`). -/
def sumcheckReplay (rounds : Array (Array ExtVar)) (claim : ExtVar) : Transcript (Array ExtVar × ExtVar) := do
  let mut point := #[]
  let mut claim := claim
  for message in rounds do
    let atOne ← Build.extSub claim message[0]!
    let values := #[message[0]!, atOne] ++ message.extract 1 message.size
    for value in message do
      Duplex.observeExt value
    let r ← Duplex.sampleExt
    claim ← Algebra.interpolate values r
    point := point.push r
  pure (point, claim)

end CompressionLean.Circuit
