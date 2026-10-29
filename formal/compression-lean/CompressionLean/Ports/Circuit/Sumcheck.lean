import CompressionLean.Ports.Circuit.Hash

namespace CompressionLean.Circuit

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
