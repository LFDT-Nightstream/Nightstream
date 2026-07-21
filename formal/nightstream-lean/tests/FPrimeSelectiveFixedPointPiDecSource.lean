import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Nifs.PiDec.SourceRefinement

/-! Focused interface regression for the bounded active strict-`PiDEC`
source-row artifact. -/

namespace Nightstream.Tests.FPrimeSelectiveFixedPointPiDecSource

open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Nifs.PiDec.SourceRefinement

#check sourceRows_length
#check SourceArtifact.compilerRows_length
#check SourceArtifact.sourceRows_exact
#check SourceArtifact.compilerRows_exact
#check SourceArtifact.shapeValid
#check SourceArtifact.activeProfile
#check sourceRows_imply_compilerAccepted
#check sourceRows_imply_typedAccepted
#check sourceRows_imply_paperAccepted

end Nightstream.Tests.FPrimeSelectiveFixedPointPiDecSource
