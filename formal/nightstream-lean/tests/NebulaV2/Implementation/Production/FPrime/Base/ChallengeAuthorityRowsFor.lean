import Nightstream.Implementation.NebulaV2.Production.FPrime.Base.ChallengeAuthoritySoundFor

/-! Regression surface for the row-derived base challenge authority. -/

namespace tests.NebulaV2ProductionBaseChallengeAuthorityRowsFor

open Nightstream.Implementation.NebulaV2.ProductionBaseChallengeAuthorityRowsFor

#check Program.dynamicAuthorityLinkRows_length
#check Program.rows_length_exact
#check Program.rows_length_25
#check Program.rows_length_26
#check Program.rows_imply_dynamicAuthorityExact
#check Program.rows_imply_initialStateAuthorityLane
#check Program.rows_imply_preCarryAuthorityLane
#check Program.rows_imply_openingAuthorityPlaced
#check Program.satisfies_of_matchesArtifact

end tests.NebulaV2ProductionBaseChallengeAuthorityRowsFor
