import Nightstream.Implementation.R1CS.Ownership.FPrimeRecursive.FPrimeRecursiveManifestSchema

/-! Generated diagnostic direct-CCS bit-carrier data by `gadgets_f_prime_recursive_manifest`; do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeRecursiveManifest

def schemaVersion : Nat := 3
def artifactKind : String := "r1cs/f-prime-recursive-program-manifest"
def profile : String := "diagnostic/direct-ccs-bit-carrier/plain/stateless/steady-recursive"
def piCcsSourceCount : Nat := 15
def piCcsMatrixCount : Nat := 3
def piCcsOutputFieldCount : Nat := 6683
def totalRows : Nat := 9503595
def totalColumns : Nat := 9316338
def nifsRowStart : Nat := 20038
def nifsRowEnd : Nat := 5905540
def nifsRowCount : Nat := 5885502
def totalNonzeroEntries : Nat := 68752233
def totalSha256 : String := "1e29a670b8c98c78f2fd8fa3a24bddf7ccfaf46caf9672b60f3964a38953d2d9"

def topLevelFamilies : List RowRange :=
  [ { name := "fprime.recursive.prelude", rowStart := 0, rowEnd := 6176, nonzeroEntries := 46006, sha256 := "eac9e7aa86d2ca6825807ebfaa4e5de419f0b424ccc52c01fd412b336d8d19c0" }
  , { name := "fprime.recursive.transcript", rowStart := 6176, rowEnd := 20038, nonzeroEntries := 110540, sha256 := "abc1fd3a86a978c6226ba6e2116560b5f67dd1ecbea7f10c2bbdcc184dbacea5" }
  , { name := "fprime.recursive.nifs", rowStart := 20038, rowEnd := 5905540, nonzeroEntries := 39928100, sha256 := "b9cf49840ab41575bbbe74991b281e6384909131a7cb27717d26eee6a8c2ead5" }
  , { name := "fprime.recursive.prior_link", rowStart := 5905540, rowEnd := 5910785, nonzeroEntries := 38016, sha256 := "45bd49d84c9e1693d3dd38ee50587bc55940cde746c4e38a545eaca637ea8aea" }
  , { name := "fprime.recursive.nebula", rowStart := 5910785, rowEnd := 5910785, nonzeroEntries := 0, sha256 := "84a39c68a24131c8c8d551f5b97e172f633bf78f3e4d0d703615fc7032b314c0" }
  , { name := "fprime.recursive.accumulator", rowStart := 5910785, rowEnd := 9498305, nonzeroEntries := 28591308, sha256 := "77d3746e5084760c64449aa5361b0b1f09b0e41d0be98d12c9e98da460724ca1" }
  , { name := "fprime.recursive.counter", rowStart := 9498305, rowEnd := 9498827, nonzeroEntries := 2136, sha256 := "efa6aabdca286592f9a556ea2e6910ffba8486bf75131a8a0372cd788f3b8eeb" }
  , { name := "fprime.recursive.output", rowStart := 9498827, rowEnd := 9503595, nonzeroEntries := 36127, sha256 := "bd4825c2eba78de8924a5fa6feefc7b1ea90747641f353fcac95220f8a73f050" }
  ]

def nifsFamilies : List RowRange :=
  [ { name := "nifs.pi_ccs", rowStart := 20038, rowEnd := 5228628, nonzeroEntries := 36706775, sha256 := "98d619b01667facee31a4739e10e0581752e1d8dbce965fbc36e989a7e9affa9" }
  , { name := "nifs.pi_rlc", rowStart := 5228628, rowEnd := 5894851, nonzeroEntries := 3165255, sha256 := "4bd7375f19156143b0bb80ab9b43aefe460a26bccb3be62be710b6d34fce722f" }
  , { name := "nifs.pi_dec", rowStart := 5894851, rowEnd := 5905532, nonzeroEntries := 56046, sha256 := "5e0b3b1da50615f1f5eadbf2934a7c2b38df18de8a8e7e6ef1c440764749b504" }
  , { name := "nifs.point_binding", rowStart := 5905532, rowEnd := 5905540, nonzeroEntries := 24, sha256 := "b1288f019d8b18f9d54b7cd0dcf62461d5d6ecde22c89f29d6d67bce3b923d74" }
  ]

def projectionShared : RowRange := { name := "nifs.pi_rlc.projection_shared", rowStart := 5832267, rowEnd := 5834159, nonzeroEntries := 7520, sha256 := "9f1bfa6e886d07f9006015a1bc5a83d3301783892b98bcfff1f62b44999b0fd6" }
def projectionIdentityCount : Nat := 31
def projectionIdentityRows : Nat := 59396
def projectionPairCounts : List Nat := [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
def projectionIdentityRanges : List RowRange :=
  [ { name := "nifs.pi_rlc.projection_identity", rowStart := 5834159, rowEnd := 5836075, nonzeroEntries := 7647, sha256 := "d9308dfc01ce6b37ab3455b4672b22947bf8838b2ecc5fb85a49e2842023585a" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5836075, rowEnd := 5837991, nonzeroEntries := 7647, sha256 := "6132ddeaa28e75deb28318a5579b22c81eaca68049d903a38dcf876509dc1d4e" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5837991, rowEnd := 5839907, nonzeroEntries := 7647, sha256 := "59243a5db132a5edc0511449f2cd0bf795e54fa934d85458120757d5d0f207de" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5839907, rowEnd := 5841823, nonzeroEntries := 7647, sha256 := "76ee26076eda1ed0475b1ee4dcc45c1c0d8836cd2465a7f74fd616b9d81e470c" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5841823, rowEnd := 5843739, nonzeroEntries := 7647, sha256 := "a0411462b00f5d9895c6a70b4111714fc8aa0b5380c4d40766ebc30572d546cd" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5843739, rowEnd := 5845655, nonzeroEntries := 7647, sha256 := "d29b426f780b5f4f658a4ecf6e4077f014c3ff000758048e9a0ef2408fb22bea" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5845655, rowEnd := 5847571, nonzeroEntries := 7647, sha256 := "be1697b81eb43a1ae88156f19686f029c1563edbd9295c439d98881d373b44d2" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5847571, rowEnd := 5849487, nonzeroEntries := 7647, sha256 := "00694ccc4d88e7e4598d52ace88ba541233d9e192941cd8b452aa21d9a9892c4" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5849487, rowEnd := 5851403, nonzeroEntries := 7647, sha256 := "deed5dd8836c06f4d18173b86f313f306d32792761822ccea869d231cab5d9fa" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5851403, rowEnd := 5853319, nonzeroEntries := 7647, sha256 := "ee18c876630d2793c9b64455a94002af750fa0dbb1dc6b9f1fcdc48ec0987f32" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5853319, rowEnd := 5855235, nonzeroEntries := 7647, sha256 := "b0e43fccbd09853b2c8005f35e76fd55eda58b90594e81f83f67560362a6badc" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5855235, rowEnd := 5857151, nonzeroEntries := 7647, sha256 := "a65884790b037834d389fc34a21983ce0659ec1dcd0566e941b5cfef7e148781" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5857151, rowEnd := 5859067, nonzeroEntries := 7647, sha256 := "6651250ecb12fb2f89600180338969efcb0935eb718d11244b0e769ff79c592d" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5859067, rowEnd := 5860983, nonzeroEntries := 7647, sha256 := "8049f4a7613335c10578f05779ee3cdb5f6d0405a9febb2e36c814046f03621d" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5860983, rowEnd := 5862899, nonzeroEntries := 7647, sha256 := "8c5e31278483db4de405cf7995f39bd62388f18154a2737db8933755b7576cbd" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5862899, rowEnd := 5864815, nonzeroEntries := 7647, sha256 := "6bd141e49f35716ceb15d1fa56892c4b94d0a070fb303f5b3e81b2a02592e928" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5864815, rowEnd := 5866731, nonzeroEntries := 7647, sha256 := "cef8a8a34a5073066fa5525cf98a9018d42c61587eb617e25c045addc1d25774" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5866731, rowEnd := 5868647, nonzeroEntries := 7647, sha256 := "01f420205fd59dc43728ae6453c381ba935d95904b8ba3ad530d1d929bce6832" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5868647, rowEnd := 5870563, nonzeroEntries := 7647, sha256 := "d7699c112cde0d8cc459878906af0a18fa4a66ff0f2cddb806508598f09f0363" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5870563, rowEnd := 5872479, nonzeroEntries := 7647, sha256 := "0dc46ab22c3d81b050c75b13971427d63a911c082f787395d65bc19d99caec09" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5872479, rowEnd := 5874395, nonzeroEntries := 7647, sha256 := "29050c74a98257c14de6bf0446a40b0de452f9df19e19922b1927f7c675b90c4" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5874395, rowEnd := 5876311, nonzeroEntries := 7647, sha256 := "c7b6a5fa7872b5261634892485ad2032753b6445fa26ed705a36257e032cedbc" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5876311, rowEnd := 5878227, nonzeroEntries := 7647, sha256 := "c93d3d15353ed8832196562bf9222130d2114048dc089c35b90241e6bcaeae6f" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5878243, rowEnd := 5880159, nonzeroEntries := 7647, sha256 := "ece565d49605d796f3e8749f1624ab48dee2909ab6196070db9f71295bc797ce" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5880159, rowEnd := 5882075, nonzeroEntries := 7647, sha256 := "2e30d69dfb9d1715beee6f9058a40c05bc8091ec7ff05da36e9dc80619ceb049" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5882395, rowEnd := 5884311, nonzeroEntries := 7647, sha256 := "264e474420aa8fef87222db12aa4a8cd91e1223a4c00be53529b2f1fb4591ca3" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5884311, rowEnd := 5886227, nonzeroEntries := 7647, sha256 := "2babc234ce6c1a0cd7c59ab40f65aa318e05e8148f3125ae3f44e4c89a41942e" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5886547, rowEnd := 5888463, nonzeroEntries := 7647, sha256 := "37a53107a04836817c6b703f955ff2fedc3e942f89d8140e37c2e948b3197f19" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5888463, rowEnd := 5890379, nonzeroEntries := 7647, sha256 := "0fe076953e38f27b087a5b15eb7ff83e6f87c67214eb9e963b922ddac988bb97" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5890699, rowEnd := 5892615, nonzeroEntries := 7647, sha256 := "029706120f2d5f952ab98be16e149ccc8cdba7b3a74925afecebab68f95785bd" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5892615, rowEnd := 5894531, nonzeroEntries := 7647, sha256 := "0a1787c62bdb5ffbb6d85017fcd3c34ddf4177eb2b06188aae07ae3b7f058067" }
  ]

end Nightstream.Implementation.R1CS.FPrimeRecursiveManifest
