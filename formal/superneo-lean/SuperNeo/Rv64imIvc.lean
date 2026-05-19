import SuperNeo.Rv64imIvcInterface

namespace SuperNeo

namespace Rv64imIvc

def implementationModule : String :=
  Rv64imIvcInterface.implementationModule

theorem init_contract :
    Rv64imIvcInterface.initContract := by
  exact Rv64imIvcInterface.initContract_true

theorem append_contract :
    Rv64imIvcInterface.appendContract := by
  exact Rv64imIvcInterface.appendContract_true

theorem verify_contract :
    Rv64imIvcInterface.verifyContract := by
  exact Rv64imIvcInterface.verifyContract_true

theorem compress_contract :
    Rv64imIvcInterface.compressContract := by
  exact Rv64imIvcInterface.compressContract_true

theorem resume_contract :
    Rv64imIvcInterface.resumeContract := by
  exact Rv64imIvcInterface.resumeContract_true

end Rv64imIvc

end SuperNeo
