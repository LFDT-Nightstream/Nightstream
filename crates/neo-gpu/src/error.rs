use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum NeoGpuError {
    #[error("failed to load GPU library from {path:?}: {source}")]
    LoadLibrary {
        path: PathBuf,
        #[source]
        source: libloading::Error,
    },
    #[error("missing required GPU symbol `{symbol}`")]
    MissingSymbol { symbol: &'static str },
    #[error("GPU ABI mismatch: expected {expected}, got {observed}")]
    AbiMismatch { expected: u32, observed: u32 },
    #[error("GPU device {api:?}:{device_id} is unavailable")]
    DeviceUnavailable {
        api: crate::DeviceApi,
        device_id: u32,
    },
    #[error("GPU session open failed with status code {status}")]
    SessionOpenFailed { status: i32 },
    #[error("GPU session close failed with status code {status}")]
    SessionCloseFailed { status: i32 },
    #[error("GPU operation `{op}` is unavailable in the loaded backend")]
    UnsupportedOperation { op: &'static str },
    #[error("GPU operation `{op}` failed with status code {status}")]
    OperationFailed { op: &'static str, status: i32 },
}
