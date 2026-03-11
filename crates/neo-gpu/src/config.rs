use std::path::PathBuf;

#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeviceApi {
    Auto = 4,
    #[default]
    Cpu = 0,
    Metal = 1,
    Cuda = 2,
    Hip = 3,
}

impl DeviceApi {
    #[inline]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    #[inline]
    pub fn candidate_order(self) -> &'static [DeviceApi] {
        const CPU_ONLY: &[DeviceApi] = &[DeviceApi::Cpu];
        const METAL_ONLY: &[DeviceApi] = &[DeviceApi::Metal];
        const CUDA_ONLY: &[DeviceApi] = &[DeviceApi::Cuda];
        const HIP_ONLY: &[DeviceApi] = &[DeviceApi::Hip];
        #[cfg(target_os = "macos")]
        const AUTO_ORDER: &[DeviceApi] = &[DeviceApi::Metal, DeviceApi::Cpu];
        #[cfg(not(target_os = "macos"))]
        const AUTO_ORDER: &[DeviceApi] = &[DeviceApi::Cuda, DeviceApi::Hip, DeviceApi::Cpu];

        match self {
            DeviceApi::Auto => AUTO_ORDER,
            DeviceApi::Cpu => CPU_ONLY,
            DeviceApi::Metal => METAL_ONLY,
            DeviceApi::Cuda => CUDA_ONLY,
            DeviceApi::Hip => HIP_ONLY,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MojoBackendConfig {
    pub library_path: Option<PathBuf>,
    pub device_api: DeviceApi,
    pub device_id: u32,
    pub fallback_to_cpu: bool,
}

impl MojoBackendConfig {
    #[inline]
    pub fn auto() -> Self {
        Self {
            library_path: None,
            device_api: DeviceApi::Auto,
            device_id: 0,
            fallback_to_cpu: true,
        }
    }

    #[inline]
    pub fn new(device_api: DeviceApi) -> Self {
        Self {
            library_path: None,
            device_api,
            device_id: 0,
            fallback_to_cpu: false,
        }
    }

    #[inline]
    pub fn with_library_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.library_path = Some(path.into());
        self
    }

    #[inline]
    pub fn with_device_id(mut self, device_id: u32) -> Self {
        self.device_id = device_id;
        self
    }

    #[inline]
    pub fn with_device_api(mut self, device_api: DeviceApi) -> Self {
        self.device_api = device_api;
        self
    }

    #[inline]
    pub fn allow_cpu_fallback(mut self) -> Self {
        self.fallback_to_cpu = true;
        self
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ProverComputeBackend {
    #[default]
    Cpu,
    Mojo(MojoBackendConfig),
}

impl ProverComputeBackend {
    #[inline]
    pub fn auto() -> Self {
        Self::Mojo(MojoBackendConfig::auto())
    }
}
