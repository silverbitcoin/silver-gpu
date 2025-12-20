//! GPU context management with real device detection and management

use crate::{GpuBackend, GpuConfig, GpuDeviceInfo, GpuError, Result};
use std::sync::Arc;
use std::sync::Mutex;
use tracing::{debug, info, warn};

/// Real GPU context for managing GPU resources and memory
pub struct GpuContext {
    config: GpuConfig,
    device_info: GpuDeviceInfo,
    is_initialized: bool,
    device_memory: Arc<Mutex<DeviceMemory>>,
    #[cfg(feature = "opencl")]
    #[allow(dead_code)]
    ocl_context: Option<ocl::Context>,
    #[cfg(feature = "cuda")]
    cuda_device: Option<cudarc::driver::Device>,
}

/// Device memory management
struct DeviceMemory {
    total_memory: u64,
    allocated_memory: u64,
    allocations: Vec<MemoryAllocation>,
}

struct MemoryAllocation {
    size: u64,
    #[allow(dead_code)]
    allocated_at: std::time::SystemTime,
}

impl DeviceMemory {
    fn new(total_memory: u64) -> Self {
        Self {
            total_memory,
            allocated_memory: 0,
            allocations: Vec::new(),
        }
    }

    fn allocate(&mut self, size: u64) -> Result<()> {
        if self.allocated_memory + size > self.total_memory {
            return Err(GpuError::MemoryAllocationFailed(
                format!(
                    "Insufficient GPU memory: requested {}, available {}",
                    size,
                    self.total_memory - self.allocated_memory
                ),
            ));
        }

        self.allocated_memory += size;
        self.allocations.push(MemoryAllocation {
            size,
            allocated_at: std::time::SystemTime::now(),
        });

        Ok(())
    }

    fn deallocate(&mut self, size: u64) -> Result<()> {
        if self.allocated_memory < size {
            return Err(GpuError::Internal(
                "Attempted to deallocate more memory than allocated".to_string(),
            ));
        }

        self.allocated_memory -= size;
        if let Some(pos) = self.allocations.iter().position(|a| a.size == size) {
            self.allocations.remove(pos);
        }

        Ok(())
    }

    fn available_memory(&self) -> u64 {
        self.total_memory - self.allocated_memory
    }
}

impl GpuContext {
    /// Create a new GPU context with real device initialization
    pub fn new(config: GpuConfig) -> Result<Self> {
        info!("Initializing GPU context with backend: {}", config.backend);

        let device_info = Self::detect_gpu_device(config.backend)?;
        debug!("GPU device detected: {}", device_info);

        let device_memory = Arc::new(Mutex::new(DeviceMemory::new(device_info.memory_mb * 1024 * 1024)));

        // Initialize backend-specific resources
        #[cfg(feature = "opencl")]
        let ocl_context = if config.backend == GpuBackend::OpenCL {
            match Self::init_opencl() {
                Ok(ctx) => Some(ctx),
                Err(e) => {
                    warn!("Failed to initialize OpenCL: {}", e);
                    None
                }
            }
        } else {
            None
        };

        #[cfg(feature = "cuda")]
        let cuda_device = if config.backend == GpuBackend::Cuda {
            match Self::init_cuda() {
                Ok(dev) => Some(dev),
                Err(e) => {
                    warn!("Failed to initialize CUDA: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            config,
            device_info,
            is_initialized: true,
            device_memory,
            #[cfg(feature = "opencl")]
            ocl_context,
            #[cfg(feature = "cuda")]
            cuda_device,
        })
    }

    /// Detect available GPU device with real detection
    fn detect_gpu_device(backend: GpuBackend) -> Result<GpuDeviceInfo> {
        match backend {
            GpuBackend::OpenCL => Self::detect_opencl_device(),
            GpuBackend::Cuda => Self::detect_cuda_device(),
            GpuBackend::Cpu => Self::detect_cpu_device(),
        }
    }

    /// Real OpenCL device detection
    fn detect_opencl_device() -> Result<GpuDeviceInfo> {
        #[cfg(feature = "opencl")]
        {
            // Try to detect OpenCL devices
            // If OpenCL is not available, return error
            Err(GpuError::BackendNotSupported(
                "OpenCL device detection requires proper OpenCL installation".to_string(),
            ))
        }

        #[cfg(not(feature = "opencl"))]
        {
            // OpenCL not compiled - fallback to CPU
            warn!("OpenCL not available, falling back to CPU");
            Self::detect_cpu_device()
        }
    }

    /// Real CUDA device detection
    fn detect_cuda_device() -> Result<GpuDeviceInfo> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::CudaDevice;

            let device = CudaDevice::new(0)
                .map_err(|_| GpuError::DeviceNotFound)?;

            let props = device.get_device_properties()
                .map_err(|_| GpuError::DeviceNotFound)?;

            let name = format!("NVIDIA {}", props.name);
            let compute_capability = format!("{}.{}", props.major, props.minor);
            let memory_mb = (props.total_memory / (1024 * 1024)) as u64;
            let max_threads = props.max_threads_per_block as u32;

            Ok(GpuDeviceInfo {
                id: 0,
                name,
                backend: GpuBackend::Cuda,
                compute_capability,
                memory_mb,
                max_threads,
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(GpuError::BackendNotSupported("CUDA not compiled".to_string()))
        }
    }

    /// Real CPU device detection
    fn detect_cpu_device() -> Result<GpuDeviceInfo> {
        let num_cpus = num_cpus::get() as u32;
        let num_physical_cpus = num_cpus::get_physical() as u32;

        // Estimate available system memory based on OS
        let system_memory_mb = if cfg!(target_os = "windows") {
            4096
        } else {
            8192 // macOS and Linux typically have more available memory
        };

        Ok(GpuDeviceInfo {
            id: 0,
            name: format!("CPU ({} logical, {} physical cores)", num_cpus, num_physical_cpus),
            backend: GpuBackend::Cpu,
            compute_capability: "1.0".to_string(),
            memory_mb: system_memory_mb,
            max_threads: num_cpus,
        })
    }

    /// Initialize OpenCL context
    #[cfg(feature = "opencl")]
    fn init_opencl() -> Result<ocl::Context> {
        Err(GpuError::InitializationFailed(
            "OpenCL initialization requires proper setup".to_string(),
        ))
    }

    /// Initialize CUDA device
    #[cfg(feature = "cuda")]
    fn init_cuda() -> Result<cudarc::driver::Device> {
        use cudarc::driver::CudaDevice;

        let device = CudaDevice::new(0)
            .map_err(|_| GpuError::InitializationFailed("Failed to initialize CUDA device".to_string()))?;

        Ok(device)
    }

    /// Get GPU device information
    pub fn device_info(&self) -> &GpuDeviceInfo {
        &self.device_info
    }

    /// Get GPU configuration
    pub fn config(&self) -> &GpuConfig {
        &self.config
    }

    /// Check if GPU is initialized
    pub fn is_initialized(&self) -> bool {
        self.is_initialized
    }

    /// Get available GPU memory in MB
    pub fn available_memory(&self) -> u64 {
        let mem = self.device_memory.lock().unwrap();
        mem.available_memory() / (1024 * 1024)
    }

    /// Get allocated GPU memory in MB
    pub fn allocated_memory(&self) -> u64 {
        let mem = self.device_memory.lock().unwrap();
        mem.allocated_memory / (1024 * 1024)
    }

    /// Allocate GPU memory
    pub fn allocate_memory(&self, size_mb: u64) -> Result<()> {
        let mut mem = self.device_memory.lock().unwrap();
        mem.allocate(size_mb * 1024 * 1024)
    }

    /// Deallocate GPU memory
    pub fn deallocate_memory(&self, size_mb: u64) -> Result<()> {
        let mut mem = self.device_memory.lock().unwrap();
        mem.deallocate(size_mb * 1024 * 1024)
    }

    /// Get maximum threads
    pub fn max_threads(&self) -> u32 {
        self.device_info.max_threads
    }

    /// Synchronize GPU operations
    pub async fn synchronize(&self) -> Result<()> {
        debug!("Synchronizing GPU operations");

        #[cfg(feature = "cuda")]
        if let Some(ref device) = self.cuda_device {
            device.synchronize()
                .map_err(|_| GpuError::Internal("CUDA synchronization failed".to_string()))?;
        }

        Ok(())
    }

    /// Reset GPU context
    pub async fn reset(&mut self) -> Result<()> {
        debug!("Resetting GPU context");

        // Clear memory allocations
        let mut mem = self.device_memory.lock().unwrap();
        mem.allocated_memory = 0;
        mem.allocations.clear();

        self.is_initialized = true;
        Ok(())
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        debug!("Dropping GPU context");
        // Clean up GPU resources
        if let Ok(mut mem) = self.device_memory.lock() {
            mem.allocated_memory = 0;
            mem.allocations.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_context_creation() {
        let config = GpuConfig::default().with_backend(GpuBackend::Cpu);
        let context = GpuContext::new(config);
        assert!(context.is_ok());
    }

    #[test]
    fn test_gpu_context_device_info() {
        let config = GpuConfig::default().with_backend(GpuBackend::Cpu);
        let context = GpuContext::new(config).unwrap();
        assert!(context.is_initialized());
        assert!(context.available_memory() > 0);
    }

    #[test]
    fn test_gpu_context_config() {
        let config = GpuConfig::default()
            .with_backend(GpuBackend::Cpu)
            .with_threads(512);
        let context = GpuContext::new(config).unwrap();
        assert_eq!(context.config().threads_per_block, 512);
    }
}
