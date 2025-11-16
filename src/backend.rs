//! GPU backend abstraction layer
//!
//! Provides unified interface for OpenCL, CUDA, and Metal GPU backends.

use thiserror::Error;
use tracing::{info, warn};

/// GPU backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GPUBackend {
    /// OpenCL backend (cross-platform)
    OpenCL,
    /// CUDA backend (NVIDIA GPUs)
    CUDA,
    /// Metal backend (Apple Silicon)
    Metal,
    /// No GPU available, CPU fallback
    None,
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GPUDevice {
    /// Backend type
    pub backend: GPUBackend,
    /// Device name
    pub name: String,
    /// Total memory in bytes
    pub total_memory: u64,
    /// Available memory in bytes
    pub available_memory: u64,
    /// Compute units (cores)
    pub compute_units: u32,
    /// Maximum work group size
    pub max_work_group_size: usize,
    /// Device vendor
    pub vendor: String,
}

impl GPUDevice {
    /// Create a CPU fallback device
    pub fn cpu_fallback() -> Self {
        Self {
            backend: GPUBackend::None,
            name: "CPU Fallback".to_string(),
            total_memory: 0,
            available_memory: 0,
            compute_units: 0,
            max_work_group_size: 0,
            vendor: "CPU".to_string(),
        }
    }
}

/// GPU buffer handle
#[derive(Debug)]
pub struct GPUBuffer {
    backend: GPUBackend,
    size: usize,
    #[cfg(feature = "opencl")]
    pub(crate) opencl_buffer: Option<ocl::Buffer<u8>>,
    #[cfg(feature = "cuda")]
    pub(crate) cuda_buffer: Option<cudarc::driver::CudaSlice<u8>>,
    #[cfg(feature = "metal-gpu")]
    pub(crate) metal_buffer: Option<metal::Buffer>,
}

impl GPUBuffer {
    /// Get buffer size in bytes
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get backend type
    pub fn backend(&self) -> GPUBackend {
        self.backend
    }
}

/// GPU kernel handle
#[derive(Debug)]
pub struct GPUKernel {
    _backend: GPUBackend,
    _name: String,
    #[cfg(feature = "opencl")]
    opencl_kernel: Option<ocl::Kernel>,
    #[cfg(feature = "cuda")]
    cuda_function: Option<cudarc::driver::CudaFunction>,
    #[cfg(feature = "metal-gpu")]
    metal_function: Option<metal::Function>,
}

/// GPU context for managing device resources
pub struct GPUContext {
    device: GPUDevice,
    #[cfg(feature = "opencl")]
    opencl_context: Option<ocl::Context>,
    #[cfg(feature = "opencl")]
    opencl_queue: Option<ocl::Queue>,
    #[cfg(feature = "cuda")]
    cuda_device: Option<cudarc::driver::CudaDevice>,
    #[cfg(feature = "metal-gpu")]
    metal_device: Option<metal::Device>,
    #[cfg(feature = "metal-gpu")]
    metal_queue: Option<metal::CommandQueue>,
}

impl GPUContext {
    /// Create buffer on GPU
    pub fn create_buffer(&self, _size: usize) -> Result<GPUBuffer, GPUError> {
        match self.device.backend {
            #[cfg(feature = "opencl")]
            GPUBackend::OpenCL => {
                let queue = self.opencl_queue.as_ref()
                    .ok_or(GPUError::BackendNotAvailable)?;
                let buffer = ocl::Buffer::<u8>::builder()
                    .queue(queue.clone())
                    .len(size)
                    .build()
                    .map_err(|e| GPUError::BufferCreationFailed(e.to_string()))?;
                
                Ok(GPUBuffer {
                    backend: GPUBackend::OpenCL,
                    size,
                    opencl_buffer: Some(buffer),
                    #[cfg(feature = "cuda")]
                    cuda_buffer: None,
                    #[cfg(feature = "metal-gpu")]
                    metal_buffer: None,
                })
            }
            #[cfg(feature = "cuda")]
            GPUBackend::CUDA => {
                let device = self.cuda_device.as_ref()
                    .ok_or(GPUError::BackendNotAvailable)?;
                let buffer = device.alloc_zeros::<u8>(size)
                    .map_err(|e| GPUError::BufferCreationFailed(e.to_string()))?;
                
                Ok(GPUBuffer {
                    backend: GPUBackend::CUDA,
                    size,
                    #[cfg(feature = "opencl")]
                    opencl_buffer: None,
                    cuda_buffer: Some(buffer),
                    #[cfg(feature = "metal-gpu")]
                    metal_buffer: None,
                })
            }
            #[cfg(feature = "metal-gpu")]
            GPUBackend::Metal => {
                let device = self.metal_device.as_ref()
                    .ok_or(GPUError::BackendNotAvailable)?;
                let buffer = device.new_buffer(size as u64, metal::MTLResourceOptions::StorageModeShared);
                
                Ok(GPUBuffer {
                    backend: GPUBackend::Metal,
                    size,
                    #[cfg(feature = "opencl")]
                    opencl_buffer: None,
                    #[cfg(feature = "cuda")]
                    cuda_buffer: None,
                    metal_buffer: Some(buffer),
                })
            }
            GPUBackend::None => Err(GPUError::NoGPUAvailable),
            #[allow(unreachable_patterns)]
            _ => Err(GPUError::BackendNotAvailable),
        }
    }

    /// Write data to GPU buffer
    pub fn write_buffer(&self, buffer: &mut GPUBuffer, data: &[u8]) -> Result<(), GPUError> {
        if data.len() > buffer.size {
            return Err(GPUError::BufferTooSmall);
        }

        match buffer.backend {
            #[cfg(feature = "opencl")]
            GPUBackend::OpenCL => {
                if let Some(ref mut ocl_buf) = buffer.opencl_buffer {
                    ocl_buf.write(data)
                        .enq()
                        .map_err(|e| GPUError::TransferFailed(e.to_string()))?;
                    Ok(())
                } else {
                    Err(GPUError::InvalidBuffer)
                }
            }
            #[cfg(feature = "cuda")]
            GPUBackend::CUDA => {
                if let Some(ref cuda_buf) = buffer.cuda_buffer {
                    let device = self.cuda_device.as_ref()
                        .ok_or(GPUError::BackendNotAvailable)?;
                    device.htod_copy_into(data, cuda_buf)
                        .map_err(|e| GPUError::TransferFailed(e.to_string()))?;
                    Ok(())
                } else {
                    Err(GPUError::InvalidBuffer)
                }
            }
            #[cfg(feature = "metal-gpu")]
            GPUBackend::Metal => {
                if let Some(ref metal_buf) = buffer.metal_buffer {
                    let contents = metal_buf.contents() as *mut u8;
                    unsafe {
                        std::ptr::copy_nonoverlapping(data.as_ptr(), contents, data.len());
                    }
                    Ok(())
                } else {
                    Err(GPUError::InvalidBuffer)
                }
            }
            GPUBackend::None => Err(GPUError::NoGPUAvailable),
            #[allow(unreachable_patterns)]
            _ => Err(GPUError::BackendNotAvailable),
        }
    }

    /// Read data from GPU buffer
    pub fn read_buffer(&self, buffer: &GPUBuffer, data: &mut [u8]) -> Result<(), GPUError> {
        if data.len() > buffer.size {
            return Err(GPUError::BufferTooSmall);
        }

        match buffer.backend {
            #[cfg(feature = "opencl")]
            GPUBackend::OpenCL => {
                if let Some(ref ocl_buf) = buffer.opencl_buffer {
                    ocl_buf.read(data)
                        .enq()
                        .map_err(|e| GPUError::TransferFailed(e.to_string()))?;
                    Ok(())
                } else {
                    Err(GPUError::InvalidBuffer)
                }
            }
            #[cfg(feature = "cuda")]
            GPUBackend::CUDA => {
                if let Some(ref cuda_buf) = buffer.cuda_buffer {
                    let device = self.cuda_device.as_ref()
                        .ok_or(GPUError::BackendNotAvailable)?;
                    device.dtoh_sync_copy_into(cuda_buf, data)
                        .map_err(|e| GPUError::TransferFailed(e.to_string()))?;
                    Ok(())
                } else {
                    Err(GPUError::InvalidBuffer)
                }
            }
            #[cfg(feature = "metal-gpu")]
            GPUBackend::Metal => {
                if let Some(ref metal_buf) = buffer.metal_buffer {
                    let contents = metal_buf.contents() as *const u8;
                    unsafe {
                        std::ptr::copy_nonoverlapping(contents, data.as_mut_ptr(), data.len());
                    }
                    Ok(())
                } else {
                    Err(GPUError::InvalidBuffer)
                }
            }
            GPUBackend::None => Err(GPUError::NoGPUAvailable),
            #[allow(unreachable_patterns)]
            _ => Err(GPUError::BackendNotAvailable),
        }
    }

    /// Get device information
    pub fn device(&self) -> &GPUDevice {
        &self.device
    }
}

/// Main GPU accelerator interface
pub struct GPUAccelerator {
    context: GPUContext,
}

impl GPUAccelerator {
    /// Auto-detect and initialize best available GPU backend
    pub fn new() -> Self {
        info!("Detecting GPU hardware...");

        // Try Metal first (Apple Silicon)
        #[cfg(feature = "metal-gpu")]
        if let Ok(accelerator) = Self::try_metal() {
            info!("Using Metal GPU backend");
            return accelerator;
        }

        // Try CUDA (NVIDIA)
        #[cfg(feature = "cuda")]
        if let Ok(accelerator) = Self::try_cuda() {
            info!("Using CUDA GPU backend");
            return accelerator;
        }

        // Try OpenCL (cross-platform)
        #[cfg(feature = "opencl")]
        if let Ok(accelerator) = Self::try_opencl() {
            info!("Using OpenCL GPU backend");
            return accelerator;
        }

        // Fallback to CPU
        warn!("No GPU available, falling back to CPU");
        Self::cpu_fallback()
    }

    /// Try to initialize Metal backend
    #[cfg(feature = "metal-gpu")]
    fn try_metal() -> Result<Self, GPUError> {
        let device = metal::Device::system_default()
            .ok_or(GPUError::NoGPUAvailable)?;
        
        let queue = device.new_command_queue();
        
        let gpu_device = GPUDevice {
            backend: GPUBackend::Metal,
            name: device.name().to_string(),
            total_memory: device.recommended_max_working_set_size(),
            available_memory: device.recommended_max_working_set_size(),
            compute_units: 0, // Metal doesn't expose this directly
            max_work_group_size: device.max_threads_per_threadgroup().width,
            vendor: "Apple".to_string(),
        };

        Ok(Self {
            context: GPUContext {
                device: gpu_device,
                #[cfg(feature = "opencl")]
                opencl_context: None,
                #[cfg(feature = "opencl")]
                opencl_queue: None,
                #[cfg(feature = "cuda")]
                cuda_device: None,
                metal_device: Some(device),
                metal_queue: Some(queue),
            },
        })
    }

    /// Try to initialize CUDA backend
    #[cfg(feature = "cuda")]
    fn try_cuda() -> Result<Self, GPUError> {
        use cudarc::driver::CudaDevice;
        
        let device = CudaDevice::new(0)
            .map_err(|e| GPUError::InitializationFailed(e.to_string()))?;
        
        let name = device.name()
            .map_err(|e| GPUError::InitializationFailed(e.to_string()))?;
        let total_memory = device.total_memory()
            .map_err(|e| GPUError::InitializationFailed(e.to_string()))?;
        
        let gpu_device = GPUDevice {
            backend: GPUBackend::CUDA,
            name,
            total_memory: total_memory as u64,
            available_memory: total_memory as u64,
            compute_units: 0, // Would need to query device properties
            max_work_group_size: 1024, // Common CUDA limit
            vendor: "NVIDIA".to_string(),
        };

        Ok(Self {
            context: GPUContext {
                device: gpu_device,
                #[cfg(feature = "opencl")]
                opencl_context: None,
                #[cfg(feature = "opencl")]
                opencl_queue: None,
                cuda_device: Some(device),
                #[cfg(feature = "metal-gpu")]
                metal_device: None,
                #[cfg(feature = "metal-gpu")]
                metal_queue: None,
            },
        })
    }

    /// Try to initialize OpenCL backend
    #[cfg(feature = "opencl")]
    fn try_opencl() -> Result<Self, GPUError> {
        let platform = ocl::Platform::default();
        let device = ocl::Device::first(platform)
            .map_err(|e| GPUError::InitializationFailed(e.to_string()))?;
        
        let context = ocl::Context::builder()
            .platform(platform)
            .devices(device)
            .build()
            .map_err(|e| GPUError::InitializationFailed(e.to_string()))?;
        
        let queue = ocl::Queue::new(&context, device, None)
            .map_err(|e| GPUError::InitializationFailed(e.to_string()))?;
        
        let name = device.name()
            .map_err(|e| GPUError::InitializationFailed(e.to_string()))?;
        let total_memory = device.mem_size()
            .map_err(|e| GPUError::InitializationFailed(e.to_string()))?;
        let compute_units = device.max_compute_units()
            .map_err(|e| GPUError::InitializationFailed(e.to_string()))?;
        let max_work_group_size = device.max_wg_size()
            .map_err(|e| GPUError::InitializationFailed(e.to_string()))?;
        let vendor = device.vendor()
            .map_err(|e| GPUError::InitializationFailed(e.to_string()))?;
        
        let gpu_device = GPUDevice {
            backend: GPUBackend::OpenCL,
            name,
            total_memory,
            available_memory: total_memory,
            compute_units,
            max_work_group_size,
            vendor,
        };

        Ok(Self {
            context: GPUContext {
                device: gpu_device,
                opencl_context: Some(context),
                opencl_queue: Some(queue),
                #[cfg(feature = "cuda")]
                cuda_device: None,
                #[cfg(feature = "metal-gpu")]
                metal_device: None,
                #[cfg(feature = "metal-gpu")]
                metal_queue: None,
            },
        })
    }

    /// Create CPU fallback accelerator
    pub fn cpu_fallback() -> Self {
        Self {
            context: GPUContext {
                device: GPUDevice::cpu_fallback(),
                #[cfg(feature = "opencl")]
                opencl_context: None,
                #[cfg(feature = "opencl")]
                opencl_queue: None,
                #[cfg(feature = "cuda")]
                cuda_device: None,
                #[cfg(feature = "metal-gpu")]
                metal_device: None,
                #[cfg(feature = "metal-gpu")]
                metal_queue: None,
            },
        }
    }

    /// Get GPU context
    pub fn context(&self) -> &GPUContext {
        &self.context
    }

    /// Get device information
    pub fn device(&self) -> &GPUDevice {
        &self.context.device
    }

    /// Check if GPU is available
    pub fn is_gpu_available(&self) -> bool {
        self.context.device.backend != GPUBackend::None
    }

    /// Get backend type
    pub fn backend(&self) -> GPUBackend {
        self.context.device.backend
    }
}

impl Default for GPUAccelerator {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU-related errors
#[derive(Debug, Error)]
pub enum GPUError {
    /// No GPU available
    #[error("No GPU available")]
    NoGPUAvailable,
    
    /// Backend not available
    #[error("GPU backend not available")]
    BackendNotAvailable,
    
    /// Initialization failed
    #[error("GPU initialization failed: {0}")]
    InitializationFailed(String),
    
    /// Buffer creation failed
    #[error("Buffer creation failed: {0}")]
    BufferCreationFailed(String),
    
    /// Data transfer failed
    #[error("Data transfer failed: {0}")]
    TransferFailed(String),
    
    /// Kernel compilation failed
    #[error("Kernel compilation failed: {0}")]
    KernelCompilationFailed(String),
    
    /// Kernel execution failed
    #[error("Kernel execution failed: {0}")]
    KernelExecutionFailed(String),
    
    /// Invalid buffer
    #[error("Invalid buffer")]
    InvalidBuffer,
    
    /// Buffer too small
    #[error("Buffer too small")]
    BufferTooSmall,
    
    /// Unsupported operation
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
}
