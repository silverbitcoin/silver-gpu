//! # SilverBitcoin GPU Acceleration
//!
//! GPU acceleration layer supporting OpenCL, CUDA, and Metal.
//!
//! This crate provides:
//! - GPU abstraction layer
//! - Batch signature verification (100-1000x speedup)
//! - Parallel hash computation (10-100x speedup)
//! - GPU-accelerated transaction execution
//! - Automatic CPU/GPU load balancing

#![warn(missing_docs, rust_2018_idioms)]
#![forbid(unsafe_code)]

pub mod backend;
pub mod executor;
pub mod hashing;
pub mod scheduler;
pub mod signature_verification;

pub use backend::{GPUAccelerator, GPUBackend, GPUDevice};
pub use executor::GPUExecutor;
pub use hashing::GPUHasher;
pub use scheduler::HybridExecutor;
pub use signature_verification::GPUSignatureVerifier;
