use criterion::{black_box, criterion_group, criterion_main, Criterion};
use silver_gpu::{GpuConfig, GpuBackend};

fn benchmark_gpu_config(c: &mut Criterion) {
    c.bench_function("gpu_config_creation", |b| {
        b.iter(|| {
            GpuConfig::default()
                .with_backend(black_box(GpuBackend::Cpu))
                .with_threads(black_box(256))
        })
    });
}

criterion_group!(benches, benchmark_gpu_config);
criterion_main!(benches);
