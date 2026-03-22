use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, ValueEnum};
use naga::{
    back::spv::{Options as SpvOptions, PipelineOptions as SpvPipelineOptions, write_vec},
    front::wgsl,
    valid::{Capabilities, ValidationFlags, Validator},
};
use serde::Serialize;
use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use vulkano::{
    Version, VulkanLibrary,
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
        physical::PhysicalDevice,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
        compute::{ComputePipeline, ComputePipelineCreateInfo},
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    query::{QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType},
    shader::{ShaderModule, ShaderModuleCreateInfo},
    sync::{self, GpuFuture, PipelineStage},
};

const LOCAL_SIZE_X: u32 = 64;
const ALU_SHADER_WGSL: &str = r#"
struct Data {
    values: array<u32>,
}

@group(0) @binding(0)
var<storage, read_write> data_buffer: Data;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let len = arrayLength(&data_buffer.values);
    if (index >= len) {
        return;
    }

    var value = data_buffer.values[index] ^ (index * 1664525u + 1013904223u);

    for (var i: u32 = 0u; i < 256u; i = i + 1u) {
        value = value * 1664525u + 1013904223u;
        value = value ^ (value >> 13u);
        value = value + ((i + index) * 17u);
    }

    data_buffer.values[index] = value;
}
"#;

const MEMORY_SHADER_WGSL: &str = r#"
struct Data {
    values: array<u32>,
}

@group(0) @binding(0)
var<storage, read_write> data_buffer: Data;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let len = arrayLength(&data_buffer.values);
    if (index >= len || len == 0u) {
        return;
    }

    var value = data_buffer.values[index];

    for (var i: u32 = 0u; i < 128u; i = i + 1u) {
        let probe = (index * 17u + i * 131u) % len;
        value = value + (data_buffer.values[probe] ^ (probe + i));
        value = (value << 7u) | (value >> 25u);
    }

    data_buffer.values[index] = value ^ index;
}
"#;

const MIXED_SHADER_WGSL: &str = r#"
struct Data {
    values: array<u32>,
}

@group(0) @binding(0)
var<storage, read_write> data_buffer: Data;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let len = arrayLength(&data_buffer.values);
    if (index >= len || len == 0u) {
        return;
    }

    var value = data_buffer.values[index];

    for (var i: u32 = 0u; i < 192u; i = i + 1u) {
        let probe = (index + i * 97u) % len;
        let neighbor = data_buffer.values[probe];

        if (((neighbor ^ i) & 1u) == 0u) {
            value = value ^ (neighbor + i * 41u);
            value = (value << 3u) | (value >> 29u);
        } else {
            value = value + ((neighbor ^ 0x9E3779B9u) + index);
            value = value ^ (value >> 11u);
        }
    }

    data_buffer.values[index] = value;
}
"#;

#[derive(Clone, Debug, Parser, Serialize)]
#[command(author, version, about = "Microbenchmark de compute em Vulkan/Vulkano")]
struct Cli {
    #[arg(long, value_enum, default_value_t = ShaderKind::Mixed)]
    shader: ShaderKind,

    #[arg(long, default_value_t = 4 * 1024 * 1024)]
    buffer_bytes: usize,

    #[arg(long)]
    dispatch_x: Option<u32>,

    #[arg(long, default_value_t = 1)]
    dispatch_y: u32,

    #[arg(long, default_value_t = 1)]
    dispatch_z: u32,

    #[arg(long, default_value_t = 2)]
    warmup_runs: u32,

    #[arg(long, default_value_t = 10)]
    measured_runs: u32,

    #[arg(long, default_value_t = 1)]
    dispatches_per_run: u32,

    #[arg(long)]
    recreate_pipeline: bool,

    #[arg(long)]
    rerecord_cmd: bool,

    #[arg(long)]
    use_barrier: bool,

    #[arg(long, default_value_t = 0)]
    device_index: usize,

    #[arg(long)]
    list_devices: bool,

    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Serialize, ValueEnum)]
enum ShaderKind {
    Alu,
    Memory,
    #[default]
    Mixed,
}

impl ShaderKind {
    fn description(self) -> &'static str {
        match self {
            ShaderKind::Alu => "ALU-bound",
            ShaderKind::Memory => "memory-bound",
            ShaderKind::Mixed => "mixed/branchy",
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct BenchmarkConfig {
    shader: ShaderKind,
    shader_profile: &'static str,
    buffer_bytes: usize,
    element_count: usize,
    local_size_x: u32,
    dispatch_x: u32,
    dispatch_y: u32,
    dispatch_z: u32,
    warmup_runs: u32,
    measured_runs: u32,
    dispatches_per_run: u32,
    recreate_pipeline: bool,
    rerecord_cmd: bool,
    use_barrier: bool,
}

#[derive(Clone, Debug, Serialize)]
struct DeviceMetadata {
    physical_device_index: usize,
    device_name: String,
    device_type: String,
    vendor_id: u32,
    device_id: u32,
    driver_version: String,
    api_version: String,
    queue_family_index: u32,
    queue_flags: String,
    timestamp_valid_bits: Option<u32>,
    timestamp_period_ns: Option<f32>,
}

#[derive(Clone, Debug, Serialize)]
struct BenchmarkSample {
    run_index: u32,
    pipeline_build_ns: u128,
    record_ns: u128,
    submit_ns: u128,
    wait_ns: u128,
    gpu_kernel_ns: Option<f64>,
    checksum: u64,
}

#[derive(Clone, Debug, Serialize)]
struct BenchmarkSummary {
    samples: usize,
    avg_pipeline_build_ns: f64,
    avg_record_ns: f64,
    avg_submit_ns: f64,
    avg_wait_ns: f64,
    avg_gpu_kernel_ns: Option<f64>,
    min_gpu_kernel_ns: Option<f64>,
    max_gpu_kernel_ns: Option<f64>,
    final_checksum: u64,
}

#[derive(Clone, Debug, Serialize)]
struct BenchmarkReport {
    config: BenchmarkConfig,
    device: DeviceMetadata,
    initial_pipeline_build_ns: u128,
    initial_record_ns: u128,
    samples: Vec<BenchmarkSample>,
    summary: BenchmarkSummary,
}

struct AppContext {
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    data_buffer: Subbuffer<[u32]>,
    query_pool: Option<Arc<QueryPool>>,
    timestamp_period_ns: Option<f32>,
    device_metadata: DeviceMetadata,
}

#[derive(Clone)]
struct PreparedResources {
    pipeline: Arc<ComputePipeline>,
    descriptor_set: Arc<DescriptorSet>,
}

fn main() -> Result<()> {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| run_main())
        .unwrap()
        .join()
        .unwrap()
}

fn run_main() -> Result<()> {
    let cli = Cli::parse();
    let library = VulkanLibrary::new().context("falha ao carregar o loader Vulkan")?;
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            max_api_version: Some(Version::V1_0),
            ..Default::default()
        },
    )
    .context("falha ao criar a instância Vulkan")?;

    let physical_devices = instance
        .enumerate_physical_devices()
        .context("falha ao enumerar GPUs Vulkan")?
        .collect::<Vec<_>>();

    if physical_devices.is_empty() {
        bail!("nenhuma GPU Vulkan encontrada");
    }

    if cli.list_devices {
        for (index, physical_device) in physical_devices.iter().enumerate() {
            let props = physical_device.properties();
            println!(
                "[{index}] {} ({:?}) vendor={} device={} api={:?}",
                props.device_name,
                props.device_type,
                props.vendor_id,
                props.device_id,
                props.api_version,
            );
        }
        return Ok(());
    }

    let config = build_config(&cli)?;
    let app = create_app_context(&physical_devices, cli.device_index, config.clone())?;
    let report = run_benchmark(&app, config)?;

    println!(
        "device={} queue_family={} shader={} dispatch=({}, {}, {}) runs={} gpu_timestamps={}",
        report.device.device_name,
        report.device.queue_family_index,
        report.config.shader.description(),
        report.config.dispatch_x,
        report.config.dispatch_y,
        report.config.dispatch_z,
        report.samples.len(),
        report.device.timestamp_period_ns.is_some(),
    );
    println!(
        "avg record={:.0} ns submit={:.0} ns wait={:.0} ns gpu={}",
        report.summary.avg_record_ns,
        report.summary.avg_submit_ns,
        report.summary.avg_wait_ns,
        report
            .summary
            .avg_gpu_kernel_ns
            .map(|value| format!("{value:.0} ns"))
            .unwrap_or_else(|| "n/a".to_string()),
    );

    if let Some(path) = cli.output.as_deref() {
        export_report(path, &report)?;
        println!("resultado salvo em {}", path.display());
    }

    Ok(())
}

fn build_config(cli: &Cli) -> Result<BenchmarkConfig> {
    if cli.buffer_bytes < std::mem::size_of::<u32>() {
        bail!(
            "buffer_bytes deve ser pelo menos {}",
            std::mem::size_of::<u32>()
        );
    }

    let element_count = cli.buffer_bytes / std::mem::size_of::<u32>();
    let dispatch_x = cli
        .dispatch_x
        .unwrap_or_else(|| (element_count as u32).div_ceil(LOCAL_SIZE_X));

    Ok(BenchmarkConfig {
        shader: cli.shader,
        shader_profile: cli.shader.description(),
        buffer_bytes: cli.buffer_bytes,
        element_count,
        local_size_x: LOCAL_SIZE_X,
        dispatch_x,
        dispatch_y: cli.dispatch_y,
        dispatch_z: cli.dispatch_z,
        warmup_runs: cli.warmup_runs,
        measured_runs: cli.measured_runs,
        dispatches_per_run: cli.dispatches_per_run,
        recreate_pipeline: cli.recreate_pipeline,
        rerecord_cmd: cli.rerecord_cmd || cli.recreate_pipeline,
        use_barrier: cli.use_barrier,
    })
}

fn create_app_context(
    physical_devices: &[Arc<PhysicalDevice>],
    device_index: usize,
    config: BenchmarkConfig,
) -> Result<AppContext> {
    let physical_device = physical_devices
        .get(device_index)
        .cloned()
        .ok_or_else(|| anyhow!("device_index {} inválido", device_index))?;

    let queue_family_index = select_queue_family(&physical_device)
        .ok_or_else(|| anyhow!("nenhuma queue family com suporte a compute foi encontrada"))?;

    let supported_extensions = physical_device.supported_extensions();
    let device_extensions = DeviceExtensions {
        khr_portability_subset: supported_extensions.khr_portability_subset,
        khr_storage_buffer_storage_class: supported_extensions.khr_storage_buffer_storage_class,
        khr_maintenance3: supported_extensions.khr_maintenance3,
        ..DeviceExtensions::empty()
    };

    let (device, mut queues) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .context("falha ao criar o device lógico Vulkan")?;

    let queue = queues
        .next()
        .context("falha ao obter a queue de compute criada")?;

    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    ));
    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let data_buffer = create_data_buffer(memory_allocator.clone(), config.element_count)?;

    let queue_family_properties =
        &physical_device.queue_family_properties()[queue_family_index as usize];
    let timestamp_valid_bits = queue_family_properties.timestamp_valid_bits;
    let timestamp_period_ns = if timestamp_valid_bits.is_some()
        && physical_device.properties().timestamp_compute_and_graphics
    {
        Some(physical_device.properties().timestamp_period)
    } else {
        None
    };
    let query_pool = if timestamp_valid_bits.is_some() {
        Some(
            QueryPool::new(
                device.clone(),
                QueryPoolCreateInfo {
                    query_count: 2,
                    ..QueryPoolCreateInfo::query_type(QueryType::Timestamp)
                },
            )
            .context("falha ao criar QueryPool de timestamps")?,
        )
    } else {
        None
    };

    let props = physical_device.properties();
    let device_metadata = DeviceMetadata {
        physical_device_index: device_index,
        device_name: props.device_name.clone(),
        device_type: format!("{:?}", props.device_type),
        vendor_id: props.vendor_id,
        device_id: props.device_id,
        driver_version: format!("{:?}", props.driver_version),
        api_version: format!("{:?}", props.api_version),
        queue_family_index,
        queue_flags: format!("{:?}", queue_family_properties.queue_flags),
        timestamp_valid_bits,
        timestamp_period_ns,
    };

    Ok(AppContext {
        device,
        queue,
        command_buffer_allocator,
        descriptor_set_allocator,
        data_buffer,
        query_pool,
        timestamp_period_ns,
        device_metadata,
    })
}

fn select_queue_family(physical_device: &Arc<PhysicalDevice>) -> Option<u32> {
    let mut best_compute_with_timestamps = None;
    let mut first_compute = None;

    for (index, props) in physical_device.queue_family_properties().iter().enumerate() {
        if !props.queue_flags.intersects(QueueFlags::COMPUTE) {
            continue;
        }

        let index = index as u32;
        if first_compute.is_none() {
            first_compute = Some(index);
        }
        if props.timestamp_valid_bits.is_some() {
            best_compute_with_timestamps = Some(index);
            if !props.queue_flags.intersects(QueueFlags::GRAPHICS) {
                break;
            }
        }
    }

    best_compute_with_timestamps.or(first_compute)
}

fn create_data_buffer(
    memory_allocator: Arc<StandardMemoryAllocator>,
    element_count: usize,
) -> Result<Subbuffer<[u32]>> {
    let seed_data = (0..element_count)
        .map(|index| index as u32 ^ 0xA5A5_5A5A)
        .collect::<Vec<_>>();

    Buffer::from_iter(
        memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        seed_data,
    )
    .context("falha ao criar o buffer de dados")
}

fn run_benchmark(app: &AppContext, config: BenchmarkConfig) -> Result<BenchmarkReport> {
    let mut initial_pipeline_build_ns = 0;
    let mut initial_record_ns = 0;
    let mut shared_resources = None;
    let mut shared_command_buffers = None;

    if !config.recreate_pipeline {
        let start = Instant::now();
        let resources = prepare_resources(app, config.shader)?;
        initial_pipeline_build_ns = start.elapsed().as_nanos();

        if !config.rerecord_cmd {
            let record_start = Instant::now();
            let command_buffers = build_command_buffers(app, &config, &resources, true)?;
            initial_record_ns = record_start.elapsed().as_nanos();
            shared_command_buffers = Some(command_buffers);
        }

        shared_resources = Some(resources);
    }

    for _ in 0..config.warmup_runs {
        let warmup_resources = if config.recreate_pipeline {
            Some(prepare_resources(app, config.shader)?)
        } else {
            None
        };
        let resources = warmup_resources
            .as_ref()
            .or(shared_resources.as_ref())
            .context("pipeline de warmup não disponível")?;
        let command_buffers = if config.rerecord_cmd || config.recreate_pipeline {
            build_command_buffers(app, &config, resources, false)?
        } else {
            shared_command_buffers
                .as_ref()
                .cloned()
                .context("command buffers compartilhados indisponíveis")?
        };

        submit_and_measure(app, &command_buffers)?;
    }

    let mut samples = Vec::with_capacity(config.measured_runs as usize);

    for run_index in 0..config.measured_runs {
        let (resources, pipeline_build_ns) = if config.recreate_pipeline {
            let start = Instant::now();
            let resources = prepare_resources(app, config.shader)?;
            (resources, start.elapsed().as_nanos())
        } else {
            (
                shared_resources
                    .as_ref()
                    .cloned()
                    .context("pipeline compartilhado indisponível")?,
                0,
            )
        };

        let (command_buffers, record_ns) = if config.rerecord_cmd || config.recreate_pipeline {
            let start = Instant::now();
            let command_buffers = build_command_buffers(app, &config, &resources, false)?;
            (command_buffers, start.elapsed().as_nanos())
        } else {
            (
                shared_command_buffers
                    .as_ref()
                    .cloned()
                    .context("command buffers compartilhados indisponíveis")?,
                0,
            )
        };

        let timing = submit_and_measure(app, &command_buffers)?;
        let checksum = checksum_buffer(&app.data_buffer)?;

        samples.push(BenchmarkSample {
            run_index,
            pipeline_build_ns,
            record_ns,
            submit_ns: timing.submit_ns,
            wait_ns: timing.wait_ns,
            gpu_kernel_ns: timing.gpu_kernel_ns,
            checksum,
        });
    }

    let summary = summarize(&samples);

    Ok(BenchmarkReport {
        config,
        device: app.device_metadata.clone(),
        initial_pipeline_build_ns,
        initial_record_ns,
        samples,
        summary,
    })
}

fn prepare_resources(app: &AppContext, shader_kind: ShaderKind) -> Result<PreparedResources> {
    let pipeline = create_pipeline(app.device.clone(), shader_kind)?;
    let set_layout = pipeline
        .layout()
        .set_layouts()
        .first()
        .cloned()
        .context("o pipeline de compute não expôs descriptor set layout")?;
    let descriptor_set = DescriptorSet::new(
        app.descriptor_set_allocator.clone(),
        set_layout,
        [WriteDescriptorSet::buffer(0, app.data_buffer.clone())],
        [],
    )
    .context("falha ao criar o descriptor set")?;

    Ok(PreparedResources {
        pipeline,
        descriptor_set,
    })
}

fn create_pipeline(device: Arc<Device>, shader_kind: ShaderKind) -> Result<Arc<ComputePipeline>> {
    let shader_module = compile_shader_module(device.clone(), shader_kind)?;

    let entry_point = shader_module
        .entry_point("main")
        .context("entry point `main` não encontrada no shader")?;
    let stage = PipelineShaderStageCreateInfo::new(entry_point);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&[stage.clone()])
            .into_pipeline_layout_create_info(device.clone())
            .context("falha ao derivar o layout do pipeline")?,
    )
    .context("falha ao criar o pipeline layout")?;

    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .context("falha ao criar o compute pipeline")
}

fn compile_shader_module(
    device: Arc<Device>,
    shader_kind: ShaderKind,
) -> Result<Arc<ShaderModule>> {
    let source = match shader_kind {
        ShaderKind::Alu => ALU_SHADER_WGSL,
        ShaderKind::Memory => MEMORY_SHADER_WGSL,
        ShaderKind::Mixed => MIXED_SHADER_WGSL,
    };

    let module = wgsl::parse_str(source).context("falha ao parsear WGSL")?;
    let module_info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .context("falha ao validar WGSL")?;
    let spirv = write_vec(
        &module,
        &module_info,
        &SpvOptions::default(),
        Some(&SpvPipelineOptions {
            shader_stage: naga::ShaderStage::Compute,
            entry_point: "main".into(),
        }),
    )
    .context("falha ao converter WGSL para SPIR-V")?;

    unsafe {
        ShaderModule::new(device, ShaderModuleCreateInfo::new(&spirv))
            .context("falha ao criar ShaderModule SPIR-V")
    }
}

fn build_command_buffers(
    app: &AppContext,
    config: &BenchmarkConfig,
    resources: &PreparedResources,
    reusable: bool,
) -> Result<Vec<Arc<PrimaryAutoCommandBuffer>>> {
    if config.use_barrier && config.dispatches_per_run > 1 {
        let mut buffers = Vec::with_capacity(config.dispatches_per_run as usize);

        // AutoCommandBufferBuilder doesn't expose raw pipeline barriers, so this mode models
        // synchronization-heavy workloads by splitting each dispatch into its own submission.
        for index in 0..config.dispatches_per_run {
            buffers.push(build_single_command_buffer(
                app,
                resources,
                [config.dispatch_x, config.dispatch_y, config.dispatch_z],
                1,
                reusable,
                index == 0,
                index + 1 == config.dispatches_per_run,
            )?);
        }

        return Ok(buffers);
    }

    Ok(vec![build_single_command_buffer(
        app,
        resources,
        [config.dispatch_x, config.dispatch_y, config.dispatch_z],
        config.dispatches_per_run,
        reusable,
        true,
        true,
    )?])
}

fn build_single_command_buffer(
    app: &AppContext,
    resources: &PreparedResources,
    dispatch_groups: [u32; 3],
    dispatch_count: u32,
    reusable: bool,
    write_start_timestamp: bool,
    write_end_timestamp: bool,
) -> Result<Arc<PrimaryAutoCommandBuffer>> {
    let usage = if reusable {
        CommandBufferUsage::MultipleSubmit
    } else {
        CommandBufferUsage::OneTimeSubmit
    };
    let mut builder = AutoCommandBufferBuilder::primary(
        app.command_buffer_allocator.clone(),
        app.queue.queue_family_index(),
        usage,
    )
    .context("falha ao criar o command buffer builder")?;

    if write_start_timestamp {
        if let Some(query_pool) = &app.query_pool {
            unsafe {
                builder
                    .reset_query_pool(query_pool.clone(), 0..2)
                    .context("falha ao resetar query pool")?;
                builder
                    .write_timestamp(query_pool.clone(), 0, PipelineStage::TopOfPipe)
                    .context("falha ao gravar timestamp inicial")?;
            }
        }
    }

    builder
        .bind_pipeline_compute(resources.pipeline.clone())
        .context("falha ao bindar pipeline de compute")?
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            resources.pipeline.layout().clone(),
            0,
            resources.descriptor_set.clone(),
        )
        .context("falha ao bindar descriptor set")?;

    for _ in 0..dispatch_count {
        unsafe {
            builder
                .dispatch(dispatch_groups)
                .context("falha ao gravar dispatch de compute")?;
        }
    }

    if write_end_timestamp {
        if let Some(query_pool) = &app.query_pool {
            unsafe {
                builder
                    .write_timestamp(query_pool.clone(), 1, PipelineStage::BottomOfPipe)
                    .context("falha ao gravar timestamp final")?;
            }
        }
    }

    builder.build().context("falha ao finalizar command buffer")
}

struct TimingSample {
    submit_ns: u128,
    wait_ns: u128,
    gpu_kernel_ns: Option<f64>,
}

fn submit_and_measure(
    app: &AppContext,
    command_buffers: &[Arc<PrimaryAutoCommandBuffer>],
) -> Result<TimingSample> {
    let mut submit_ns = 0;
    let mut wait_ns = 0;

    for command_buffer in command_buffers {
        let submit_start = Instant::now();
        let future = sync::now(app.device.clone())
            .then_execute(app.queue.clone(), command_buffer.clone())
            .context("falha ao submeter command buffer")?
            .then_signal_fence_and_flush()
            .context("falha ao sinalizar fence")?;
        submit_ns += submit_start.elapsed().as_nanos();

        let wait_start = Instant::now();
        future.wait(None).context("falha ao aguardar a GPU")?;
        wait_ns += wait_start.elapsed().as_nanos();
    }

    let gpu_kernel_ns = read_gpu_timing(app)?;

    Ok(TimingSample {
        submit_ns,
        wait_ns,
        gpu_kernel_ns,
    })
}

fn read_gpu_timing(app: &AppContext) -> Result<Option<f64>> {
    let (Some(query_pool), Some(timestamp_period_ns)) = (&app.query_pool, app.timestamp_period_ns)
    else {
        return Ok(None);
    };

    let mut timestamps = [0_u64; 2];
    let ready = query_pool
        .get_results(0..2, &mut timestamps, QueryResultFlags::WAIT)
        .context("falha ao ler resultados do query pool")?;

    if !ready {
        return Ok(None);
    }

    let ticks = timestamps[1].saturating_sub(timestamps[0]);
    Ok(Some(ticks as f64 * timestamp_period_ns as f64))
}

fn checksum_buffer(buffer: &Subbuffer<[u32]>) -> Result<u64> {
    let content = buffer
        .read()
        .context("falha ao mapear o buffer para checksum")?;
    Ok(content
        .iter()
        .fold(0_u64, |acc, value| acc.wrapping_add(*value as u64)))
}

fn summarize(samples: &[BenchmarkSample]) -> BenchmarkSummary {
    let avg = |values: &[f64]| -> f64 {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    };

    let pipeline_values = samples
        .iter()
        .map(|sample| sample.pipeline_build_ns as f64)
        .collect::<Vec<_>>();
    let record_values = samples
        .iter()
        .map(|sample| sample.record_ns as f64)
        .collect::<Vec<_>>();
    let submit_values = samples
        .iter()
        .map(|sample| sample.submit_ns as f64)
        .collect::<Vec<_>>();
    let wait_values = samples
        .iter()
        .map(|sample| sample.wait_ns as f64)
        .collect::<Vec<_>>();
    let gpu_values = samples
        .iter()
        .filter_map(|sample| sample.gpu_kernel_ns)
        .collect::<Vec<_>>();

    BenchmarkSummary {
        samples: samples.len(),
        avg_pipeline_build_ns: avg(&pipeline_values),
        avg_record_ns: avg(&record_values),
        avg_submit_ns: avg(&submit_values),
        avg_wait_ns: avg(&wait_values),
        avg_gpu_kernel_ns: if gpu_values.is_empty() {
            None
        } else {
            Some(avg(&gpu_values))
        },
        min_gpu_kernel_ns: gpu_values.iter().copied().reduce(f64::min),
        max_gpu_kernel_ns: gpu_values.iter().copied().reduce(f64::max),
        final_checksum: samples.last().map(|sample| sample.checksum).unwrap_or(0),
    }
}

fn export_report(path: &Path, report: &BenchmarkReport) -> Result<()> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("csv") => export_csv(path, report),
        _ => export_json(path, report),
    }
}

fn export_json(path: &Path, report: &BenchmarkReport) -> Result<()> {
    let file = File::create(path)
        .with_context(|| format!("falha ao criar o arquivo {}", path.display()))?;
    serde_json::to_writer_pretty(file, report)
        .with_context(|| format!("falha ao escrever JSON em {}", path.display()))
}

fn export_csv(path: &Path, report: &BenchmarkReport) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)
        .with_context(|| format!("falha ao criar CSV em {}", path.display()))?;

    #[derive(Serialize)]
    struct CsvRow<'a> {
        shader: &'a str,
        device_name: &'a str,
        queue_family_index: u32,
        dispatch_x: u32,
        dispatch_y: u32,
        dispatch_z: u32,
        dispatches_per_run: u32,
        buffer_bytes: usize,
        recreate_pipeline: bool,
        rerecord_cmd: bool,
        use_barrier: bool,
        run_index: u32,
        pipeline_build_ns: u128,
        record_ns: u128,
        submit_ns: u128,
        wait_ns: u128,
        gpu_kernel_ns: Option<f64>,
        checksum: u64,
    }

    for sample in &report.samples {
        writer.serialize(CsvRow {
            shader: report.config.shader_profile,
            device_name: &report.device.device_name,
            queue_family_index: report.device.queue_family_index,
            dispatch_x: report.config.dispatch_x,
            dispatch_y: report.config.dispatch_y,
            dispatch_z: report.config.dispatch_z,
            dispatches_per_run: report.config.dispatches_per_run,
            buffer_bytes: report.config.buffer_bytes,
            recreate_pipeline: report.config.recreate_pipeline,
            rerecord_cmd: report.config.rerecord_cmd,
            use_barrier: report.config.use_barrier,
            run_index: sample.run_index,
            pipeline_build_ns: sample.pipeline_build_ns,
            record_ns: sample.record_ns,
            submit_ns: sample.submit_ns,
            wait_ns: sample.wait_ns,
            gpu_kernel_ns: sample.gpu_kernel_ns,
            checksum: sample.checksum,
        })?;
    }

    writer.flush().context("falha ao finalizar escrita do CSV")
}
