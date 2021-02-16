use tensorflow_lite_sys as ffi;

/// *IMPORTANT*: Always use GpuDelegateOptionsV2::default() method to create new instance of GpuDelegateOptionsV2,
/// otherwise every new added option may break inference.
pub struct GpuDelegateOptionsV2 {
    c_options: ffi::TfLiteGpuDelegateOptionsV2,
}

impl Default for GpuDelegateOptionsV2 {
    /// Populates GpuDelegateOptionsV2 as follows:
    /// `is_precision_loss_allowed` = false
    /// `inference_preference` = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER
    /// `priority1` = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION
    /// `priority2` = TFLITE_GPU_INFERENCE_PRIORITY_AUTO
    /// `priority3` = TFLITE_GPU_INFERENCE_PRIORITY_AUTO
    /// `experimental_flags` = TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT
    /// `max_delegated_partitions` = 1
    fn default() -> Self {
        let c_options = unsafe { ffi::TfLiteGpuDelegateOptionsV2Default() };
        Self { c_options }
    }
}

impl GpuDelegateOptionsV2 {
    fn set_is_precision_loss_allowed(&mut self, is_precision_loss_allowed: bool) {
        self.c_options.is_precision_loss_allowed = if is_precision_loss_allowed { 1 } else { 0 };
    }

    fn set_inference_preference(&mut self, inference_preference: GpuInferenceUsage) {
        self.c_options.inference_preference = inference_preference as ffi::TfLiteGpuInferenceUsage;
    }

    fn set_inference_priority1(&mut self, inference_priority: GpuInferencePriority) {
        self.c_options.inference_priority1 = inference_priority as ffi::TfLiteGpuInferencePriority;
    }

    fn set_inference_priority2(&mut self, inference_priority: GpuInferencePriority) {
        self.c_options.inference_priority2 = inference_priority as ffi::TfLiteGpuInferencePriority;
    }

    fn set_inference_priority3(&mut self, inference_priority: GpuInferencePriority) {
        self.c_options.inference_priority3 = inference_priority as ffi::TfLiteGpuInferencePriority;
    }

    fn set_experimental_flags(&mut self, experimental_flags: GpuExperimentalFlags) {
        self.experimental_flags = experimental_flags;
    }

    /// A graph could have multiple partitions that can be delegated to the GPU.
    /// This limits the maximum number of partitions to be delegated.
    /// By default, it's set to 1 in TfLiteGpuDelegateOptionsV2Default().
    fn set_max_delegated_partitions(&mut self, max_delegated_partitions: u32) {
        self.max_delegated_partitions = max_delegated_partitions as i32
    }

    fn as_ptr(&self) -> *const ffi::TfLiteGpuDelegateOptionsV2 {
        &self.c_options as *const _
    }
}

type GpuExperimentalFlags = ffi::TfLiteGpuExperimentalFlags;

#[repr(u32)]
pub enum GpuInferenceUsage {
    SustainedSpeed = ffi::TfLiteGpuInferenceUsage_TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED,
    FastSingleAnswer =
        ffi::TfLiteGpuInferenceUsage_TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER,
}

#[repr(u32)]
pub enum GpuInferencePriority {
    Auto = ffi::TfLiteGpuInferencePriority_TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
    MinLatency = ffi::TfLiteGpuInferencePriority_TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY,
    MaxPrecision = ffi::TfLiteGpuInferencePriority_TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION,
    MinMemoryUsage = ffi::TfLiteGpuInferencePriority_TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE,
}

pub struct GpuDelegateV2 {
    c_ptr: *mut ffi::TfLiteDelegate,
}

impl Default for GpuDelegateV2 {
    fn default() -> Self {
        Self {
            c_ptr: unsafe { ffi::TfLiteGpuDelegateV2Create(std::ptr::null()) },
        }
    }
}

impl crate::Delegate for GpuDelegateV2 {
    fn as_mut_ptr(&mut self) -> *mut ffi::TfLiteDelegate {
        self.c_ptr
    }
}

impl Drop for GpuDelegateV2 {
    fn drop(&mut self) {
        unsafe { ffi::TfLiteGpuDelegateV2Delete(self.c_ptr) }
    }
}

impl GpuDelegateV2 {
    pub fn new(options: &GpuDelegateOptionsV2) {
        Self {
            c_ptr: unsafe { ffi::TfLiteGpuDelegateV2Create(options.as_ptr()) },
        }
    }
}
