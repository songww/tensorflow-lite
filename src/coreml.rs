use tensorflow_lite_sys as ffi;

#[repr(u32)]
pub enum CoreMlDelegateEnabledDevices {
    AllDevices = ffi::TfLiteCoreMlDelegateEnabledDevices_TfLiteCoreMlDelegateAllDevices,
    WithNeuralEngine =
        ffi::TfLiteCoreMlDelegateEnabledDevices_TfLiteCoreMlDelegateDevicesWithNeuralEngine,
}

pub struct CoreMlDelegateOptions {
    /// Only create delegate when Neural Engine is available on the device.
    pub enabled_devices: CoreMlDelegateEnabledDevices,
    /// Specifies target Core ML version for model conversion.
    /// Core ML 3 come with a lot more ops, but some ops (e.g. reshape) is not delegated due to input rank constraint.
    /// if not set to one of the valid versions, the delegate will use highest version possible in the platform.
    /// Valid versions: (2, 3)
    pub coreml_version: i32,
    /// This sets the maximum number of Core ML delegates created.
    /// Each graph corresponds to one delegated node subset in the TFLite model.
    /// Set this to 0 to delegate all possible partitions.
    pub max_delegated_partitions: i32,
    /// This sets the minimum number of nodes per partition delegated with Core ML delegate.
    /// Defaults to 2.
    pub min_nodes_per_partition: i32,
}

impl Default for CoreMlDelegateOptions {
    fn default() -> CoreMlDelegateOptions {
        CoreMlDelegateOptions {
            enabled_devices: CoreMlDelegateEnabledDevices::WithNeuralEngine,
            coreml_version: 3,
            max_delegated_partitions: 0,
            min_nodes_per_partition: 2,
        }
    }
}

impl CoreMlDelegateOptions {
    fn as_ptr(&self) -> *const ffi::TfLiteCoreMlDelegateOptions {
        ffi::TfLiteCoreMlDelegateOptions {
            enabled_devices: self.enabled_devices as ffi::TfLiteCoreMlDelegateEnabledDevices,
            coreml_version: self.coreml_version,
            max_delegated_partitions: self.max_delegated_partitions,
            min_nodes_per_partition: self.min_nodes_per_partition,
        }
    }
}

pub struct CoreMlDelegate {
    c_ptr: *mut ffi::TfLiteDelegate,
}

impl Drop for CoreMlDelegate {
    fn drop(&mut self) {
        unsafe { ffi::TfLiteCoreMlDelegateDelete(self.c_ptr) }
    }
}

impl Default for CoreMlDelegate {
    fn default() -> CoreMlDelegate {
        Self {
            c_ptr: unsafe { ffi::TfLiteCoreMlDelegateCreate(std::ptr::null()) },
        }
    }
}

impl crate::Delegate for CoreMlDelegate {
    fn as_mut_ptr(&mut self) -> *mut ffi::TfLiteDelegate {
        self.c_ptr
    }
}
