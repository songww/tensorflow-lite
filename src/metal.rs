use tensorflow_lite_sys as ffi;

pub struct MetalGpuDelegate {
    c_ptr: *mut ffi::TfLiteDelegate,
}

impl Drop for MetalGpuDelegate {
    fn drop(&mut self) {
        unsafe { ffi::TFLGpuDelegateDelete(self.c_ptr) }
    }
}

impl Default for MetalGpuDelegate {
    fn default() -> MetalGpuDelegate {
        let c_ptr = unsafe { ffi::TFLGpuDelegateCreate(ffi::TFLGpuDelegateOptionsDefault()) };
        Self { c_ptr }
    }
}

impl crate::Delegate for MetalGpuDelegate {
    fn as_mut_ptr(&mut self) -> *mut ffi::TfLiteDelegate {
        self.c_ptr
    }
}

//ffi::TFLGpuDelegateOptions
