use std::convert::TryInto;
use std::ffi::{c_void, CStr, CString};
use std::path::Path;

use tensorflow_lite_sys as ffi;

#[macro_use]
extern crate num_derive;
use num_traits::{FromPrimitive, ToPrimitive};

use thiserror::Error;

#[cfg(feature = "coreml")]
pub mod coreml;
#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "metal")]
pub mod metal;

#[derive(FromPrimitive, ToPrimitive)]
#[repr(u32)]
pub enum AllocationType {
    ArenaRw = ffi::TfLiteAllocationType_kTfLiteArenaRw,
    ArenaRwPersitent = ffi::TfLiteAllocationType_kTfLiteArenaRwPersistent,
    Custom = ffi::TfLiteAllocationType_kTfLiteCustom,
    Dynamic = ffi::TfLiteAllocationType_kTfLiteDynamic,
    MemNone = ffi::TfLiteAllocationType_kTfLiteMemNone,
    MmapRo = ffi::TfLiteAllocationType_kTfLiteMmapRo,
    PersistentRo = ffi::TfLiteAllocationType_kTfLitePersistentRo,
}

#[cfg(feature = "experimental")]
#[derive(FromPrimitive, ToPrimitive)]
#[repr(u32)]
pub enum BuiltinOperator {
    Abs = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinAbs,
    Add = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinAdd,
    AddN = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinAddN,
    ArgMax = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinArgMax,
    ArgMin = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinArgMin,
    AveragePool2d = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinAveragePool2d,
    BatchMatmul = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinBatchMatmul,
    BatchToSpaceNd = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinBatchToSpaceNd,
    BidirectionalSequenceLstm = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinBidirectionalSequenceLstm,
    BidirectionalSequenceRnn = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinBidirectionalSequenceRnn,
    Call = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinCall,
    Cast = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinCast,
    Ceil = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinCeil,
    ConcatEmbeddings = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinConcatEmbeddings,
    Concatenation = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinConcatenation,
    Conv2d = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinConv2d,
    Cos = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinCos,
    Cumsum = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinCumsum,
    Custom = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinCustom,
    Delegate = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinDelegate,
    Densify = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinDensify,
    DepthToSpace = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinDepthToSpace,
    DepthwiseConv2d = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinDepthwiseConv2d,
    Dequantize = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinDequantize,
    Div = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinDiv,
    Elu = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinElu,
    EmbeddingLookup = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinEmbeddingLookup,
    EmbeddingLookupSparse = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinEmbeddingLookupSparse,
    Equal = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinEqual,
    Exp = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinExp,
    ExpandDims = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinExpandDims,
    FakeQuant = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinFakeQuant,
    Fill = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinFill,
    Floor = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinFloor,
    FloorDiv = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinFloorDiv,
    FloorMod = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinFloorMod,
    FullyConnected = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinFullyConnected,
    Gather = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinGather,
    GatherNd = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinGatherNd,
    Greater = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinGreater,
    GreaterEqual = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinGreaterEqual,
    HardSwish = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinHardSwish,
    HashtableLookup = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinHashtableLookup,
    If = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinIf,
    L2Normalization = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinL2Normalization,
    L2Pool2d = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinL2Pool2d,
    LeakyRelu = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinLeakyRelu,
    Less = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinLess,
    LessEqual = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinLessEqual,
    LocalResponseNormalization =
        ffi::TfLiteBuiltinOperator_kTfLiteBuiltinLocalResponseNormalization,
    Log = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinLog,
    LogSoftmax = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinLogSoftmax,
    LogicalAnd = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinLogicalAnd,
    LogicalNot = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinLogicalNot,
    LogicalOr = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinLogicalOr,
    Logistic = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinLogistic,
    LshProjection = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinLshProjection,
    Lstm = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinLstm,
    MatrixDiag = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinMatrixDiag,
    MatrixSetDiag = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinMatrixSetDiag,
    MaxPool2d = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinMaxPool2d,
    Maximum = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinMaximum,
    Mean = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinMean,
    Minimum = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinMinimum,
    MirrorPad = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinMirrorPad,
    Mul = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinMul,
    Neg = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinNeg,
    NonMaxSuppressionV4 = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinNonMaxSuppressionV4,
    NonMaxSuppressionV5 = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinNonMaxSuppressionV5,
    NotEqual = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinNotEqual,
    OneHot = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinOneHot,
    Pack = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinPack,
    Pad = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinPad,
    Padv2 = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinPadv2,
    PlaceholderForGreaterOpCodes =
        ffi::TfLiteBuiltinOperator_kTfLiteBuiltinPlaceholderForGreaterOpCodes,
    Pow = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinPow,
    Prelu = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinPrelu,
    Quantize = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinQuantize,
    Range = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinRange,
    Rank = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinRank,
    ReduceAny = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinReduceAny,
    ReduceMax = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinReduceMax,
    ReduceMin = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinReduceMin,
    ReduceProd = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinReduceProd,
    Relu = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinRelu,
    Relu6 = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinRelu6,
    ReluN1To1 = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinReluN1To1,
    Reshape = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinReshape,
    ResizeBilinear = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinResizeBilinear,
    ResizeNearestNeighbor = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinResizeNearestNeighbor,
    ReverseSequence = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinReverseSequence,
    ReverseV2 = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinReverseV2,
    Rnn = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinRnn,
    Round = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinRound,
    Rsqrt = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinRsqrt,
    ScatterNd = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinScatterNd,
    SegmentSum = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinSegmentSum,
    Select = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinSelect,
    SelectV2 = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinSelectV2,
    Shape = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinShape,
    Sin = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinSin,
    SkipGram = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinSkipGram,
    Slice = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinSlice,
    Softmax = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinSoftmax,
    SpaceToBatchNd = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinSpaceToBatchNd,
    SpaceToDepth = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinSpaceToDepth,
    SparseToDense = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinSparseToDense,
    Split = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinSplit,
    SplitV = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinSplitV,
    Sqrt = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinSqrt,
    Square = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinSquare,
    SquaredDifference = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinSquaredDifference,
    Squeeze = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinSqueeze,
    StridedSlice = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinStridedSlice,
    Sub = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinSub,
    Sum = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinSum,
    Svdf = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinSvdf,
    Tanh = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinTanh,
    Tile = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinTile,
    TopkV2 = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinTopkV2,
    Transpose = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinTranspose,
    TransposeConv = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinTransposeConv,
    UnidirectionalSequenceLstm =
        ffi::TfLiteBuiltinOperator_kTfLiteBuiltinUnidirectionalSequenceLstm,
    UnidirectionalSequenceRnn = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinUnidirectionalSequenceRnn,
    Unique = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinUnique,
    Unpack = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinUnpack,
    Where = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinWhere,
    While = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinWhile,
    ZerosLike = ffi::TfLiteBuiltinOperator_kTfLiteBuiltinZerosLike,
}

#[derive(FromPrimitive, ToPrimitive)]
#[repr(u32)]
pub enum DimensionType {
    DimDense = ffi::TfLiteDimensionType_kTfLiteDimDense,
    DimSparseCSR = ffi::TfLiteDimensionType_kTfLiteDimSparseCSR,
}

#[derive(FromPrimitive, ToPrimitive)]
#[repr(u32)]
pub enum ExternalContextType {
    CpuBackendContext = ffi::TfLiteExternalContextType_kTfLiteCpuBackendContext,
    EdgeTpuContext = ffi::TfLiteExternalContextType_kTfLiteEdgeTpuContext,
    EigenContext = ffi::TfLiteExternalContextType_kTfLiteEigenContext,
    GemmLowpContext = ffi::TfLiteExternalContextType_kTfLiteGemmLowpContext,
    MaxExternalContexts = ffi::TfLiteExternalContextType_kTfLiteMaxExternalContexts,
}

#[derive(FromPrimitive, ToPrimitive)]
#[repr(u32)]
pub enum QuantizationType {
    AffineQuantization = ffi::TfLiteQuantizationType_kTfLiteAffineQuantization,
    NoQuantization = ffi::TfLiteQuantizationType_kTfLiteNoQuantization,
}

#[derive(FromPrimitive, ToPrimitive, Error, Debug)]
pub enum TFLiteError {
    #[error("application error.")]
    ApplicationError, // (#[from] ffi::TfLiteStatus_kTfLiteApplicationError),
    #[error("delegate error.")]
    DelegateError, // (#[from] ffi::TfLiteStatus_kTfLiteDelegateError),
    #[error("interpreter error.")]
    InterpreterError, // (#[from] ffi::TfLiteStatus_kTfLiteError),
}

#[derive(FromPrimitive, ToPrimitive)]
#[repr(u32)]
pub enum TFLiteStatus {
    ApplicationError = ffi::TfLiteStatus_kTfLiteApplicationError,
    DelegateError = ffi::TfLiteStatus_kTfLiteDelegateError,
    Error = ffi::TfLiteStatus_kTfLiteError,
    Ok = ffi::TfLiteStatus_kTfLiteOk,
}

type Result = std::result::Result<(), TFLiteError>;

impl TryInto<()> for TFLiteStatus {
    type Error = TFLiteError;

    fn try_into(self) -> std::result::Result<(), TFLiteError> {
        match self {
            TFLiteStatus::Ok => Ok(()),
            TFLiteStatus::ApplicationError => Err(TFLiteError::ApplicationError),
            TFLiteStatus::DelegateError => Err(TFLiteError::DelegateError),
            TFLiteStatus::Error => Err(TFLiteError::InterpreterError),
        }
    }
}

#[derive(FromPrimitive, ToPrimitive, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u32)]
pub enum Type {
    Bool = ffi::TfLiteType_kTfLiteBool,
    Complex64 = ffi::TfLiteType_kTfLiteComplex64,
    Complex128 = ffi::TfLiteType_kTfLiteComplex128,
    Float16 = ffi::TfLiteType_kTfLiteFloat16,
    Float32 = ffi::TfLiteType_kTfLiteFloat32,
    Float64 = ffi::TfLiteType_kTfLiteFloat64,
    Int8 = ffi::TfLiteType_kTfLiteInt8,
    Int16 = ffi::TfLiteType_kTfLiteInt16,
    Int32 = ffi::TfLiteType_kTfLiteInt32,
    Int64 = ffi::TfLiteType_kTfLiteInt64,
    NoType = ffi::TfLiteType_kTfLiteNoType,
    String = ffi::TfLiteType_kTfLiteString,
    UInt8 = ffi::TfLiteType_kTfLiteUInt8,
}

impl Type {
    pub fn name(&self) -> &str {
        unsafe { CStr::from_ptr(ffi::TfLiteTypeGetName(self.to_u32().unwrap())) }
            .to_str()
            .unwrap()
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Type::Bool => "bool",
            Type::Complex64 => "Complex<f64>",
            Type::Complex128 => "Complex<f128>",
            Type::Float16 => "f16",
            Type::Float32 => "f32",
            Type::Float64 => "f64",
            Type::Int8 => "i8",
            Type::Int16 => "i16",
            Type::Int32 => "i32",
            Type::Int64 => "i64",
            Type::NoType => "()",
            Type::String => "string",
            Type::UInt8 => "u8",
        };
        write!(f, "{}", name)
    }
}

pub trait DType {
    fn dtype() -> Type;
}

macro_rules! impl_dtype {
    ($type:ty, $dtype:path) => {
        impl DType for $type {
            fn dtype() -> Type {
                $dtype
            }
        }
    };
}

#[cfg(features = "half")]
use half::f16;
#[cfg(features = "half")]
impl_dtype!(f16, Type::Float16);

#[cfg(features = "complex")]
use num_complex::Complex;

#[cfg(features = "complex")]
impl_dtype!(Complex<f64>, Type::Complex64);

#[cfg(all(features = "complex", features = "f128"))]
impl_dtype!(Complex<f128>, Type::Complex128);

impl_dtype!(bool, Type::Bool);
impl_dtype!(f32, Type::Float32);
impl_dtype!(f64, Type::Float64);
impl_dtype!(i8, Type::Int8);
impl_dtype!(u8, Type::UInt8);
impl_dtype!(i16, Type::Int16);
impl_dtype!(i32, Type::Int32);
impl_dtype!(i64, Type::Int64);
impl_dtype!((), Type::NoType);
impl_dtype!(CString, Type::String);

pub trait Delegate {
    fn as_mut_ptr(&mut self) -> *mut ffi::TfLiteDelegate;
}

pub struct Registration {
    _ptr: *mut ffi::TfLiteRegistration,
}

pub struct Model {
    ptr: *mut ffi::TfLiteModel,
}

impl Model {
    pub fn create(data: &[u8]) -> Self {
        let ptr = unsafe { ffi::TfLiteModelCreate(data.as_ptr() as *const c_void, data.len()) };
        Self { ptr }
    }

    pub fn create_from_file<P: AsRef<Path>>(path: P) -> Self {
        let ptr = unsafe {
            let cstr = CString::new(path.as_ref().to_str().unwrap()).unwrap();
            ffi::TfLiteModelCreateFromFile(cstr.as_ptr())
        };
        Self { ptr }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe {
            ffi::TfLiteModelDelete(self.ptr);
        }
    }
}

pub struct Tensor {
    ptr: *const ffi::TfLiteTensor,
    shape: Vec<usize>,
    ndim: usize,
}

impl Tensor {
    fn as_mut_ptr(&mut self) -> *mut ffi::TfLiteTensor {
        self.ptr as _
    }

    pub fn into_raw(self) -> *const ffi::TfLiteTensor {
        self.ptr
    }

    pub fn from_raw(raw_ptr: *const ffi::TfLiteTensor) -> Self {
        Self::from(raw_ptr)
    }

    pub fn data_mut<T>(&mut self) -> &mut [T]
    where
        T: DType,
    {
        assert_eq!(
            self.dtype(),
            T::dtype(),
            "Tensor's dtype is {}, not {}",
            self.dtype(),
            T::dtype()
        );
        unsafe {
            std::slice::from_raw_parts_mut(
                ffi::TfLiteTensorData(self.as_mut_ptr()) as *mut T,
                ffi::TfLiteTensorByteSize(self.as_mut_ptr()) / std::mem::size_of::<T>(),
            )
        }
    }

    pub fn copy_from<T>(&mut self, buffer: &[T]) -> Result
    where
        T: DType,
    {
        assert_eq!(self.dtype(), T::dtype());
        TFLiteStatus::from_u32(unsafe {
            ffi::TfLiteTensorCopyFromBuffer(
                self.as_mut_ptr(),
                buffer.as_ptr() as *const c_void,
                buffer.len() * std::mem::size_of::<T>(),
            )
        })
        .unwrap()
        .try_into()
    }

    /*
    fn data_free(&mut self) {
        unsafe { ffi::TfLiteTensorDataFree(self.as_mut_ptr()) }
    }
    */

    pub fn shape(&self) -> &[usize] {
        self.shape.as_slice()
    }

    pub fn ndim(&self) -> usize {
        self.ndim
    }

    pub fn data<T>(&self) -> &[T]
    where
        T: DType,
    {
        assert_eq!(
            self.dtype(),
            T::dtype(),
            "Tensor's dtype is {}, not {}",
            self.dtype(),
            T::dtype()
        );
        unsafe {
            std::slice::from_raw_parts(
                ffi::TfLiteTensorData(self.ptr) as *mut T,
                ffi::TfLiteTensorByteSize(self.ptr) / std::mem::size_of::<T>(),
            )
        }
    }

    fn as_ptr(&self) -> *const ffi::TfLiteTensor {
        self.ptr as _
    }

    pub fn copy_to<T>(&self, buffer: &mut [T]) -> Result
    where
        T: DType,
    {
        assert_eq!(self.dtype(), T::dtype());
        TFLiteStatus::from_u32(unsafe {
            ffi::TfLiteTensorCopyToBuffer(
                self.ptr,
                buffer.as_mut_ptr() as *mut c_void,
                buffer.len() * std::mem::size_of::<T>(),
            )
        })
        .unwrap()
        .try_into()
    }

    fn byte_size(&self) -> usize {
        unsafe { ffi::TfLiteTensorByteSize(self.as_ptr()) }
    }

    fn dtype(&self) -> Type {
        Type::from_u32(unsafe { ffi::TfLiteTensorType(self.as_ptr()) }).unwrap()
    }

    fn name(&self) -> &CStr {
        unsafe { CStr::from_ptr(ffi::TfLiteTensorName(self.as_ptr())) }
    }

    /*
    fn reset(&mut self) {}
        TfLiteTensorReset
        */
    /*
        fn realloc()
    TfLiteTensorRealloc
        */
}

impl From<*mut ffi::TfLiteTensor> for Tensor {
    fn from(ptr: *mut ffi::TfLiteTensor) -> Self {
        let mut shape = Vec::new();
        let ndim = unsafe { ffi::TfLiteTensorNumDims(ptr) };
        for dim in 0..ndim {
            shape.push(unsafe { ffi::TfLiteTensorDim(ptr, dim) } as usize)
        }
        Self {
            shape,
            ptr,
            ndim: ndim as usize,
        }
    }
}

impl From<*const ffi::TfLiteTensor> for Tensor {
    fn from(ptr: *const ffi::TfLiteTensor) -> Self {
        let mut shape = Vec::new();
        let ndim = unsafe { ffi::TfLiteTensorNumDims(ptr) };
        for dim in 0..ndim {
            shape.push(unsafe { ffi::TfLiteTensorDim(ptr, dim) } as usize)
        }
        Self {
            shape,
            ptr,
            ndim: ndim as usize,
        }
    }
}

/*
TfLiteTensorFree⚠
TfLiteTensorName⚠
⚠
TfLiteTensorQuantizationParams⚠
TfLiteTensorType⚠
}
*/

pub struct Interpreter {
    ptr: *mut ffi::TfLiteInterpreter,
    input_tensors: Vec<Tensor>,
    input_tensor_count: i32,
    output_tensors: Vec<Tensor>,
    output_tensor_count: i32,
}

impl Interpreter {
    pub fn allocate_tensors(&mut self) -> Result {
        TFLiteStatus::from_u32(unsafe { ffi::TfLiteInterpreterAllocateTensors(self.ptr) })
            .unwrap()
            .try_into()
    }

    pub fn new(model: &Model) -> Self {
        Self::create(model, None)
    }

    pub fn new_with_options(model: &Model, options: &InterpreterOptions) -> Self {
        Self::create(model, Some(options))
    }

    pub fn create(model: &Model, options: Option<&InterpreterOptions>) -> Self {
        let ptr = match options {
            Some(options) => unsafe { ffi::TfLiteInterpreterCreate(model.ptr, options.ptr) },
            None => unsafe { ffi::TfLiteInterpreterCreate(model.ptr, std::ptr::null()) },
        };
        Self::from(ptr)
    }

    #[cfg(feature = "experimental")]
    pub fn create_with_selected_ops(model: &Model, options: &InterpreterOptions) -> Self {
        let ptr = unsafe { ffi::TfLiteInterpreterCreateWithSelectedOps(model.ptr, options.ptr) };
        Self::from(ptr)
    }

    pub fn get_input_tensor(&mut self, input_index: usize) -> Option<&mut Tensor> {
        self.input_tensors.get_mut(input_index)
    }

    pub fn get_input_tensor_count(&self) -> i32 {
        self.input_tensor_count
    }

    pub fn get_output_tensor(&self, output_index: usize) -> Option<&Tensor> {
        self.output_tensors.get(output_index)
    }

    pub fn get_output_tensor_count(&self) -> i32 {
        self.output_tensor_count
    }

    pub fn invoke(&mut self) -> Result {
        TFLiteStatus::from_u32(unsafe { ffi::TfLiteInterpreterInvoke(self.ptr) })
            .unwrap()
            .try_into()
    }

    #[cfg(feature = "experimental")]
    pub fn reset_variable_tensor(&mut self) -> Result {
        TFLiteStatus::from_u32(unsafe { ffi::TfLiteInterpreterResetVariableTensors(self.ptr) })
            .unwrap()
            .try_into()
    }

    pub fn resize_input_tensor(&mut self, input_index: i32, input_dims: &[i32]) -> Result {
        TFLiteStatus::from_u32(unsafe {
            ffi::TfLiteInterpreterResizeInputTensor(
                self.ptr,
                input_index,
                input_dims.as_ptr(),
                input_dims.len() as i32,
            )
        })
        .unwrap()
        .try_into()
    }
}

impl From<*mut ffi::TfLiteInterpreter> for Interpreter {
    fn from(ptr: *mut ffi::TfLiteInterpreter) -> Self {
        let input_tensor_count = unsafe { ffi::TfLiteInterpreterGetInputTensorCount(ptr) };

        assert!(input_tensor_count > 0);
        let mut input_tensors = Vec::with_capacity(input_tensor_count as usize);

        for input_index in 0..input_tensor_count {
            let tensor_ptr = unsafe { ffi::TfLiteInterpreterGetInputTensor(ptr, input_index) };
            input_tensors.push(Tensor::from_raw(tensor_ptr));
        }

        let output_tensor_count = unsafe { ffi::TfLiteInterpreterGetOutputTensorCount(ptr) };
        assert!(output_tensor_count > 0);
        let mut output_tensors = Vec::with_capacity(output_tensor_count as usize);

        for output_index in 0..output_tensor_count {
            let tensor_ptr = unsafe { ffi::TfLiteInterpreterGetOutputTensor(ptr, output_index) };
            output_tensors.push(Tensor::from_raw(tensor_ptr));
        }

        Self {
            ptr,
            input_tensor_count,
            input_tensors,
            output_tensor_count,
            output_tensors,
        }
    }
}

impl Drop for Interpreter {
    fn drop(&mut self) {
        unsafe { ffi::TfLiteInterpreterDelete(self.ptr) };
    }
}

pub struct InterpreterOptions {
    ptr: *mut ffi::TfLiteInterpreterOptions,
}

impl InterpreterOptions {
    #[cfg(feature = "experimental")]
    /// Adds an op registration for a builtin operator.
    pub fn add_builtin_op(
        &mut self,
        op: BuiltinOperator,
        registration: &Registration,
        min_version: i32,
        max_version: i32,
    ) {
        unsafe {
            ffi::TfLiteInterpreterOptionsAddBuiltinOp(
                self.ptr,
                op as ffi::TfLiteBuiltinOperator,
                registration.ptr,
                min_version,
                max_version,
            )
        }
    }

    #[cfg(feature = "experimental")]
    /// Adds an op registration for a custom operator.
    pub fn add_custom_op(
        &mut self,
        name: &str,
        registration: &Registration,
        min_version: i32,
        max_version: i32,
    ) {
        unsafe {
            ffi::TfLiteInterpreterOptionsAddCustomOp(
                self.ptr,
                name.as_ptr() as *const c_char,
                registration.ptr,
                min_version,
                max_version,
            )
        }
    }

    pub fn add_delegate<D: Delegate>(&mut self, delegate: &mut D) {
        unsafe { ffi::TfLiteInterpreterOptionsAddDelegate(self.ptr, delegate.as_mut_ptr()) }
    }

    fn create() -> Self {
        let ptr = unsafe { ffi::TfLiteInterpreterOptionsCreate() };
        Self { ptr }
    }

    pub fn new() -> Self {
        Self::create()
    }

    /*
    fn set_error_reporter(&mut self, reporter: R, user_data: &mut Data) {
        unsafe {
            ffi::TfLiteInterpreterOptionsSetErrorReporter(self.ptr, reporter, user_data);
        }
        // reporter: Option<unsafe extern "C" fn(user_data: *mut c_void, format: *const c_char, args: *mut __va_list_tag)>,
        // user_data: *mut c_void
    }
    */
    pub fn set_num_threads(&mut self, num_threads: i32) {
        unsafe { ffi::TfLiteInterpreterOptionsSetNumThreads(self.ptr, num_threads) }
    }

    /*
    /// Registers callbacks for resolving builtin or custom operators.
    fn set_op_resolver<FB, FC>(&mut self, find_builtin_op: FB, find_custom_op: FC) {
        // find_builtin_op: Option<unsafe extern "C" fn(user_data: *mut c_void, op: TfLiteBuiltinOperator, version: c_int) -> *const TfLiteRegistration>,
        // find_custom_op: Option<unsafe extern "C" fn(user_data: *mut c_void, custom_op: *const c_char, version: c_int) -> *const TfLiteRegistration>,
        // op_resolver_user_data: *mut c_void
        unsafe { ffi::TfLiteInterpreterOptionsSetOpResolver(self.ptr, find_builtin_op, find_custom_op) }
    }
    */

    #[cfg(feature = "experimental")]
    /// Enable or disable the NN API for the interpreter (true to enable).
    /// *WARNING*: This is an experimental API and subject to change.
    pub fn set_use_nnapi(&mut self, enable: bool) {
        unsafe { ffi::TfLiteInterpreterOptionsSetUseNNAPI(self.ptr, enable) };
    }
}

impl Default for InterpreterOptions {
    fn default() -> Self {
        Self::create()
    }
}

impl Drop for InterpreterOptions {
    fn drop(&mut self) {
        unsafe { ffi::TfLiteInterpreterOptionsDelete(self.ptr) };
    }
}

pub fn version() -> &'static str {
    unsafe { CStr::from_ptr(ffi::TfLiteVersion()) }
        .to_str()
        .unwrap()
}

#[cfg(feature = "xnnpack")]
mod xnnpack {
    use super::*;

    pub struct XNNPackDelegateOptions {
        options: ffi::TfLiteXNNPackDelegateOptions,
        num_threads: i32,
    }

    impl Default for XNNPackDelegateOptions {
        fn default() -> Self {
            let options = unsafe { ffi::TfLiteXNNPackDelegateOptionsDefault() };
            XNNPackDelegateOptions {
                options,
                num_threads: options.num_threads,
            }
        }
    }

    impl XNNPackDelegateOptions {
        pub fn set_num_threads(&mut self, num_threads: i32) {
            self.num_threads = num_threads;
            self.options.num_threads = num_threads;
        }

        pub fn num_threads(&self) -> i32 {
            self.num_threads
        }

        fn as_ptr(&self) -> *const ffi::TfLiteXNNPackDelegateOptions {
            &self.options as *const ffi::TfLiteXNNPackDelegateOptions
        }
    }

    pub struct XNNPackDelegate {
        c_ptr: *mut ffi::TfLiteDelegate,
    }

    impl XNNPackDelegate {
        pub fn new(options: Option<XNNPackDelegateOptions>) -> Self {
            let c_ptr =
                unsafe { ffi::TfLiteXNNPackDelegateCreate(options.unwrap_or_default().as_ptr()) };
            Self { c_ptr }
        }
    }

    impl Drop for XNNPackDelegate {
        fn drop(&mut self) {
            unsafe {
                ffi::TfLiteXNNPackDelegateDelete(self.c_ptr);
            }
        }
    }

    impl Delegate for XNNPackDelegate {
        fn as_mut_ptr(&mut self) -> *mut ffi::TfLiteDelegate {
            self.c_ptr
        }
    }
}

#[cfg(feature = "xnnpack")]
pub use xnnpack::*;
