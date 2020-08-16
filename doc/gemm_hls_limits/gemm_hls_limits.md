# Notations
These are the notations used in our paper to describe `gemm_hls` parameters.
```
x_M = kOuterTileSizeN;
y_M = kOuterTileSizeM;
x_{Pr} = kInnerTileSizeN;
y_{Pr} = kComputeTileSizeM;
```

# Constants
```
kComputeTileSizeN = 1;
```

# Original Sanity Checks
These are the original asserts from `gemm_hls` project for the given parameters.
```
static_assert(kMemoryWidthBytesK % sizeof(CONFIG_DTYPE) == 0, "Memory width in K not divisible by size of data type.");
static_assert(kMemoryWidthBytesM % sizeof(CONFIG_DTYPE) == 0, "Memory width in M not divisible by size of data type.");
static_assert(kTransposeWidthBytes % sizeof(CONFIG_DTYPE) == 0, "Transpose width must be divisible by data size.");
static_assert(kTransposeWidthBytes % kMemoryWidthBytesK == 0, "Transpose width must be divisible by memory port width.");
static_assert(kOuterTileSizeM % kMemoryWidthM == 0, "Outer memory tile size in M must be divisible by memory port width.");
static_assert(kOuterTileSizeN % kInnerTileSizeN == 0, "Outer tile size must be divisible by the inner tile size.");
static_assert(kOuterTileSizeM % kComputeTileSizeM == 0, "Outer tile size must be divisible by compute tile size in M.");
static_assert(kInnerTileSizeN % kComputeTileSizeN == 0, "Inner tile size must be divisible by compute tile size.");
```

# Sanity Checks with The New Notations
```
static_assert(kMemoryWidthBytesK % sizeof(CONFIG_DTYPE) == 0, "Memory width in K not divisible by size of data type.");
static_assert(kMemoryWidthBytesM % sizeof(CONFIG_DTYPE) == 0, "Memory width in M not divisible by size of data type.");
static_assert(kTransposeWidthBytes % sizeof(CONFIG_DTYPE) == 0, "Transpose width must be divisible by data size.");
static_assert(kTransposeWidthBytes % kMemoryWidthBytesK == 0, "Transpose width must be divisible by memory port width.");
static_assert(y_M % kMemoryWidthM == 0, "Outer memory tile size in M must be divisible by memory port width.");
static_assert(x_M  % x_{Pr} == 0, "Outer tile size must be divisible by the inner tile size.");
static_assert(y_M % y_{Pr} == 0, "Outer tile size must be divisible by compute tile size in M.");
static_assert(x_{Pr} % 1 == 0, "Inner tile size must be divisible by compute tile size.");
```


