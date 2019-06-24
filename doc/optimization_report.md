# FPGA Implementation Optimization Report

## Ground Rules
* Each op runs on specific input tensor. 
* Reports are for C/C++ HLS entery point, not OpenCL.


## Summary
* Task-Base: HW-EMU of non optimized task kernels.
* Best Results: results of the best HW-EMU of optimized versions of task kernels.


Op Name    |Task-Base 		|Best Results|
---        |---       		|---         |
Concat2    |0.017          	|            |	    
Sqrt       |0.004          	|            |	    
ReduceMax  |0.005, 0.008    |            |	    
ReduceSum4D|          		|            |	    
ReduceSum  |          		|            |	    
Mean       |          		|            |	    
Variance   |          		|            |	    
Tile       |          		|            |	    
MatOps     |          		|            |	    
Transpose  |0.013          	|            |	    
Conv2d     |          		|            |	    
ReLU       |0.002          	|            |	    
Matmul     |          		|            |	    
Square     |0.002          	|            |	    
TopK       |0.062          	|            |	    
Gather     |          		|            |	    

## In-depth Review
### Concat2 (Task)
    InputTn1 = 2,2,2,3
    InputTn2 = 2,2,2,2
    ConcatDim = 3
Commit|Duration|Description
---|---|---
000|000|0000

* Different bundles are used for inputs and output to achieve seperat mAXI and burst read and write of gmem.
* The loop_flatten pragma requires perfect or semi-perfect nested loops which means that outermost for-loop should have constant bound.
* The loop_flatten pragma hinders loop_tripcount pragma and causes it to
* Compiler and or linker options are needed to assign different memory banks to mAXI ports.

### Sqrt (Task)
    InputTn1 = 2,2,2,2
Commit|Duration|Description
---|---|---
000|000|0000

### ReduceMax (Task)
    InputTn1 = 2,2,3,2
    OverDim = 2
    ---
    InputTn1 = 2,3,2,2
    OverDim = 1
Commit|Duration|Description
---|---|---
000|000|0000

### ReduceSum4D (Task)
    InputTn1 = 2,2,2,5
    OverDim = TTTF
Commit|Duration|Description
---|---|---
000|000|0000

### ReduceSum (Task)
    InputTn1 = 2,2,2
    OverDim = TFF, FTF, FFT
Commit|Duration|Description
---|---|---
000|000|0000
    
### Tile (Task)
    InputTn1 = 2,2,1,2
    TileCount = 8
    TileAxis = 2
    ---
    InputTn1 = 2,2,1
    TileCount = 8
    TileAxis = 2
    ---
    InputTn1 = 2,1,2
    TileCount = 8
    TileAxis = 1
Commit|Duration|Description
---|---|---
000|000|0000

### Transpose (Task)
    InputTn1 = 3,4,5 
Commit|Duration|Description
---|---|---
000|000|0000

### ReLU (Task)
    InputTn1 = 2,2,2 
Commit|Duration|Description
---|---|---
000|000|0000

### Square (Task)
    InputTn1 = 2,2,2 
Commit|Duration|Description
---|---|---
000|000|0000

### MatOps (Task)
    InputTn1 = 2,2,2,2 
    InputTn2 = 2,2,2,2 
    ---
    InputTn1 = 2,2,2,2 
    InputTn2 = 2,2,2
    ---
    InputTn1 = 2,2,2,2
    InputTn2 = 2,2
    ---
    InputTn1 = 2,2,2,2
    InputTn2 = 2
    ---
    InputTn1 = 2,2,2,2
    InputTn2 = SCALAR
    ---
    InputTn1 = 2,2,2
    InputTn2 = 2,2,2
    ---
    InputTn1 = 2,2,2
    InputTn2 = 2,2 
    ---
    InputTn1 = 2,2,2 
    InputTn2 = 2
    ---
    InputTn1 = 2,2,2 
    InputTn2 = SCALAR
    ---
    InputTn1 = 2,2 
    InputTn2 = 2,2 
    ---
    InputTn1 = 2,2 
    InputTn2 = 2 
    ---
    InputTn1 = 2,2 
    InputTn2 = SCALAR
    ---
    InputTn1 = 2 
    InputTn2 = 2 
    ---
    InputTn1 = 2 
    InputTn2 = SCALAR
    
Commit|Duration|Description
---|---|---
000|000|0000

### Mean (Task)
    InputTn1 = 2,2,2,5
    OverDim = TTTF
Commit|Duration|Description
---|---|---
000|000|0000

### Variance (Task)
    InputTn1 = 2,2,2,5
    OverDim = TTTF
Commit|Duration|Description
---|---|---
000|000|0000

### MatMul (Task)
    InputTn1 = 1,5,2
    InputTn2 = 1,2,5
    ---
    InputTn1 = 3,4
    InputTn2 = 4,5
Commit|Duration|Description
---|---|---
000|000|0000

### Conv2MLP (Task)
    InputTn1 = 2,2,3,3
    Weight = 1,1,3,4
    Bias = 4
Commit|Duration|Description
---|---|---
000|000|0000

### TopK (Task)
    InputTn1 = 2,5,5
    K = 3
Commit|Duration|Description
---|---|---
000|000|0000

### Gather (Task)
    InputTn1 = 5,5,2
    IndicesTn = 5,5,3
Commit|Duration|Description
---|---|---
000|000|0000
