# FPGA Implementation Optimization Report

## Ground Rules
* Each op runs on specific input tensor. 


## Summary
* Task-Base: HW-EMU of non optimized task kernels.
* Best Results: results of the best HW-EMU of optimized versions of task kernels.


Op Name    |Task-Base |Best Results|
---        |---       |---         |
Concat2    |          |            |	    
Sqrt       |          |            |	    
ReduceMax  |          |            |	    
ReduceSum4D|          |            |	    
ReduceSum  |          |            |	    
Mean       |          |            |	    
Variance   |          |            |	    
Tile       |          |            |	    
MatOps     |          |            |	    
Transpose  |          |            |	    
Conv2d     |          |            |	    
ReLU       |          |            |	    
Matmul     |          |            |	    
Square     |          |            |	    
TopK       |          |            |	    
Gather     |          |            |	    

## In-depth Review
### Concat2 (Task)
    InputTn1 = 5,2,50,20
    InputTn2 = 5,2,50,30
    ConcatDim = 3
Commit|Duration|Description
---|---|---
000|000|0000

### Sqrt (Task)
    InputTn1 = 5,2,50,20 
Commit|Duration|Description
---|---|---
000|000|0000

### ReduceMax (Task)
    InputTn1 = 5,2,50,20 
    OverDim = 2
    ---
    InputTn1 = 5,5,1,20 
    OverDim = 1
Commit|Duration|Description
---|---|---
000|000|0000

### ReduceSum4D (Task)
    InputTn1 = 5,1024,20,256
    OverDim = TTTF
Commit|Duration|Description
---|---|---
000|000|0000

### ReduceSum (Task)
    InputTn1 = 50,25,20
    OverDim = TFF, FTF, FFT
Commit|Duration|Description
---|---|---
000|000|0000
    
### Tile (Task)
    InputTn1 = 5,2,1,20
    TileCount = 8
    TileAxis = 2
    ---
    InputTn1 = 5,2,1
    TileCount = 8
    TileAxis = 2
    ---
    InputTn1 = 5,1,20
    TileCount = 8
    TileAxis = 1
Commit|Duration|Description
---|---|---
000|000|0000

### Transpose (Task)
    InputTn1 = 5,1,20 
Commit|Duration|Description
---|---|---
000|000|0000

### ReLU (Task)
    InputTn1 = 10,50,20 
Commit|Duration|Description
---|---|---
000|000|0000

### Square (Task)
    InputTn1 = 10,50,20 
Commit|Duration|Description
---|---|---
000|000|0000

### MatOps (Task)
    InputTn1 = 11,12,13,14 
    InputTn2 = 11,12,13,14 
    ---
    InputTn1 = 11,12,13,14 
    InputTn2 = 12,13,14 
    ---
    InputTn1 = 11,12,13,14 
    InputTn2 = 13,14 
    ---
    InputTn1 = 11,12,13,14 
    InputTn2 = 14 
    ---
    InputTn1 = 11,12,13,14 
    InputTn2 = SCALAR
    ---
    InputTn1 = 12,13,14 
    InputTn2 = 12,13,14 
    ---
    InputTn1 = 12,13,14 
    InputTn2 = 13,14 
    ---
    InputTn1 = 12,13,14 
    InputTn2 = 14 
    ---
    InputTn1 = 12,13,14 
    InputTn2 = SCALAR
    ---
    InputTn1 = 13,14 
    InputTn2 = 13,14 
    ---
    InputTn1 = 13,14 
    InputTn2 = 14 
    ---
    InputTn1 = 13,14 
    InputTn2 = SCALAR
    ---
    InputTn1 = 14 
    InputTn2 = 14 
    ---
    InputTn1 = 14 
    InputTn2 = SCALAR
    
Commit|Duration|Description
---|---|---
000|000|0000

### Mean (Task)
    InputTn1 = 5,1024,20,256
    OverDim = TTTF
Commit|Duration|Description
---|---|---
000|000|0000

### Variance (Task)
    InputTn1 = 5,1024,20,256
    OverDim = TTTF
Commit|Duration|Description
---|---|---
000|000|0000

### MatMul (Task)
    InputTn1 = 1,50,20
    InputTn2 = 1,20,60
    ---
    InputTn1 = 25,1024
    InputTn2 = 1024,512
Commit|Duration|Description
---|---|---
000|000|0000

### Conv2MLP (Task)
    InputTn1 = 25,1024,20,6
    Weight = 1,1,6,64
    Bias = 64
Commit|Duration|Description
---|---|---
000|000|0000

### TopK (Task)
    InputTn1 = 5,1024,1024
    K = 20
Commit|Duration|Description
---|---|---
000|000|0000

### Gather (Task)
    InputTn1 = 5,1024,6
    IndicesTn = 5,1024,20 
Commit|Duration|Description
---|---|---
000|000|0000
