# Multiplatform Implementation Summary

## Task List
- [ ] Improve Conv2D @CUDA
- [ ] Improve MatMul @CUDA
- [ ] hide latencies @CUDA


## Ops
Op Name        | Batch | CPU  | CUDA  | OCL  |Shape1    | Shape2| Combinations                  | Sett1         |Val1   |Sett2      |Val2   | Notes|
---            | ---   | ---  | ---   | ---  | ---      | ---   | --------------------          | ---           | ---   | ---       | ---   |  --- |
Concat2        |     No|Yes   |**Yes**|**Yes**|4D        |4D     |-                              |Concat2        |3      |           |-      |--
ReduceMax      |     No|Yes   |**Yes**|**Yes**|4D        |-      |-                              |reductionDim   |1,2    |           |-      |--
ReduceSum4D    |     No|Yes   |**Yes**|**Yes**|4D        |-      |{1-1-1-0}                      |               |-      |           |-      |--
ReduceSum      |     No|Yes   |**Yes**|**Yes**|2D,3D     |-      |{3D: 0-0-1}, {2D: 0-1-0}       |               |-      |           |-      |--
Mean           |     No|Yes   |**Yes**|**Yes**|2D,4D     |-      |{1-0-0-0}, {1-1-1-0}           |               |-      |           |-      |--
Variance       |     No|Yes   |**Yes**|**Yes**|2D,4D     |-      |{2D: 1-0-0-0}, {4D: 1-1-1-0}   |               |-      |           |-      |--
Tile           |     No|Yes   |**Yes**|**Yes**|3D,4D     |-      |-                              |tileAxis       |1,2    |tileCount  |20,1024|only tileAxis=2 implemented
MatAdd         |     No|Yes   |**Yes**|   -   |1D,3D     |1D,3D  |-                              |               |-      |           |-      |Replaced by MatOps
MatAddTiled    |     No|Yes   |**Yes**|   -   |2D,4D     |1D     |-                              |               |-      |           |-      |Replaced by MatOps
MatAddTiled scr|     No|Yes   |**Yes**|   -   |1D        |-      |-                              |               |-      |           |-      |Replaced by MatOps
MatSub         |     No|Yes   |**Yes**|   -   |1D,4D     |1D,4D  |-                              |               |-      |           |-      |Replaced by MatOps
MatSubTiled    |     No|Yes   |**Yes**|   -   |1D,2D,4D  |1D     |-                              |               |-      |           |-      |Replaced by MatOps
MatSubTiled scr|     No|Yes   |**Yes**|   -   |1D        |-      |-                              |               |-      |           |-      |Replaced by MatOps
MultiplyTiled  |Yes    |Yes   |**Yes**|   -   |2D,4D     |1D     |-                              |               |-      |           |-      |Replaced by MatOps
DivideTiled    |Yes    |Yes   |**Yes**|   -   |4D,2D     |1D     |-                              |               |-      |           |-      |Replaced by MatOps
Matmul_Scalar  |Yes    |Yes   |**Yes**|   -   |1D,3D     |-      |-                              |               |-      |           |-      |Replaced by MatOps
MatOps         |Yes    |Yes   |**Yes**|**Yes**|4D,3D,2D,1D|4D,3D,2D,1D,0D|-                              |               |-      |           |-      |ADD,SUB,MUL_ELEMENT,DIV_ELEMENT, shapes could be different
Transpose      |Yes    |Yes   |**Yes**|**Yes**|3D        |-      |-                              |               |-      |           |-      |--
Conv2d         |Yes    |Yes   |**Yes**|**Yes**|4D        |-      |-                              |overrideDim2   |-1     |           |-      |3x Less performance compared to the tensorflow
ReLU           |Yes    |Yes   |**Yes**|**Yes**|2D,4D     |-      |-                              |               |-      |           |-      |--
Matmul         |Yes    |Yes   |**Yes**|**Yes**|2D,3D     |2D,3D  |-                              |               |-      |           |-      |20x Less performance compared to the tensorflow
Square         |Yes    |Yes   |**Yes**|**Yes**|3D        |-      |-                              |               |-      |           |-      |--
Sqrt           |Yes    |Yes   |**Yes**|**Yes**|3D        |-      |-                              |               |-      |           |-      |--
TopK           |Yes    |Yes   |**Yes**|**Yes**|3D        |-      |-                              |axis           |2      |k          |20     |From PointNet++
Gather         |Yes    |Yes   |**Yes**|**Yes**|3D        |3D     |-                              |indices_axis   |1      |           |-      |From PointNet++

