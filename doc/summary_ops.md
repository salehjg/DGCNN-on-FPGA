# Multiplatform Implementation Summary

## Task List
- [ ] Blah blah blah


## Ops
Op Name        | Batch | CPU  | OCL TASK | OCL NDRng| Optimized|Shape1    | Shape2| Combinations                  | Sett1         |Val1   |Sett2      |Val2   | Notes|
---            | ---   | ---  | ---      | ---      | ---      | ---      | ---   | --------------------          | ---           | ---   | ---       | ---   |  --- |
Concat2        |     No|Yes   |**Yes**   |        No|        No|4D        |4D     |-                              |Concat2        |3      |           |-      |--
Sqrt           |Yes    |Yes   |**Yes**   |**Yes**   |        No|3D        |-      |-                              |               |-      |           |-      |--
ReduceMax      |     No|Yes   |**Yes**   |**Yes**   |        No|4D        |-      |-                              |reductionDim   |1,2    |           |-      |--
ReduceSum4D    |     No|Yes   |**Yes**   |**BROKEN**|        No|4D        |-      |{1-1-1-0}                      |               |-      |           |-      |--
ReduceSum      |     No|Yes   |**Yes**   |        No|        No|2D,3D     |-      |{3D: 0-0-1}, {2D: 0-1-0}       |               |-      |           |-      |--
Mean           |     No|Yes   |        No|        No|        No|2D,4D     |-      |{1-0-0-0}, {1-1-1-0}           |               |-      |           |-      |--
Variance       |     No|Yes   |        No|        No|        No|2D,4D     |-      |{2D: 1-0-0-0}, {4D: 1-1-1-0}   |               |-      |           |-      |--
Tile           |     No|Yes   |**Yes**   |        No|        No|3D,4D     |-      |-                              |tileAxis       |1,2    |tileCount  |20,1024|only tileAxis=2 implemented
MatOps         |Yes    |Yes   |        No|        No|        No|4D,3D,2D,1D|4D,3D,2D,1D,0D|-                              |               |-      |           |-      |ADD,SUB,MUL_ELEMENT,DIV_ELEMENT, shapes could be different
Transpose      |Yes    |Yes   |**Yes**   |        No|        No|3D        |-      |-                              |               |-      |           |-      |--
Conv2d         |Yes    |Yes   |        No|        No|        No|4D        |-      |-                              |overrideDim2   |-1     |           |-      |3x Less performance compared to the tensorflow
ReLU           |Yes    |Yes   |**Yes**   |        No|        No|2D,4D     |-      |-                              |               |-      |           |-      |--
Matmul         |Yes    |Yes   |        No|        No|        No|2D,3D     |2D,3D  |-                              |               |-      |           |-      |20x Less performance compared to the tensorflow
Square         |Yes    |Yes   |**Yes**   |        No|        No|3D        |-      |-                              |               |-      |           |-      |--
TopK           |Yes    |Yes   |        No|        No|        No|3D        |-      |-                              |axis           |2      |k          |20     |From PointNet++
Gather         |Yes    |Yes   |        No|        No|        No|3D        |3D     |-                              |indices_axis   |1      |           |-      |From PointNet++

