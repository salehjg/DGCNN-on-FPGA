# Vendors
Currently, only Xilinx devices are supported.

## Xilinx Kernels
| # | Name | Sub | Bus | Bus Count | Burst R/W | PE | Dataflow |
|---|------|-----|-----|-----------|-----------|----|----------|
|1.1|concat2|ConcatLastDimSubVec_V2|512b|2|Yes|-|No|
|1.2|concat2|ConcatLastDimSuperVec_V2|512b|2|Yes|-|Yes|
|2|conv2|-|512b|4|Yes|-|Yes|
|3|datamover|-|512b|1<=N<=4|Yes|-|Yes|
|4|gather|-|512b|3|Yes|-|Yes|
|5|matmul|-|512b|2|Yes|-|No|
|6|matops|-|512b|2|Yes|-|Yes|
|7.1|pad_unpad|PadLastDimSuperVec|512b|1|Yes|-|No|
|7.2|pad_unpad|UnpadLastDimSuperVec|512b|1|Yes|-|No|
|8.1|reduce|ReduceSumRank3Axis2_V2|512b|**1**|Yes|-|Yes|
|8.2|reduce|ReduceSumRank4Axes012_V4|512b|**1**|Yes|-|No|
|8.3|reduce|ReduceMaxRank3Axis1_V3|512b|**1**|Yes|-|No|
|9.1|relu_sqrt_square|Relu_V1|512b|1|Yes|-|No|
|9.2|relu_sqrt_square|Sqrt_V1|512b|1|Yes|-|No|
|9.3|relu_sqrt_square|Square_V1|512b|1|Yes|-|No|
|10.1|tile|TileRank2Axis1|512b|1|Yes|-|No|
|10.2|tile|TileRank2Axis2|512b|1|Yes|-|No|
|11|topk_mergesortdf_pe|-|512b|2|Yes|Multiple|Yes|
|12|transpose|-|512b|1|No|-|No|