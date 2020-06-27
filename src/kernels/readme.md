# Vendors
Currently, only Xilinx devices are supported.

## Xilinx Kernels
| # | Name | Sub | Bus | Bus Count | Burst R/W | PE | Dataflow |
|---|------|-----|-----|-----------|-----------|----|----------|
|1.1|concat2|ConcatLastDimSubVec_V1|512b|2|Yes|-|No|
|1.2|concat2|ConcatLastDimSuperVec_V1|512b|2|No|-|No|
|2|conv2|-|512b|**4**|Yes|-|Yes|
|3|datamover|-|512b|1<=N<=4|Yes|-|No|
|4|gather|-|512b|3|Yes|-|Yes|
|5|matmul|-|512b|2|Yes|-|No|
|6|matops|-|512b|2|Yes|-|Yes|
|7.1|pad_unpad|PadLastDimSuperVec|512b|1|No|-|No|
|7.2|pad_unpad|UnpadLastDimSuperVec|512b|1|No|-|No|
|8.1|reduce|ReduceSum3Axis2_V1|512b|2|Yes|**2**|No|
|8.2|reduce|ReduceSumRank4Axes012_V3|512b|**2**|Yes|-|Yes|
|8.3|reduce|ReduceMax3Axis1_V2|512b|2|Yes|**2**|No|
|9.1|relu_sqrt_square|Relu_V1|512b|1|Yes|-|No|
|9.2|relu_sqrt_square|Sqrt_V1|512b|1|Yes|-|No|
|9.3|relu_sqrt_square|Square_V1|512b|1|Yes|-|No|
|10|tile|TileRank2Axis1|512b|1|No|-|No|
|10|tile|TileRank2Axis2|512b|1|No|-|No|
|11|topk_mergesort|-|512b|2|Yes|Multiple|Yes|
|12|transpose|-|512b|**2**|No|-|Yes|