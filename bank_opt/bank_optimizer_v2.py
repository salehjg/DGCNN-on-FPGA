import numpy as np
import kernel_obj as h

# Allowed banks per kernel (banks 0 to 3)
common_datamover = h.KernelObj('datamover', [1, 2], 12, 0, 1, 1, 0)
obj_transpose = h.KernelObj('transpose', [1, 2], 10, 1, 1, 3, 0)
obj_matmul = h.KernelObj('matmul', [1, 2], 8, 14, 4, 6, 0)
obj_matops = h.KernelObj('matops', [1, 2], 9, 12, 8, 17, 0)
obj_relusqrtsquare = h.KernelObj('relusqrtsquare', [1, 2], 2, 2, 1, 3, 0)
obj_reduce = h.KernelObj('reduce', [1, 2], 7, 7, 7, 15, 0)
obj_tile = h.KernelObj('tile', [1, 2], 4, 0, 4, 8, 0)
obj_topk = h.KernelObj('topk', [1,2], 14, 1, 8, 20, 0)
obj_gather = h.KernelObj('gather', [1, 2], 7, 3, 4, 7, 0)
obj_concat = h.KernelObj('concat', [1,2], 8, 7, 16, 36, 0)
obj_padunpad = h.KernelObj('padunpad', [1,2], 2, 1, 4, 7, 0)
obj_conv = h.KernelObj('conv', [1,2], 42, 32, 13, 56, 0)

def get_objective(
    transpose_in,
    transpose_out,

    matmul_in1,
    matmul_in2,
    matmul_out,

    matops_in1,
    matops_in2,
    matops_out,

    relusqrtsquare_in,
    relusqrtsquare_out,

    reduce_in,
    reduce_out,

    tile_in,
    tile_out,

    topk_in,
    topk_out,

    gather_in1,
    gather_in2,
    gather_out,

    concat_in1,
    concat_in2,
    concat_out,

    padunpad_in,
    padunpad_out,

    conv_in,
    conv_w,
    conv_b,
    conv_out):

    unknownTag = 1 # the default oclTensorF/I memory bank
    undefined_tag = unknownTag # the default weight tensor bank(=unknownTag)

    objective = abs(unknownTag-transpose_in) + abs(unknownTag-matmul_in1) + abs(transpose_out-matmul_in2) + abs(matmul_out-matops_in1) + abs(unknownTag-matops_in2) + abs(unknownTag-relusqrtsquare_in) + abs(relusqrtsquare_out-reduce_in) + abs(reduce_out-tile_in) + abs(reduce_out-tile_in) + abs(tile_out-matops_in1) + abs(tile_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-topk_in) + abs(unknownTag-gather_in1) + abs(topk_out-gather_in2) + abs(unknownTag-tile_in) + abs(gather_out-matops_in1) + abs(tile_out-matops_in2) + abs(tile_out-concat_in1) + abs(matops_out-concat_in2) + abs(concat_out-conv_in) + abs(conv_w-conv_w) + abs(conv_b-conv_b) + abs(conv_w-padunpad_in) + abs(unknownTag-padunpad_in) + abs(conv_out-reduce_in) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(conv_out-reduce_in) + abs(conv_out-reduce_in) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(conv_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-relusqrtsquare_in) + abs(matops_out-matops_in1) + abs(relusqrtsquare_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-relusqrtsquare_in) + abs(relusqrtsquare_out-conv_in) + abs(conv_w-conv_w) + abs(conv_b-conv_b) + abs(conv_w-padunpad_in) + abs(unknownTag-padunpad_in) + abs(conv_out-reduce_in) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(conv_out-reduce_in) + abs(conv_out-reduce_in) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(conv_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-relusqrtsquare_in) + abs(matops_out-matops_in1) + abs(relusqrtsquare_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-relusqrtsquare_in) + abs(relusqrtsquare_out-reduce_in) + abs(reduce_out-conv_in) + abs(conv_w-conv_w) + abs(conv_b-conv_b) + abs(conv_out-reduce_in) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(conv_out-reduce_in) + abs(conv_out-reduce_in) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(conv_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-relusqrtsquare_in) + abs(matops_out-matops_in1) + abs(relusqrtsquare_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-relusqrtsquare_in) + abs(relusqrtsquare_out-reduce_in) + abs(reduce_out-matmul_in1) + abs(matmul_in2-matmul_in2) + abs(matmul_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-reduce_in) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-reduce_in) + abs(matops_out-reduce_in) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-relusqrtsquare_in) + abs(matops_out-matops_in1) + abs(relusqrtsquare_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-relusqrtsquare_in) + abs(relusqrtsquare_out-matmul_in1) + abs(matmul_in2-matmul_in2) + abs(matmul_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-reduce_in) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-reduce_in) + abs(matops_out-reduce_in) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-relusqrtsquare_in) + abs(matops_out-matops_in1) + abs(relusqrtsquare_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-relusqrtsquare_in) + abs(undefined_tag-matops_in1) + abs(unknownTag-matops_in2) + abs(relusqrtsquare_out-matmul_in1) + abs(undefined_tag-matmul_in2) + abs(matmul_out-matops_in1) + abs(matops_out-matops_in2) + abs(unknownTag-matmul_in1) + abs(unknownTag-matmul_in2) + abs(matmul_out-transpose_in) + abs(matmul_out-matmul_in1) + abs(transpose_out-matmul_in2) + abs(matmul_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matmul_out-relusqrtsquare_in) + abs(relusqrtsquare_out-reduce_in) + abs(reduce_out-tile_in) + abs(reduce_out-tile_in) + abs(tile_out-matops_in1) + abs(tile_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-topk_in) + abs(matmul_out-gather_in1) + abs(topk_out-gather_in2) + abs(matmul_out-tile_in) + abs(gather_out-matops_in1) + abs(tile_out-matops_in2) + abs(tile_out-concat_in1) + abs(matops_out-concat_in2) + abs(concat_out-conv_in) + abs(conv_w-conv_w) + abs(conv_b-conv_b) + abs(conv_w-padunpad_in) + abs(unknownTag-padunpad_in) + abs(conv_out-reduce_in) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(conv_out-reduce_in) + abs(conv_out-reduce_in) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(conv_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-relusqrtsquare_in) + abs(matops_out-matops_in1) + abs(relusqrtsquare_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-relusqrtsquare_in) + abs(relusqrtsquare_out-reduce_in) + abs(reduce_out-transpose_in) + abs(reduce_out-matmul_in1) + abs(transpose_out-matmul_in2) + abs(matmul_out-matops_in1) + abs(unknownTag-matops_in2) + abs(reduce_out-relusqrtsquare_in) + abs(relusqrtsquare_out-reduce_in) + abs(reduce_out-tile_in) + abs(reduce_out-tile_in) + abs(tile_out-matops_in1) + abs(tile_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-topk_in) + abs(reduce_out-gather_in1) + abs(topk_out-gather_in2) + abs(reduce_out-tile_in) + abs(gather_out-matops_in1) + abs(tile_out-matops_in2) + abs(tile_out-concat_in1) + abs(matops_out-concat_in2) + abs(concat_out-conv_in) + abs(conv_w-conv_w) + abs(conv_b-conv_b) + abs(conv_w-padunpad_in) + abs(unknownTag-padunpad_in) + abs(conv_out-reduce_in) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(conv_out-reduce_in) + abs(conv_out-reduce_in) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(conv_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-relusqrtsquare_in) + abs(matops_out-matops_in1) + abs(relusqrtsquare_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-relusqrtsquare_in) + abs(relusqrtsquare_out-reduce_in) + abs(reduce_out-transpose_in) + abs(reduce_out-matmul_in1) + abs(transpose_out-matmul_in2) + abs(matmul_out-matops_in1) + abs(unknownTag-matops_in2) + abs(reduce_out-relusqrtsquare_in) + abs(relusqrtsquare_out-reduce_in) + abs(reduce_out-tile_in) + abs(reduce_out-tile_in) + abs(tile_out-matops_in1) + abs(tile_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-topk_in) + abs(reduce_out-gather_in1) + abs(topk_out-gather_in2) + abs(reduce_out-tile_in) + abs(gather_out-matops_in1) + abs(tile_out-matops_in2) + abs(tile_out-concat_in1) + abs(matops_out-concat_in2) + abs(concat_out-conv_in) + abs(conv_w-conv_w) + abs(conv_b-conv_b) + abs(conv_w-padunpad_in) + abs(unknownTag-padunpad_in) + abs(conv_out-reduce_in) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(conv_out-reduce_in) + abs(conv_out-reduce_in) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(conv_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-relusqrtsquare_in) + abs(matops_out-matops_in1) + abs(relusqrtsquare_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-relusqrtsquare_in) + abs(relusqrtsquare_out-reduce_in) + abs(reduce_out-transpose_in) + abs(reduce_out-matmul_in1) + abs(transpose_out-matmul_in2) + abs(matmul_out-matops_in1) + abs(unknownTag-matops_in2) + abs(reduce_out-relusqrtsquare_in) + abs(relusqrtsquare_out-reduce_in) + abs(reduce_out-tile_in) + abs(reduce_out-tile_in) + abs(tile_out-matops_in1) + abs(tile_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-topk_in) + abs(reduce_out-gather_in1) + abs(topk_out-gather_in2) + abs(reduce_out-tile_in) + abs(gather_out-matops_in1) + abs(tile_out-matops_in2) + abs(tile_out-concat_in1) + abs(matops_out-concat_in2) + abs(concat_out-conv_in) + abs(conv_w-conv_w) + abs(conv_b-conv_b) + abs(conv_w-padunpad_in) + abs(unknownTag-padunpad_in) + abs(conv_out-reduce_in) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(conv_out-reduce_in) + abs(conv_out-reduce_in) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(conv_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-relusqrtsquare_in) + abs(matops_out-matops_in1) + abs(relusqrtsquare_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-relusqrtsquare_in) + abs(relusqrtsquare_out-reduce_in) + abs(reduce_out-concat_in1) + abs(reduce_out-concat_in2) + abs(concat_out-concat_in1) + abs(reduce_out-concat_in2) + abs(concat_out-concat_in1) + abs(reduce_out-concat_in2) + abs(concat_out-conv_in) + abs(conv_w-conv_w) + abs(conv_b-conv_b) + abs(conv_out-reduce_in) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(conv_out-reduce_in) + abs(conv_out-reduce_in) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(conv_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-relusqrtsquare_in) + abs(matops_out-matops_in1) + abs(relusqrtsquare_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-relusqrtsquare_in) + abs(relusqrtsquare_out-reduce_in) + abs(reduce_out-matmul_in1) + abs(matmul_in2-matmul_in2) + abs(matmul_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-reduce_in) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-reduce_in) + abs(matops_out-reduce_in) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-relusqrtsquare_in) + abs(matops_out-matops_in1) + abs(relusqrtsquare_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-relusqrtsquare_in) + abs(relusqrtsquare_out-matmul_in1) + abs(matmul_in2-matmul_in2) + abs(matmul_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-reduce_in) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-reduce_in) + abs(matops_out-reduce_in) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(reduce_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_in1-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_out-matops_in2) + abs(matops_out-matops_in1) + abs(unknownTag-matops_in2) + abs(matops_out-relusqrtsquare_in) + abs(matops_out-matops_in1) + abs(relusqrtsquare_out-matops_in2) + abs(matops_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-matops_in1) + abs(matops_in2-matops_in2) + abs(matops_out-relusqrtsquare_in) + abs(relusqrtsquare_out-matmul_in1) + abs(matmul_in2-matmul_in2) + abs(matmul_out-matops_in1) + abs(matops_in2-matops_in2)

    return objective

def brute_force():
    min_datamover_launches = 9999999999999
    acceptable_combinations = []
    result_bank0 = 0
    result_bank1 = 0
    result_bank2 = 0
    result_bank3 = 0
    for transpose in obj_transpose.possible_banks:
        for matmul in obj_matmul.possible_banks:
            for matops in obj_matops.possible_banks:
                for rss in obj_relusqrtsquare.possible_banks:
                    for _reduce in obj_reduce.possible_banks:
                        for tile in obj_tile.possible_banks:
                            for topk in obj_topk.possible_banks:
                                for gather in obj_gather.possible_banks:
                                    for concat in obj_concat.possible_banks:
                                        for padunpad in obj_padunpad.possible_banks:
                                            for conv in obj_conv.possible_banks:
                                                val = get_objective(transpose,transpose, matmul,matmul,matmul, matops,matops,matops, rss,rss, _reduce,_reduce, tile,tile, topk,topk, gather,gather,gather, concat,concat,concat, padunpad,padunpad, conv,conv,conv,conv)
                                                # m_axi's per kernel are considered here:
                                                currentparams = [transpose, matmul,matmul, matops,matops, rss, _reduce, tile, topk,topk , gather,gather,gather, concat,concat, padunpad, conv,conv,conv,conv]
                                                currentparams = np.array(currentparams)
                                                bank0 = np.sum(currentparams==0)
                                                bank1 = np.sum(currentparams==1)
                                                bank2 = np.sum(currentparams==2)
                                                bank3 = np.sum(currentparams==3)

                                                cloned_objs = [
                                                    obj_transpose.clone(transpose),
                                                    obj_matmul.clone(matmul),
                                                    obj_matops.clone(matops),
                                                    obj_relusqrtsquare.clone(rss),
                                                    obj_reduce.clone(_reduce),
                                                    obj_tile.clone(tile),
                                                    obj_topk.clone(topk),
                                                    obj_gather.clone(gather),
                                                    obj_concat.clone(concat),
                                                    obj_padunpad.clone(padunpad),
                                                    obj_conv.clone(conv)]

                                                stats_bank0 = h.SlrStats(-1, 0, 0, 0, 0, 0)
                                                stats_bank1 = h.SlrStats(-1, 0, 0, 0, 0, 0)
                                                stats_bank2 = h.SlrStats(-1, 0, 0, 0, 0, 0)
                                                stats_bank3 = h.SlrStats(-1, 0, 0, 0, 0, 0)

                                                for i in range(len(cloned_objs)):
                                                    if cloned_objs[i].assigned_bank == 0:
                                                        stats_bank0 = stats_bank0 + cloned_objs[i].util_stats
                                                    if cloned_objs[i].assigned_bank == 1:
                                                        stats_bank1 = stats_bank1 + cloned_objs[i].util_stats
                                                    if cloned_objs[i].assigned_bank == 2:
                                                        stats_bank2 = stats_bank2 + cloned_objs[i].util_stats
                                                    if cloned_objs[i].assigned_bank == 3:
                                                        stats_bank3 = stats_bank3 + cloned_objs[i].util_stats

                                                util_cond_bank0 = stats_bank0 < h.SlrStats(-1,100,100,100,100,100)
                                                util_cond_bank0 = True

                                                util_cond_bank1 = stats_bank1 < h.SlrStats(-1,80,80,80,80,80)
                                                #util_cond_bank1 = True

                                                # We are trying to make sure that SLR2(bank1) utilization stays within the limits.
                                                # Also by relaxing util_cond_bank2, the excess circuitry will be placed in the closest SLR (slr0)
                                                # this mitigates the timing and congestion violations due to long cross-slr routes.
                                                # Note that the connections between SLRs are like :
                                                #   SLR0 <---> SLR1 <---> SLR2
                                                # and here we are trying to minimize the routes that cross multiple SLRs.

                                                util_cond_bank2 = stats_bank2 < h.SlrStats(-1,100,100,100,100,100)
                                                util_cond_bank2 = True

                                                util_cond_bank3 = stats_bank3 < h.SlrStats(-1,100,100,100,100,100)
                                                util_cond_bank3 = True

                                                # 15 = 16 -1, 1 axi is reserved for DataMover
                                                if min_datamover_launches > val and \
                                                        bank0<=15 and bank1<=15 and \
                                                        bank2<=15 and bank3<=15 and \
                                                        abs(bank1-bank2)<10 and \
                                                        util_cond_bank0 and \
                                                        util_cond_bank1 and \
                                                        util_cond_bank2 and \
                                                        util_cond_bank3 and \
                                                        True:

                                                    result_bank0 = bank0
                                                    result_bank1 = bank1
                                                    result_bank2 = bank2
                                                    result_bank3 = bank3
                                                    min_datamover_launches = val
                                                    acceptable_combinations.append({
                                                        'datamover_launches':min_datamover_launches,
                                                        'combination':cloned_objs,
                                                        'per_bank_util':[stats_bank0,stats_bank1,stats_bank2,stats_bank3],
                                                        'axi_per_bank': [result_bank0,result_bank1,result_bank2,result_bank3]
                                                    })

    return acceptable_combinations


kernelnames = ["transpose","matmul","matops","relusqrtsquare","reduce","tile","topk","gather","concat","padunpad","conv"]
solutions = brute_force()
print('Solutions Count: ', len(solutions))
for solution in solutions:
    print('=====================================================================================================')
    kernels_on_bank1 = []
    kernels_on_bank2 = []
    for i in range(len(solution['combination'])):
        if solution['combination'][i].assigned_bank==1:
            kernels_on_bank1.append(solution['combination'][i].kernel_name)
        if solution['combination'][i].assigned_bank==2:
            kernels_on_bank2.append(solution['combination'][i].kernel_name)

    print('Required DataMover Launches: ' + str(solution['datamover_launches']),'\n')
    print('m_axi s on bank 0: ' + str(solution['axi_per_bank'][0]))
    print('m_axi s on bank 1: ' + str(solution['axi_per_bank'][1]))
    print('m_axi s on bank 2: ' + str(solution['axi_per_bank'][2]))
    print('m_axi s on bank 3: ' + str(solution['axi_per_bank'][3]),'\n')
    print('Kernels on bank 1 : ', kernels_on_bank1)
    print('Kernels on bank 2 : ', kernels_on_bank2,'\n')
    print('SLR usage for bank 0: ', solution['per_bank_util'][0])
    print('SLR usage for bank 1: ', solution['per_bank_util'][1])
    print('SLR usage for bank 2: ', solution['per_bank_util'][2])
    print('SLR usage for bank 3: ', solution['per_bank_util'][3])