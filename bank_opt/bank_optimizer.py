import numpy as np

# Allowed banks per kernel (banks 0 to 3)
banks_transpose = [1,2]
banks_matmul = [1,2]
banks_matops = [1,2]
banks_relusqrtsquare = [1,2]
banks_reduce = [1,2]
banks_tile = [1,2]
banks_topk = [2]
banks_gather = [1,2]
banks_concat = [1,2]
banks_padunpad = [1,2]
banks_conv = [1]

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
    values = []
    minval = 9999999999999
    minparams = []
    result_bank0 = 0
    result_bank1 = 0
    result_bank2 = 0
    result_bank3 = 0
    for transpose in banks_transpose:
        for matmul in banks_matmul:
            for matops in banks_matops:
                for rss in banks_relusqrtsquare:
                    for _reduce in banks_reduce:
                        for tile in banks_tile:
                            for topk in banks_topk:
                                for gather in banks_gather:
                                    for concat in banks_concat:
                                        for padunpad in banks_padunpad:
                                            for conv in banks_conv:
                                                val = get_objective(transpose,transpose, matmul,matmul,matmul, matops,matops,matops, rss,rss, _reduce,_reduce, tile,tile, topk,topk, gather,gather,gather, concat,concat,concat, padunpad,padunpad, conv,conv,conv,conv)
                                                values.append(val)
                                                # m_axi's per kernel are considered here:
                                                currentparams = [transpose, matmul,matmul, matops,matops, rss, _reduce, tile, topk,topk , gather,gather,gather, concat,concat, padunpad, conv,conv,conv,conv]
                                                currentparams = np.array(currentparams)
                                                bank0 = np.sum(currentparams==0)
                                                bank1 = np.sum(currentparams==1)
                                                bank2 = np.sum(currentparams==2)
                                                bank3 = np.sum(currentparams==3)

                                                # 15 = 16 -1, 1 axi is reserved for Datamover
                                                if minval > val and bank0<=15 and bank1<=15 and bank2<=15 and bank3<=15 and abs(bank1-bank2)<2:
                                                    result_bank0 = bank0
                                                    result_bank1 = bank1
                                                    result_bank2 = bank2
                                                    result_bank3 = bank3
                                                    minval = val
                                                    minparams = [transpose,matmul,matops,rss,_reduce,tile,topk,gather,concat,padunpad,conv]

    return minval, minparams, [result_bank0,result_bank1,result_bank2,result_bank3]

minval, minparams, banks = brute_force()
print('=============================================')
print('Required DataMover Launches: ' + str(minval))
print('m_axi s on bank0: ' + str(banks[0]))
print('m_axi s on bank1: ' + str(banks[1]))
print('m_axi s on bank2: ' + str(banks[2]))
print('m_axi s on bank3: ' + str(banks[3]))
print('=============================================')
print('Memory Banks Per Kernel(each layer is limited to use only one memory bank):')
print('---------------------------------------------')
print('transpose_in: ' + str(minparams[0]))
print('transpose_out: ' + str(minparams[0]))
print('---------------------------------------------')
print('matmul_in1: ' + str(minparams[1]))
print('matmul_in2: ' + str(minparams[1]))
print('matmul_out: ' + str(minparams[1]))
print('---------------------------------------------')
print('matops_in1: ' + str(minparams[2]))
print('matops_in2: ' + str(minparams[2]))
print('matops_out: ' + str(minparams[2]))
print('---------------------------------------------')
print('relusqrtsquare_in: ' + str(minparams[3]))
print('relusqrtsquare_out: ' + str(minparams[3]))
print('---------------------------------------------')
print('reduce_in: ' + str(minparams[4]))
print('reduce_out: ' + str(minparams[4]))
print('---------------------------------------------')
print('tile_in: ' + str(minparams[5]))
print('tile_out: ' + str(minparams[5]))
print('---------------------------------------------')
print('topk_in: ' + str(minparams[6]))
print('topk_out: ' + str(minparams[6]))
print('---------------------------------------------')
print('gather_in1: ' + str(minparams[7]))
print('gather_in2: ' + str(minparams[7]))
print('gather_out: ' + str(minparams[7]))
print('---------------------------------------------')
print('concat_in1: ' + str(minparams[8]))
print('concat_in2: ' + str(minparams[8]))
print('concat_out: ' + str(minparams[8]))
print('---------------------------------------------')
print('padunpad_in: ' + str(minparams[9]))
print('padunpad_out: ' + str(minparams[9]))
print('---------------------------------------------')
print('conv_in: ' + str(minparams[10]))
print('conv_w: ' + str(minparams[10]))
print('conv_b: ' + str(minparams[10]))
print('conv_out: ' + str(minparams[10]))
print('=============================================')