# Gemm HLS:
    C_{SizeN*SizeM}=A_{SizeN*SizeK}*B{SizeK*SizeM}
    A_{SizeN*SizeK} = inputTn_{[B.N.K]*D1}
    B_{SizeK*SizeM} = weightTn_{D1*D2}

*So*:
	SizeN = B.N.K
	SizeK = D1
	SizeM = D2
	
# Shapes:
	SizeN = B.N.K=B*1024*20
	SizeK = D1=6, 64, 128, 320, 1024
	SizeM = D2=64, 128, 1024
	
# Parameters:
	x_M     = OuterTilesN (OuterTile Count!, NOT SIZE)
	y_M     = OuterTilesM (OuterTile Count!, NOT SIZE)
	x_{Pr}  = InnerTileSizeN (SIZE, NOT COUNT!)
	y_{Pr}  = kComputeTileSizeM 
              kComputeTileSizeN = 1
			  
# Aliases in the paper:
	## OuterTileSize: 
		x_{tot}, y_{tot} = OuterTileSizeN, OuterTileSizeM
        OuterTileSizeN = x_b * x_t
        OuterTileSizeM = y_b * y_t
	
	## Parallelisim Parameters:
        InnerTileSizeN = x_p
        1 = y_p
        kComputeTileSizeM = y_c
        1 = y_c
		
	## Number of PE Dataflow Units:
		N_p = (InnerTileSizeN/kComputeTileSizeN) = InnerTileSizeN / 1 = InnerTileSizeN = x_p
        So, N_p = x_p, and y_p = 1

	
# Hierarchy:
	1 MxN Matrix = x_M * y_M  mem tiles
	1 mem tile = x_b * y_b block tiles
	1 block tile = x_t * y_t compute tiles
	1 compute tile = x_p * (y_p=1) PEs
	1 PE = (x_c=1) * y_c CUs
	
# Editable Parameters:
    OuterTileSizeN = x_tot = x_b * x_t
    OuterTileSizeM = y_tot = y_b * y_t
    InnerTileSizeN = x_p
    kComputeTileSizeM = y_c

# Restrictions:
	SizeN % OuterTileSizeN = 0
	SizeM % OuterTileSizeM = 0
	SizeM % 16 = 0
    16 % kComputeTileSizeM = 0
	OuterTileSizeM % 16 = 0
	OuterTileSizeN % InnerTileSizeN = 0
	OuterTileSizeM % kComputeTileSizeM = 0
	InnerTileSizeN * kComputeTileSizeM <= OuterTileSizeM


	
