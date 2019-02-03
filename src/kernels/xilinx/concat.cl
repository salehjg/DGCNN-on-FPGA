kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void task_concat( __global float* inputTn1,
            __global float* inputTn2,
            __global float* outputTn,

			unsigned int dimA0,
			unsigned int dimA1,
			unsigned int dimA2,
			unsigned int dimA3,

			unsigned int dimB0,
			unsigned int dimB1,
			unsigned int dimB2,
			unsigned int dimB3
          )
{
	unsigned int  dimR0,dimR1,dimR2,dimR3;
    dimR0 = dimA0;
    dimR1 = dimA1;
    dimR2 = dimA2;
    dimR3 = dimA3 + dimB3;
    unsigned long indxS1,indxS2,indxD;

    for(int d0=0;d0<dimA0;d0++){
		for(int d1=0;d1<dimA1;d1++){
			for(int d2=0;d2<dimA2;d2++){
				for(int d3=0;d3<dimA3;d3++){
					indxS1 = d0*dimA1*dimA2*dimA3 +
							 d1*dimA2*dimA3+
							 d2*dimA3+
							 d3;
					indxD = (d0)*dimR1*dimR2*dimR3 +
							(d1)*dimR2*dimR3+
							(d2)*dimR3+
							(d3);
					outputTn[indxD] = inputTn1[indxS1];
				}
			}
		}
	}

	for(int d0=0;d0<dimB0;d0++){
		for(int d1=0;d1<dimB1;d1++){
			for(int d2=0;d2<dimB2;d2++){
				for(int d3=0;d3<dimB3;d3++){
					indxS2 = d0*dimB1*dimB2*dimB3 +
							 d1*dimB2*dimB3+
							 d2*dimB3+
							 d3;
					indxD  = (d0+0)*dimR1*dimR2*dimR3 +
							 (d1+0)*dimR2*dimR3+
							 (d2+0)*dimR3+
							 (d3+dimA3);
					outputTn[indxD] = inputTn2[indxS2];
				}
			}
		}
	}
}
