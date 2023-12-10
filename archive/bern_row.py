
import numpy as np
from utils import estimate_batch_rle_complexity

num = 200
tcx = 0
for i in range(num):
    bern_row = np.random.randint(0,2,size=(1,32))
    cx = estimate_batch_rle_complexity(bern_row)
    tcx += cx
print(f"Average: {tcx/num}")

bern_row = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])
print(estimate_batch_rle_complexity(bern_row))

