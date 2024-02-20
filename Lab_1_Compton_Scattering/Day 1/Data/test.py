import numpy as np
from pprint import * 
import matplotlib.pyplot as plt
filepath = './Cs_137_Calibration.Chn'
data = list(np.fromfile(filepath, dtype=np.int32, count=16384, offset=32))
# print([(i, j) for( i, j) in enumerate(data) if j > 500])
y = np.fromfile(filepath, dtype=np.int32, count=16384, offset=32)
x = np.linspace(0, 2**14, 2**14)
plt.yscale("log")  
plt.plot(x, y)
plt.show()