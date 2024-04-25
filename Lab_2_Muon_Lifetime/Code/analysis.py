import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from sklearn.metrics import r2_score
import scipy.stats as ss
from scipy.signal import savgol_filter

plt.style.use("bmh")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def bin_to_time(bin):
    m, m_unc = 0.0012032098687220002, 1.367386544213397e-07
    return m * bin, m_unc * bin


file_name = "Lab_2_Muon_Lifetime/Data (Testing)/maybe4_11.Chn"
data = np.fromfile(file_name, dtype=np.int32, count=2**14, offset=32)

bins = np.arange(0, 2**14, 1)
time_bins_and_unc = bin_to_time(bins)
time_bins = time_bins_and_unc[0]

low_end = np.argmax(data)
hi_end = 2**12

data_new = data[low_end:hi_end]
data_new = savgol_filter(data_new, window_length=50, polyorder=3)

bins_new = bins[low_end:hi_end]
time = bin_to_time(bins_new)[0]
t_0 = time[0]


def curveFitExp(x, tau, A, B):
    return A * np.exp(-x / tau) + B


def curveFitLin(x, m, b):
    return m * x + b


# params, param_cov = curve_fit(
#     f=curveFitExp,
#     xdata=time - t_0,
#     ydata=data_new,
#     sigma=np.sqrt(data_new),
#     p0=(2.2, data_new[0], data_new[-1]),
# )

# params, param_cov = curve_fit(
#     f=curveFitLin,
#     xdata=(time - t_0)[2000:10000],
#     ydata=np.log((np.sum(data_new) - np.cumsum(data_new))[2000:10000]),
#     # sigma=np.sqrt(data_new),
#     # p0=(2.2, data_new[0], data_new[-1]),
# )
# file_name = "Data\ (Testing)/calibration.Chn"
# data_cal = np.fromfile(file_name, dtype=np.int32, count=2**14, offset=32)
# bins = np.arange(0, 2**14, 1)
plt.plot(data)
plt.xlabel("Bins")
plt.ylabel("Counts")
plt.title("Counts vs. Bins")
plt.xlim(left = 2500)
plt.ylim(top = 50, bottom = 0)
plt.show()
