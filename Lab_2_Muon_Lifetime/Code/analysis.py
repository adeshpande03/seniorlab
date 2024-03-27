import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from sklearn.metrics import r2_score
import scipy.stats as ss

plt.style.use("bmh")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def bin_to_time(bin):
    m, m_unc = 0.0012032098687220002, 1.367386544213397e-07
    return m * bin, m_unc * bin


file_name = "Lab_2_Muon_Lifetime/Data (Testing)/maybe2.Chn"
data = np.fromfile(file_name, dtype=np.int32, count=2**14, offset=32)

bins = np.arange(0, 2**14, 1)
time_bins_and_unc = bin_to_time(bins)
time_bins = time_bins_and_unc[0]
noise_avg = np.mean(data[time_bins > 5.0])
cleaned_data = data - noise_avg
cleaned_data[cleaned_data < 0] = 0
data_peak = np.argmax(cleaned_data)
cutoff_ind = np.where(abs(time_bins - 5.0) < 1e-3)[0][0]
t0 = time_bins[data_peak]
new_time = time_bins[data_peak:cutoff_ind]
new_data = np.cumsum(cleaned_data[data_peak:cutoff_ind][::-1])[::-1]
plt.semilogy(
    time_bins[data_peak:cutoff_ind] - t0,
    np.cumsum(cleaned_data[data_peak:cutoff_ind][::-1])[::-1],
)
plt.show()
plt.clf()


def curveFit_exp(x, tau, A):
    return A * np.exp(-x / tau)


def curveFit_lin(x, m, b):
    return m * x + b


param, param_cov = curve_fit(
    xdata=time_bins[data_peak:cutoff_ind] - t0,
    ydata=np.cumsum(cleaned_data[data_peak:cutoff_ind][::-1])[::-1],
    f=curveFit_exp,
    p0=(2.2, 2000),
)

plt.plot(
    time_bins[data_peak:cutoff_ind] - t0,
    np.cumsum(cleaned_data[data_peak:cutoff_ind][::-1])[::-1],
)
plt.plot(
    time_bins[data_peak:cutoff_ind] - t0,
    curveFit_exp(time_bins[data_peak:cutoff_ind] - t0, *param),
)
plt.show()
plt.clf()

time_start = np.where(abs(new_time - 1.5) < 1e-3)[0][0]
time_end = np.where(abs(new_time - 3.5) < 1e-3)[0][0]
new_time = new_time[time_start:time_end]
new_data = np.log(new_data[time_start:time_end])
plt.plot(new_time, new_data)

param, param_cov = curve_fit(xdata=new_time, ydata=new_data, f=curveFit_lin)
plt.plot(new_time, new_data)
plt.plot(new_time, curveFit_lin(new_time, *param))
plt.show()
print(param)
print(-1 / param[0])
print(np.sqrt(param_cov[0][0]) / param[0] ** 2)
