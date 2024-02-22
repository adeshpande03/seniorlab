import numpy as np
from pprint import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def bins_to_energy():
    """

    References

    https://www.gammaspectacular.com/blue/gamma_spectra/cs137-spectrum
    https://www.gammaspectacular.com/blue/gamma_spectra/ba133-spectrum
    https://www.nrc.gov/docs/ML1122/ML11229A699.pdf

    """

    cs_filepath = "./Data/Cs_137_Calibration.Chn"
    cs_data = list(np.fromfile(cs_filepath, dtype=np.int32, count=16384, offset=32))
    ba_filepath = "./Data/Ba_133_Calibration.Chn"
    ba_data = list(np.fromfile(ba_filepath, dtype=np.int32, count=16384, offset=32))

    data_Cs = cs_data[9000:12000]
    x_Cs = np.linspace(9000, 12000, len(data_Cs))
    data_Ba = ba_data[5650:6300]
    x_Ba = np.linspace(5650, 6300, len(data_Ba))
    Cs_gamma_peak = 661.6
    Ba_gamma_peak = 356.0

    def curveFit(x, mu, sigma, A, v_off):
        return A * np.exp(-(((x - mu) / sigma) ** 2)) + v_off

    def peak_x(x_data, y_data, p0):
        param, param_cov = curve_fit(curveFit, x_data, y_data, p0=p0)
        unc = np.sqrt(np.diag(param_cov)[0])
        return param[0], unc

    Cs_p0 = (10500, 500, 4000, 0)
    Ba_p0 = (6000, 200, 1000, 0)
    peak_Cs, unc_Cs = peak_x(x_Cs, data_Cs, Cs_p0)
    peak_Ba, unc_Ba = peak_x(x_Ba, data_Ba, Ba_p0)

    # print(peak_Cs, " \pm ", unc_Cs)
    # print(peak_Ba, " \pm ", unc_Ba)
    m = (Cs_gamma_peak - Ba_gamma_peak) / (peak_Cs - peak_Ba)
    b = -m * peak_Cs + Cs_gamma_peak
    vals = {"m": m, "b": b}
    print(vals)
    # print(f"{m} * x + {b}")

    y = np.fromfile(cs_filepath, dtype=np.int32, count=16384, offset=32)
    x = np.linspace(0, 2**14, 2**14)
    y2 = np.fromfile(ba_filepath, dtype=np.int32, count=16384, offset=32)

    plt.yscale("log")
    plt.plot(x, y)
    plt.plot(x, y2)
    plt.title("Counts vs. Bin Numbers")
    plt.ylabel("Counts")
    plt.xlabel("Bin Number")
    plt.show()
    return vals


if __name__ == "__main__":
    bins_to_energy()
