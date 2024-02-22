import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from sklearn.metrics import r2_score
import scipy.stats as ss
from bins_to_energy import bins_to_energy
from pprint import *


def chn_to_readable(filename):
    return np.array(np.fromfile(filename, dtype=np.int32, count=16384, offset=32))


def main():
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    vals = bins_to_energy()
    m = vals["m"]
    b = vals["b"]
    # print(m, b)

    def vals_to_energy(x, unc_x):
        E = m * x + b
        unc_E = m * unc_x
        return E, unc_E

    files = {
        "0": ["Data/0_degrees_rod_1hr.Chn", "Data/0_degrees_background_1hr.Chn"],
        "10": ["Data/10_degrees_rod_1hr.Chn", "Data/10_degrees_background_1hr.Chn"],
        "20": ["Data/20_degrees_rod_1hr.Chn", "Data/20_degrees_background_1hr.Chn"],
        "30": ["Data/30_degrees_rod_3hr.Chn", "Data/30_degrees_background_3hr.Chn"],
        "40": ["Data/40_degrees_rod_18hr.Chn", "Data/40_degrees_background_18hr.Chn"],
        "50": ["Data/50_degrees_rod_18hr.Chn", "Data/50_degrees_background_18hr.Chn"],
        "60": ["Data/60_degrees_rod_18hr.Chn", "Data/60_degrees_background_18hr.Chn"],
        "70": ["Data/70_degrees_rod_18hr.Chn", "Data/70_degrees_background_18hr.Chn"],
        "80": ["Data/80_degrees_rod_24hr.Chn", "Data/80_degrees_background_24hr.Chn"],
    }

    data = {}
    x = np.linspace(0, 2**14, 2**14)
    for a in files:
        # if "8" in a[0]:
        #     y = chn_to_readable(files[a][0])
        #     y2 = chn_to_readable(files[a][1])
        #     subt = np.subtract(y, y2)
        #     data[a] = [chn_to_readable(files[a][0]), chn_to_readable(files[a][1])]
        #     # plt.yscale("log")
        #     plt.plot(x, y)
        #     plt.show()
        #     # plt.yscale("log")
        #     plt.plot(x, y2)
        #     plt.show()
        #     # plt.yscale("log")
        #     plt.plot(x, subt, marker='.')
        #     plt.show()
        data[a] = [chn_to_readable(files[a][0]), chn_to_readable(files[a][1])]

    def curveFit(x, mu, sigma, A, v_off):
        return A * np.exp(-(((x - mu) / sigma) ** 2)) + v_off

    def df_dmu(x, mu, sigma, A, v_off):
        return 2 * A * (x - mu) * np.exp(-(((x - mu) / sigma) ** 2)) / sigma**2

    def peak_x(angle, x_data, y_data, p0, plot_flag=False):
        param, param_cov = curve_fit(curveFit, x_data, y_data, p0=p0)
        peak = param[0]
        peak_unc = abs(param[1])
        if plot_flag:
            plt.scatter(x_data, y_data, color=colors[0], label="Data", alpha=0.5)
            plt.plot(
                x_data, curveFit(x_data, *param), color=colors[1], label="Curve Fit"
            )
            plt.xlabel("Bin")
            plt.ylabel("Count")
            plt.legend()
            plt.title(f"{angle} degrees\nPeak = {peak:.2f}$\pm${peak_unc:.2f}")
            plt.show()
        return param[0], abs(param[1])

    data_range = {
        "0": [10000, 12000],
        "10": [10000, 12500],
        "20": [9000, 12000],
        "30": [9000, 11250],
        "40": [8000, 10000],
        "50": [7000, 9500],
        "60": [6000, 8250],
        "70": [5800, 7000],
        "80": [5400, 6000],
    }

    E_vals = {}

    for angle in data:
        lo_end = data_range[angle][0]
        hi_end = data_range[angle][1]
        y_data = data[angle][0][lo_end:hi_end] - data[angle][1][lo_end:hi_end]
        x_data = np.arange(data_range[angle][0], data_range[angle][1], 1)
        p0 = (
            (lo_end + hi_end) / 2,
            (hi_end - lo_end) / 2,
            y_data[(hi_end - lo_end) // 2],
            y_data[0],
        )
        peak_curr = peak_x(angle, x_data, y_data, p0, plot_flag=True)
        print(angle, ":", vals_to_energy(*peak_curr))
        E_vals[angle] = vals_to_energy(*peak_curr)

    def compton(x):
        return E_vals["0"][0] / (1 + E_vals["0"][0] / 511 * (1 - np.cos(np.radians(x))))

    def compton_lin(x):
        return 1 + E_vals["0"][0] / 511 * (1 - np.cos(np.radians(x)))

    compton_x = np.array([int(key) for key in E_vals.keys()])
    compton_y = np.array([value[0] for value in E_vals.values()])
    y_err = np.array([value[1] for value in E_vals.values()])
    x_err = np.ones(len(compton_x))
    plt.errorbar(
        compton_x, compton_y, yerr=y_err, fmt="o", color=colors[0], label="Data"
    )
    plt.plot(
        np.linspace(0, 90, 100),
        compton(np.linspace(0, 90, 100)),
        color=colors[1],
        label="Theoretical Plot",
    )

    plt.xlabel("$\\theta$ (Degrees)")
    plt.ylabel("Energy (KeV)")
    plt.title("Compton Scattering")
    plt.legend()

    plt.show()

    compton_x_lin = 1 - np.cos(np.radians(compton_x))
    compton_y_lin = E_vals["0"][0] / compton_y
    y_err_lin = compton_y_lin * y_err / compton_y
    plot_x = 1 - np.cos(np.radians(np.linspace(0, 90, 100)))
    plot_y = compton_lin(np.linspace(0, 90, 100))

    plt.errorbar(
        compton_x_lin,
        compton_y_lin,
        yerr=y_err_lin,
        fmt="o",
        color=colors[0],
        label="Data",
    )
    plt.plot(plot_x, plot_y, color=colors[1], label="Theoretical Plot")

    r2 = r2_score(compton_y_lin, compton_lin(compton_x))

    plt.xlabel("$1-cos(\\theta)$")
    plt.ylabel("$E(0)/E(\\theta)$")
    plt.title(f"Compton Scattering, $R^2={r2:.3f}$")
    plt.legend()

    plt.show()

    def KN(x):
        r_e2 = 7.94  # fm^2
        ep = E_vals["0"][0] / 511
        lambda_ratio = 1 / (1 + ep * (1 - np.cos(np.radians(x))))
        return (
            (r_e2 / 2)
            * lambda_ratio**2
            * (lambda_ratio + 1 / lambda_ratio - np.sin(np.radians(x)) ** 2)
        )

    def KN_exp(angle):
        r_e2 = 7.94  # fm^2
        ep = E_vals["0"][0] / 511
        lambda_ratio = E_vals[angle][0] / E_vals["0"][0]
        lamb_err = E_vals[angle][1] / E_vals["0"][0]

        def KN_val_helper(lamb_val):
            return (
                (r_e2 / 2)
                * lamb_val**2
                * (lamb_val + 1 / lamb_val - np.sin(np.radians(int(angle))) ** 2)
            )

        KN_val = KN_val_helper(lambda_ratio)
        KN_err = abs(
            (
                KN_val_helper(lambda_ratio + lamb_err)
                - KN_val_helper(lambda_ratio - lamb_err)
            )
            / 2
        )
        return KN_val, KN_err

    def Thomson(angle):
        r_e2 = 7.94  # fm^2
        return (r_e2 / 2) * (1 + np.cos(np.radians(angle)) ** 2)

    KN_x = np.array([int(key) for key in E_vals.keys()])
    KN_y = np.array([KN_exp(angle)[0] for angle in E_vals])
    KN_y_err = np.array([KN_exp(angle)[1] for angle in E_vals])
    print(KN_y_err)

    plt.errorbar(KN_x, KN_y, yerr=KN_y_err, fmt="o", color=colors[0], label="Data")
    plt.plot(
        np.linspace(0, 90, 100),
        KN(np.linspace(0, 90, 100)),
        color=colors[1],
        label="Klein-Nishina Plot",
    )
    plt.plot(
        np.linspace(0, 90, 100),
        Thomson(np.linspace(0, 90, 100)),
        linestyle="--",
        color=colors[2],
        label="Thomson Plot",
    )

    plt.xlabel("$\\theta$ (Degrees)")
    plt.ylabel("$d\sigma/d\Omega\ (\mathrm{fm}^2/\mathrm{sr})$ ")
    plt.title("Differential Cross-Section\nAngular Relation")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
