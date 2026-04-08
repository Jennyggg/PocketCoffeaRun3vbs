import matplotlib.pyplot as plt
import numpy as np
import json
def plot_hist_with_errors(bin_edges, values, up_errors, down_errors, output_name):
    """
    Plot a histogram with asymmetric error bars and save as a PNG.

    Parameters
    ----------
    bin_edges : list or array
        The edges of the histogram bins (length N+1).
    values : list or array
        The bin contents (length N).
    up_errors : list or array
        The upward variations for each bin (length N).
    down_errors : list or array
        The downward variations for each bin (length N).
    output_name : str
        The output file name (e.g. "histogram.png").
    """

    # Convert to numpy arrays for safety
    bin_edges = np.array(bin_edges)
    values = np.array(values)
    up_errors = np.array(up_errors)
    down_errors = np.array(down_errors)
    up_errors = abs(up_errors-values)
    down_errors = abs(values-down_errors)
    # Compute bin centers and widths
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)

    # Create the figure
    plt.figure(figsize=(6, 5))

    # Plot the histogram with error bars
    plt.errorbar(
        bin_centers, values,
        yerr=[down_errors, up_errors],
        xerr=bin_widths / 2,
        fmt='.',               # Marker style
        color='black',
        ecolor='black',        # Error bar color
        elinewidth=0.5,
        capsize=0,
        label='PU weights'
    )

    # Axis labels and style
    plt.xlabel("nPU")
    plt.ylabel("weight")
    plt.yscale("log")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.legend()

    # Save plot
    plt.savefig(output_name, dpi=300)
    plt.close()

    print(f"Saved histogram to {output_name}")

with open("puWeights.json",'r') as f:
    data = json.load(f)
bin_edges = data['corrections'][0]['data']['content'][0]['value']['edges']
weights = data['corrections'][0]['data']['content'][0]['value']['content']
weights_up = data['corrections'][0]['data']['content'][1]['value']['content']
weights_down = data['corrections'][0]['data']['content'][1]['value']['content']
plot_hist_with_errors(bin_edges, weights, weights_up, weights_down, "hist_nPU_2022postEE.png")