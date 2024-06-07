import time
import argparse

import cv2
import numpy as np
import matplotlib as mpl
from tqdm import tqdm

mpl.use("Agg")

import pyqubo
import seaborn as sns
import jijmodeling as jm
import matplotlib.pyplot as plt
import jijmodeling_transpiler as jmt
from dwave.cloud import Client
from dwave.system import LeapHybridSampler
from skimage.data import shepp_logan_phantom
from dwave.samplers import SimulatedAnnealingSampler

# visualization settings
rc = {
    "figure.dpi": 150,
    "figure.autolayout": False,
    "font.family": "sans-serif",
    "font.sans-serif": "Arial",
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}
sns.set_theme(context="notebook", style="white", rc=rc)


def rotate_image(image, radian):
    """image rotation by angle in radians"""
    height, width = image.shape[:2]
    deg = np.rad2deg(radian)
    M = cv2.getRotationMatrix2D((width / 2, height / 2), deg, 1.0)
    return cv2.warpAffine(
        image,
        M,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=9999,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--use_leap", action="store_true")

    args = parser.parse_args()

    label = f"CT_recon_{args.size}px_{args.bits}bits"

    client = Client.from_config()
    print("Available solvers:")
    for s in client.get_solvers():
        print(f" - {s.id:s}")
    print("")

    # load Shepp-Logan phantom
    image = shepp_logan_phantom().astype("float32")
    image = np.pad(image, ((56, 56), (56, 56)), mode="constant", constant_values=0)
    print("Data (before scaling)")
    print(f" - size: {image.shape[1]}x{image.shape[0]}")
    print(f" - range: [{image.min():.1f}, {image.max():.1f}]")
    print("")

    # shrink image size
    img_size = args.size
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
    image = (image - image.min()) / (image.max() - image.min())
    print("Data (after scaling)")
    print(f" - size: {image.shape[1]}x{image.shape[0]}")
    print(f" - range: [{image.min():.1f}, {image.max():.1f}]")
    print("")

    # quantize image
    n_bits = args.bits
    factor = (0x01 << n_bits) - 1
    if n_bits == 1:
        image = (image * 255.0).astype("uint8")
        image = cv2.threshold(image, 127, 255, cv2.THRESH_OTSU)[1]
        image = image.astype("float32") / 255.0
    else:
        image = np.floor(image * factor)
        image = np.clip(image, 0, factor) / factor

    # make image grid
    height, width = image.shape[:2]
    xs = np.arange(0, width, dtype="float32") + 0.5
    ys = np.arange(0, height, dtype="float32") + 0.5
    xs, ys = np.meshgrid(xs, ys)
    zs = np.zeros_like(xs)
    grid = np.stack([xs, ys, zs], axis=-1)
    grid = grid.astype("float32")

    # geometry
    n_proj = max(width, height)
    angles = np.linspace(0, 2.0 * np.pi, n_proj, endpoint=False)

    # compute sinogram
    sinogram = np.zeros((n_proj, width), dtype="float32")
    for n, a in enumerate(angles):
        rotated_grid = rotate_image(grid, a)
        for j in range(width):
            for i in range(height):
                xp = rotated_grid[i, j, 0]
                yp = rotated_grid[i, j, 1]
                if xp < 0 or xp >= width or yp < 0 or yp >= height:
                    continue

                xi, yi = int(xp), int(yp)
                xt, yt = xp - xi, yp - yi
                w1 = (1 - xt) * (1 - yt)
                w2 = xt * (1 - yt)
                w3 = (1 - xt) * yt
                w4 = xt * yt

                sinogram[n, j] += w1 * image[yi, xi]
                if xi + 1 < width:
                    sinogram[n, j] += w2 * image[yi, xi + 1]
                if yi + 1 < height:
                    sinogram[n, j] += w3 * image[yi + 1, xi]
                if xi + 1 < width and yi + 1 < height:
                    sinogram[n, j] += w4 * image[yi + 1, xi + 1]

    # binary variables
    vars = np.empty(shape=(height, width), dtype="object")
    for y in range(height):
        for x in range(width):
            vars[y, x] = pyqubo.UnaryEncInteger(f"x[{y}][{x}]", (0, factor))

    # construct QUBO
    time_s = time.time()
    hamiltonian = pyqubo.Num(0.0)
    pbar = tqdm(total=n_proj * width)
    pbar.set_description("QUBO construction")
    for n, a in enumerate(angles):
        rotated_grid = rotate_image(grid, a)
        for j in range(width):
            intensity = 0.0
            for i in range(height):
                xp, yp, _ = rotated_grid[i, j]
                if xp < 0 or xp >= width or yp < 0 or yp >= height:
                    continue

                xi, yi = int(xp), int(yp)
                xt, yt = xp - xi, yp - yi
                w = 1.0 / factor
                w1 = w * (1 - xt) * (1 - yt)
                w2 = w * xt * (1 - yt)
                w3 = w * (1 - xt) * yt
                w4 = w * xt * yt
                intensity += w1 * vars[yi, xi]
                if xi + 1 < width:
                    intensity += w2 * vars[yi, xi + 1]
                if yi + 1 < height:
                    intensity += w3 * vars[yi + 1, xi]
                if xi + 1 < width and yi + 1 < height:
                    intensity += w4 * vars[yi + 1, xi + 1]

            diff = sinogram[n, j] - intensity
            hamiltonian += diff * diff
            pbar.update()

    pbar.close()

    # obtain a qubo
    model = hamiltonian.compile()
    qubo, _ = model.to_qubo()
    time_e = time.time()
    print(f"Model construction: {time_e - time_s:.2f} sec")

    # perform QA
    if args.use_leap:
        print("Use D-Wave Leap")
        solver = LeapHybridSampler()
    else:
        print("Use local computer")
        solver = SimulatedAnnealingSampler()

    # compute and decode result
    time_s = time.time()
    sampleset = solver.sample_qubo(qubo, label=label)
    time_e = time.time()
    print(f"Quantum annealing: {time_e - time_s:.2f} sec")

    decoded_samples = model.decode_sampleset(sampleset)
    res = min(decoded_samples, key=lambda x: x.energy)

    # get reconstruction
    pred = np.zeros((height, width), dtype="float32")
    for y in range(height):
        for x in range(width):
            try:
                v = res.subh[f"x[{y}][{x}]"]
            except (IndexError, KeyError):
                continue

            pred[y, x] += v / factor

    # error
    err = np.abs(image - pred)
    rmse = np.sqrt(np.mean(err**2))

    # save result
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs[0, 0].imshow(image, vmin=0.0, vmax=1.0, cmap="gray")
    axs[0, 0].set_title("Original")
    axs[0, 0].set(xticks=[], yticks=[])

    axs[0, 1].imshow(sinogram, cmap="gray")
    axs[0, 1].set_title("Sinogram")
    axs[0, 1].set(xticks=[], yticks=[])

    axs[1, 0].imshow(pred, cmap="gray")
    axs[1, 0].set_title("Reconstruction")
    axs[1, 0].set(xticks=[], yticks=[])

    ims = axs[1, 1].imshow(err, vmin=0.0, vmax=1.0, cmap="viridis")
    axs[1, 1].set_title(f"RMSE={rmse:.4f}")
    axs[1, 1].set(xticks=[], yticks=[])
    fig.colorbar(ims, ax=axs[1, 1], shrink=0.9)

    fig.savefig(f"{label}.png", bbox_inches="tight")
    plt.close()

    print("File saved:", f"{label}.png")


if __name__ == "__main__":
    main()
