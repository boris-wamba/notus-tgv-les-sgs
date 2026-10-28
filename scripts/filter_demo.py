"""Compare trapezoid, Simpson and the Notus [1, c, 1] filters on a noisy 2D field (replaces test_filter.py)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tgvtools.filters import filter_field, highpass, transfer_function

rng = np.random.default_rng(0)
n = 256
x = np.linspace(0, 4 * np.pi, n, endpoint=False)
X, Y = np.meshgrid(x, x)
u = np.sin(2 * X + 3 * Y) + 0.1 * rng.standard_normal((n, n))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
theta = np.linspace(0, np.pi, 200)
for kind, w, ls in (("trapezoid", 2, "-"), ("simpson", 2, "--"), ("simpson", 4, "-."), ("simpson", 8, ":")):
    ax[0].plot(theta / np.pi, transfer_function(kind, w, theta), ls, label=f"{kind} {w}dx")
ax[0].set_xlabel(r"$k\Delta x/\pi$"); ax[0].set_ylabel("H"); ax[0].legend()
ax[1].plot(u[n // 2], alpha=0.5, label="noisy")
ax[1].plot(filter_field(u, "trapezoid", 2)[n // 2], label="trapezoid 2dx")
ax[1].plot(filter_field(u, "simpson", 2)[n // 2], label="simpson 2dx")
ax[1].plot(highpass(u, 4.0)[n // 2], label="Notus high-pass (c=4)")
ax[1].legend()
fig.savefig("filter_demo.png", dpi=150, bbox_inches="tight")
print("saved filter_demo.png")
