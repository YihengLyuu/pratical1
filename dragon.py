"""
Dragon Curve (Heighway Dragon) — PyTorch implementation
- Uses complex tensors and vectorized operations on CPU/GPU.
- Avoids `range` and `itertools` per user preference.
- n_iters produces 2**n_iters + 1 points (e.g., n_iters=16 -> 65,537 points).
"""
import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def dragon_curve_points(n_iters=16, device=None, dtype=torch.complex64):
    """
    Generate Dragon Curve polyline points as a 1D complex tensor.
    Args:
        n_iters (int): number of folding iterations (>=1 recommended).
        device: torch.device or str (e.g., 'cuda' or 'cpu').
        dtype: complex dtype, default complex64 to save memory.
    Returns:
        torch.Tensor: shape [2**n_iters + 1], complex tensor of points.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    # Start with two points along the real axis.
    z = torch.tensor([0+0j, 1+0j], dtype=dtype, device=device)

    # 90° CCW rotation constant (i).
    j = torch.tensor(1j, dtype=dtype, device=device)

    # Use while-loop (no `range`).
    i = 0
    while i < n_iters:
        # Exclude the last point, reverse the prefix, translate so last point is origin
        tail = z[:-1]
        rev = tail.flip(0)
        vecs = rev - z[-1]

        # Rotate 90° CCW and translate back
        rotated = vecs * j
        new_segment = z[-1] + rotated

        # Concatenate original polyline with the rotated copy
        z = torch.cat([z, new_segment], dim=0)

        i += 1

    return z

def plot_dragon(z, outfile=None, show=True, linewidth=0.5):
    """
    Plot the polyline given complex points z.
    - Follows the rule: matplotlib only, single plot, no specific colors.
    """
    # Bring data to CPU for plotting
    x = z.real.detach().cpu().numpy()
    y = z.imag.detach().cpu().numpy()

    plt.figure(figsize=(12, 8))
    plt.axis('equal')
    plt.axis('off')
    plt.plot(x, y, linewidth=linewidth)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', pad_inches=0, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

if __name__ == '__main__':
    # Default example rendering
    pts = dragon_curve_points(n_iters=16)  # ~65k points, fast on CPU/GPU
    plot_dragon(pts, outfile='dragon_curve.png', show=True)
    print('Saved: dragon_curve.png')