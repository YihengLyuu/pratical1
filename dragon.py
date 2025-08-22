# Generate Dragon Curve polyline points as a 1D complex tensor.
# device: torch.device or str (e.g., 'cuda' or 'cpu').
# dtype: complex dtype, default complex64 to save memory.
import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def dragon_curve_points(n_iters=16, device=None, dtype=torch.complex64):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    # 总点数：2**n + 1
    # Make this program more simplified and efficient.
    total = (1 << n_iters) + 1

     # 预分配（避免反复 cat）
    z = torch.empty(total, dtype=dtype, device=device)

    # 初始两个点
    z[0] = torch.tensor(0 + 0j, dtype=dtype, device=device)
    z[1] = torch.tensor(1 + 0j, dtype=dtype, device=device)

    # 90° 逆时针旋转常数 i
    j = torch.tensor(1j, dtype=dtype, device=device)

    # 当前已有长度（前 len_ 的内容有效）
    len_ = 2
    i = 0
    while i < n_iters:
        # 旧段的“尾巴”（不含最后一个点），反转后平移到原点再旋转并平移回来
        pivot = z[len_ - 1]
        tail = z[:len_ - 1]
        rev = tail.flip(0)
        # 生成新段
        new_segment = pivot + (rev - pivot) * j

        # 原地写入：新段紧随其后
        z[len_: len_ + (len_ - 1)] = new_segment

        # 更新长度：len_new = len_old + (len_old - 1)
        len_ = len_ + (len_ - 1)
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