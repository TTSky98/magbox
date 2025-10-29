# ==========  1. 生成数据  ==========
import numpy as np
from pathlib import Path
from magbox import llg3, spin3, Lattice, Vars
import matplotlib.pyplot as plt
from torch import sin, cos

def run_thermal(dt: float = 0.1,
                alpha: float =0.1,
                Temp: float= 0.1,
                gamma: float =1,
                T: float = 100,
                run_id: int =1,
                out_dir: Path  = Path("."),
                device: str = "cpu",
                dtype: str = 'f32',
                seed = None,
                spin_num: int = 2**8):
    """
    运行一次 LLG 热模拟，把结果保存到 <out_dir>/thermal_test_<run_id>.npz
    文件中额外存下所有输入参数，方便后期读取。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(seed)
    N1 = spin_num
    lt = Lattice(type="square", size=[N1], periodic=True)
    vars = Vars(J=0)

    theta0 = rng.random(N1) * 0.1
    phi0 = rng.random(N1) * 2 * np.pi

    x0=np.sin(theta0)*np.cos(phi0)
    y0=np.sin(theta0)*np.sin(phi0)
    z0=np.cos(theta0)

    sp = spin3(x0,y0,z0, lattice_type=lt, device=device, dtype=dtype)
    sf = llg3(sp, vars=vars, dt=dt, alpha=alpha, T = T, Temp=Temp, gamma=gamma, rtol=1e-2)  # type: ignore

    t, S, stats, erro_info = sf.run(sp)
    z=S[2::3,:].detach().cpu().numpy()

    en=1/2*(1-z**2)

    t_np = t.cpu().detach().numpy()

    save_file = out_dir / f"thermal_test_{run_id}.npz"
    np.savez(save_file,
             t_np=t_np,
             en=en,
             alpha=alpha,
             Temp=Temp,
             dt=dt,
             gamma=gamma,
             T=T,)
    print(f"[run_thermal] 数据已保存 -> {save_file}")
    return save_file


# ==========  2. 读取并画图  ==========
def plot_thermal(npz_file:  Path,
                 start_time: float = 20,
                 bins: int = 100,
                 fit_counts=None,
                 fig_dir: Path = Path(".")):
    """
    读取 run_thermal 生成的 npz 文件，
    画能量时序图（并标出 start_time 位置）以及能量直方图 + 线性拟合。
    图片保存为与 npz 同名（扩展名换成 .png）。
    """
    npz_file = Path(npz_file)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(exist_ok=True)

    data = np.load(npz_file)
    t_np = data['t_np']
    en = data['en']
    alpha = data['alpha']
    Temp = data['Temp']
    dt = data['dt']
    gamma = data['gamma']

    # 计算截取起点对应的索引
    idx = min(len(t_np) -2*bins, int(start_time / (t_np[1] - t_np[0])))

    # 直方图
    en_mean=np.mean(en,0)
    en_st=en[:,idx:]
    en_st=en_st.reshape(-1)
    hist, bin_edges = np.histogram(en_st, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # 态密度
    dens=np.sqrt(2*bin_centers*(1-2*bin_centers))
    # loghist=np.log(hist)
    loghist=np.log((hist+1)/dens)

    # 拟合
    if fit_counts is None:
        k, b = np.polyfit(bin_centers, loghist, deg=1)
        x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 200)
    else:
        # fit_counts = min(fit_counts, len(bin_centers))
        k,b = np.polyfit(bin_centers[:fit_counts], loghist[:fit_counts], deg=1)
        x_smooth = np.linspace(bin_centers[:fit_counts].min(), bin_centers[:fit_counts].max(), 200)
    y_smooth = k * x_smooth + b

    # 画图
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))

    # 图 1：能量时序
    axs[0].plot(t_np, en_mean, label="energy")
    axs[0].axvline(t_np[idx], color='red', linestyle='--', label=f'start={start_time}')
    axs[0].set_xlabel("Time")
    axs[0].set_title("energy")
    axs[0].legend()

    # 图 2：散点图 + 拟合
    axs[1].scatter(bin_centers, loghist,
                   s=60, facecolors='none', edgecolors='dodgerblue', lw=1.2,
                   alpha=0.85, label='Energy counts')
    axs[1].plot(x_smooth, y_smooth, color='red',
                label=f'Fit y={k:.2f}x+{b:.2f}')
    axs[1].set_xlabel("Energy")
    axs[1].set_ylabel("Counts")
    axs[1].set_title(f"Energy Histogram "
                     f"(file={npz_file.stem}, α={alpha}, 1/T={1/Temp:.2e}, "
                     f"dt={dt}, γ={gamma}), total number={np.sum(hist):.1e}")
    axs[1].legend()

    fig.tight_layout()

    # 保存
    fig_file = fig_dir / (npz_file.stem + ".png")
    fig.savefig(fig_file, dpi=300)
    print(f"[plot_thermal] 图片已保存 -> {fig_file}")
    plt.close(fig)
    return fig_file


# ==========  3. 示例用法  ==========
if __name__ == "__main__":
    # 生成一组数据
    npz = run_thermal(dt=0.1, alpha=0.2, Temp=0.5,
                      gamma=0.17, T=100, run_id=1)

    # 读取并画图
    plot_thermal(npz, start_time=30, bins=60)