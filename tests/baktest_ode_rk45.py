import torch
import warnings
import matplotlib.pyplot as plt
import numpy as np
from magbox.boxlib import ode_rk45
# from magbox.boxlib import ode_rk45 
def sin_ode(t, x):
    """
    定义微分方程: dx/dt = sin(x)
    """
    # return torch.sin(x)
    # return torch.sin(t)
    # return x**2
    return x
def out_fun(t,x0):
    
    # return 1/2*(t**2)+x0
    # return 1+x0-np.cos(t)
    # return -x0/(t*x0-1)
    return np.exp(t)*x0
   

def test_sin_ode():
    """
    测试求解 dx/dt = sin(x)
    """
    print("测试求解微分方程: dx/dt = sin(x)")
    
    # 设置初始条件和时间范围
    # tspan = torch.tensor([0,10])
    tspan=torch.linspace(0,50,1001,dtype=torch.float64)
    # x0 = torch.tensor([0.05])  # 初始条件 x(0) 
    x0=torch.tensor([1e15],dtype=torch.float64)
    
    # 设置求解器选项
    fxoptions = {
        'rel_tol': 1e-4,
        'abs_tol': 1e-6,
        'MaxStep': 5,
        'waitbar': True
    }
    
    print(f"初始条件: x(0) = {x0.item()}")
    print(f"时间范围: {tspan[0].item()} 到 {tspan[-1].item()}")
    
    # 求解微分方程
    t, x, stats, errInfo = ode_rk45(sin_ode, tspan, x0, fxoptions)
    
    print(f"\n求解完成!")
    print(f"最终解: x({t[-1]:.2f}) = {x[0, -1]:.6f}")
    print(f"函数调用次数: {stats['n_calls']}")
    print(f"求解段: {stats['n_steps']}")
    print(f"最大步长误差: {errInfo['max_step_error']:.2e}")
    
    return t, x, stats, errInfo

def plot_results(t, x,errInfo):
    """
    绘制结果
    """
    # 转换为numpy用于绘图
    t_np = t.detach().numpy()
    x_np = x.detach().numpy()

    x_theo = out_fun(t_np, x_np[:,0])
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制数值解
    plt.subplot(2, 2, 1)
    plt.plot(t_np, x_np[0], 'b-', linewidth=2, label='numeric')
    plt.plot(t_np,x_theo,'r--',label="theory")
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.title('solution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 绘制差距图 (x vs dx/dt)
    plt.subplot(2, 2, 2)
    plt.plot(t_np, np.abs(x_np/x_theo-1)[0], 'r-', linewidth=1)
    plt.xlabel('t')
    plt.ylabel('diff')
    plt.title('numeric-theory')
    plt.grid(True, alpha=0.3)
    
    # 绘制向量场
    plt.subplot(2, 2, 3)
    x_range = np.linspace(-1, 8, 20)
    t_range = np.linspace(0, 10, 10)
    T, X = np.meshgrid(t_range, x_range)
    
    # 计算向量场
    dX = np.sin(X)
    dT = np.ones_like(T)
    
    # 归一化箭头长度
    magnitude = np.sqrt(dT**2 + dX**2)
    dT_norm = dT / magnitude
    dX_norm = dX / magnitude
    
    plt.quiver(T, X, dT_norm, dX_norm, color='gray', alpha=0.6)
    plt.plot(t_np, x_np[0], 'b-', linewidth=2, label='solutio orbit')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('vector field and solution orbit')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 绘制误差分析（如果有）
    plt.subplot(2, 2, 4)
    # 计算数值导数
    
    error = errInfo.get('err_history')
    plt.semilogy(np.linspace(1,len(error),len(error)), error, 'g-', alpha=0.7)
    plt.xlabel('step')
    plt.ylabel('erro')
    plt.title('erro')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_different_initial_conditions():
    """
    比较不同初始条件的解
    """
    print("\n比较不同初始条件的解...")
    
    tspan = torch.tensor([0.0, 10.0])
    initial_conditions = [0.05, 0.11, 2.0, 3.0]
    
    fxoptions = {
        'rel_tol': 1e-3, 
        'abs_tol': 1e-6,
        'waitbar': True
    }
    
    plt.figure(figsize=(10, 6))
    
    for i, x0_val in enumerate(initial_conditions):
        x0 = torch.tensor([x0_val])
        t, x, stats, errInfo = ode_rk45(sin_ode, tspan, x0, fxoptions)
        
        t_np = t.detach().numpy()
        x_np = x.detach().numpy()
        
        plt.plot(t_np, x_np[0], linewidth=2, 
                label=f'x(0) = {x0_val}')
        
        print(f"初始条件 x(0) = {x0_val:.1f}: 最终值 x(10) = {x_np[0, -1]:.4f}")
    
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.title('solution with different init')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    # 运行测试
    warnings.filterwarnings("ignore", message="libpng")
    print("=" * 50)
    print("ode_rk45 测试程序: dx/dt = sin(x)")
    print("=" * 50)
    
    # 基本测试
    t, x, stats, errInfo = test_sin_ode()
    
    # 绘制结果
    plot_results(t, x, errInfo)
    
    # 比较不同初始条件
    compare_different_initial_conditions()
    
    print("\n所有测试完成!")