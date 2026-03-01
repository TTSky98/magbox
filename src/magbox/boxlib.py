import torch
from typing import Callable, Tuple, Dict, Any, Optional
import warnings
import math
from .Wait_bar import Wait_bar
from .initial import Lattice
from .solver.sde_solver import sde_solver
from .solver.sde3_solver import sde3_solver
from .solver.sde3_fix_solver import sde3_fix_solver
from .solver.eq_solver import eq_solver
from .solver.eq3_solver import eq3_solver
from .solver.eq3_fix_solver import eq3_fix_solver

def get_data_type(type):
    if type=="f32":
        data_type=torch.float32
    elif type=="f64":
        data_type=torch.float64
    elif type=='f16':
        data_type=torch.float16
    else:
        raise ValueError("type must be f16, f32 or f64")
    return data_type
    
def get_device(device):
    if device=="gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("CUDA is not available, using CPU instead.")
            return torch.device("cpu")
    elif device=="cpu":
        return torch.device("cpu")
    else:
        raise ValueError("device must be 'cpu' or 'gpu'")
def get_Jmtx(lattice_type: Lattice,device=torch.device("cuda"),data_type=torch.float32) -> torch.Tensor:
    l_type=lattice_type.type

    if l_type=="square":
        N=lattice_type.size
        N_dim=len(N)
        if N_dim==1:
            N=N+[1,1]
        elif N_dim==2:
            N=N+[1]
            
        pd=lattice_type.periodic #type: ignore
        Force_keep_pd = lattice_type.force_periodic
        if is_bool_or_single_bool_list(pd):
            pd=create_bool_list(pd, N) 
            if N_dim==1:
                pd[1]=False
                pd[2]=False
            elif N_dim==2:
                pd[2]=False
        pd: list[bool] = pd
        direction=lattice_type.J_direction
        totalN=math.prod(N)
        N1=N[0]
        N2=N[1]
        N3=N[2]
        pd_warning_flag = [False, False, False]
        forced_dim=[]
        for i in range(3):
            if N[i] <=2 and pd[i]:
                if not Force_keep_pd:
                    pd[i] = False
                forced_dim.append(i)
                pd_warning_flag[i] = True
        if Force_keep_pd and any(pd_warning_flag):
            warnings.warn(f"Periodic boundary condition in the dimension with length equal to 1 or 2 is set to True. This may cause errors in the simulation. Forced dimension:{forced_dim}", UserWarning, stacklevel=2)
            
        if direction is None:
            # 所有方向的耦合
            v = torch.ones(3 * totalN - N1 * N2 - N2 * N3 - N3 * N1, dtype=data_type,device=device) / 2
            
            # get backward coupling, direction 1
            i = torch.arange(1, totalN)  
            back_boundary = (i % N1 == 0)
            i = i[~back_boundary]
            j = i.clone()  
            
            # get right coupling, direction 2
            itmp = torch.arange(1, totalN)
            right_boundary = ((itmp - 1) % (N1 * N2) - N1 * (N2 - 1)) >= 0
            itmp = itmp[~right_boundary]
            i = torch.cat([i, itmp])
            j = torch.cat([j, itmp + N1 - 1])  # 调整索引
            
            # get bottom coupling, direction 3
            i = torch.cat([i, torch.arange(1, totalN - N1 * N2 + 1)])
            j = torch.cat([j, torch.arange(N1 * N2 , totalN )])
            
            if pd[0]:  # periodic boundary condition in direction 1
                back_forward_i = torch.arange(1, totalN + 1, N1)
                back_forward_j = torch.arange(N1-1, totalN, N1)
                i=torch.cat([i, back_forward_i])
                j=torch.cat([j, back_forward_j])
                v= torch.cat([v, torch.ones(N2 * N3, dtype=data_type,device=device) / 2])
            if pd[1]:  # periodic boundary condition in direction 2
                left_right_i = torch.arange(1, totalN + 1)
                tmpbd = ((left_right_i - 1) % (N1 * N2)) >= N1
                left_right_i = left_right_i[~tmpbd]
                left_right_j = torch.arange(0, totalN)
                tmpbd = (left_right_j  % (N1 * N2)) < N1 * (N2 - 1)
                left_right_j = left_right_j[~tmpbd]
                i=torch.cat([i, left_right_i])
                j=torch.cat([j, left_right_j])
                v= torch.cat([v, torch.ones(N1 * N3, dtype=data_type,device=device) / 2])
            if pd[2]:  # periodic boundary condition in direction 3
                up_down_i = torch.arange(1, N1 * N2 + 1)
                up_down_j = torch.arange(N1 * N2 * (N3 - 1), totalN)
                i=torch.cat([i, up_down_i])
                j=torch.cat([j, up_down_j])
                v= torch.cat([v, torch.ones(N1 * N2, dtype=data_type,device=device) / 2])

        else:
            # 只有一个方向的耦合
            if direction==0: # backward耦合（x方向）
                i = torch.arange(1, totalN)
                back_boundary = (i % N1 == 0)
                i = i[~back_boundary]
                j = i.clone()  # 调整索引
                v = torch.ones(len(i), dtype=data_type,device=device) / 2
                if pd[0]:  # periodic boundary condition in direction 1
                    back_forward_i = torch.arange(1, totalN + 1, N1)
                    back_forward_j = torch.arange(N1-1, totalN , N1)
                    i=torch.cat([i, back_forward_i])
                    j=torch.cat([j, back_forward_j])

                v = torch.ones(len(i), dtype=data_type,device=device) / 2
            elif direction==1: # right耦合（y方向）
                i = torch.arange(1, totalN)
                right_boundary = ((i - 1) % (N1 * N2) - N1 * (N2 - 1)) >= 0
                i = i[~right_boundary]
                j = i + N1 - 1  # 调整索引
                if pd[1]:  # periodic boundary condition in direction 2
                    left_right_i = torch.arange(1, totalN + 1)
                    tmpbd = ((left_right_i - 1) % (N1 * N2)) >= N1
                    left_right_i = left_right_i[~tmpbd]
                    left_right_j = torch.arange(0, totalN )
                    tmpbd = (left_right_j % (N1 * N2)) < N1 * (N2 - 1)
                    left_right_j = left_right_j[~tmpbd]
                    i=torch.cat([i, left_right_i])
                    j=torch.cat([j, left_right_j])
                v = torch.ones(len(i), dtype=data_type,device=device) / 2
            elif direction == 2:  # bottom耦合（z方向）
                i = torch.arange(1, totalN - N1 * N2 + 1)
                j = torch.arange(N1 * N2, totalN)
                if pd:  # 周期性边界条件
                    up_down_i = torch.arange(1, N1 * N2 + 1)
                    up_down_j = torch.arange(N1 * N2 * (N3 - 1), totalN)
                    
                    i = torch.cat([i, up_down_i])
                    j = torch.cat([j, up_down_j])
                v = torch.ones(len(i), dtype=data_type,device=device) / 2
            else:
                raise ValueError('direction must be 1, 2, 3 or None')
        i = i - 1  # Convert to 0-based index
        Jmtx=torch.sparse_coo_tensor(torch.stack([i, j]), v, (totalN, totalN),dtype=data_type,device=device)
       
    return Jmtx+Jmtx.t()
def is_bool_or_single_bool_list(x):
    if isinstance(x, bool):
        return True
    elif isinstance(x, list) and len(x) == 1 and isinstance(x[0], bool):
        return True
    return False
def create_bool_list(x, y) -> list[bool]:
    """创建与y同长的布尔列表"""
    # 获取实际的布尔值
    if isinstance(x, bool):
        bool_val = x
    elif isinstance(x, list) and len(x) == 1 and isinstance(x[0], bool):
        bool_val = x[0]
    else:
        raise ValueError("x必须是布尔值或单元素布尔值列表")
    
    return [bool_val] * len(y)    

def ode_rk45(ode_fun: Callable, t_span: torch.Tensor, y0: torch.Tensor, 
             options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict, Dict]:
    """
    Modified rk45
    
    Parameters:
    -----------
    ode_fun : callable
        ODE function: f(t, y) 
    t_span : torch.Tensor
        Time span [t0, t1, ..., tfinal]
    y0 : torch.Tensor
        Initial conditions
    options : dict
        Options dictionary with keys:
        - rel_tol = 1e-3 ：relative tolerence
        - abs_tol = 1e-6 ：absolute tolerence
        - waitbar = True : whether to show progress
        - max_consecutive_failures = 10: Maximum number of consecutive step failures
    
    Returns:
    --------
    t : torch.Tensor
        Time points
    y : torch.Tensor
        Solution values
    stats : dict
        Statistics (n_calls)
    err_info : dict
        Error history and max step error
    """
    solver= 'RK45'
    sf=eq_solver(ode_fun, t_span, y0, solver, options)               
    bar=Wait_bar(t_span, sf.waitbar)  # Initialize the progress bar
    t, ang, stats, error_info =sf.run(bar)
    return t, ang, stats, error_info

def ode3_rk45(ode_fun: Callable, t_span: torch.Tensor, y0: torch.Tensor, 
             options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict, Dict]:
    """
    Modified rk45
    
    Parameters:
    -----------
    ode_fun : callable
        ODE function: f(t, y) 
    t_span : torch.Tensor
        Time span [t0, t1, ..., tfinal]
    y0 : torch.Tensor
        Initial conditions
    options : dict
        Options dictionary with keys:
        - rel_tol = 1e-3 ：relative tolerence
        - abs_tol = 1e-6 ：absolute tolerence
        - waitbar = True : whether to show progress
        - max_consecutive_failures = 10: Maximum number of consecutive step failures
    
    Returns:
    --------
    t : torch.Tensor
        Time points
    y : torch.Tensor
        Solution values
    stats : dict
        Statistics (n_calls)
    err_info : dict
        Error history and max step error
    """
    solver= 'RK45'
    fix_step = bool((options or {}).get('fix_step', False))
    if fix_step:
        sf = eq3_fix_solver(ode_fun, t_span, y0, solver, options)
    else:
        sf = eq3_solver(ode_fun, t_span, y0, solver, options)
    bar=Wait_bar(t_span, sf.waitbar)  # Initialize the progress bar
    t, ang, stats, error_info =sf.run(bar)
    return t, ang, stats, error_info

def ode_sde_em(f: Callable, t_span: torch.Tensor, y0: torch.Tensor, options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict, Dict]:
    solver = 'EM'
    sf = sde_solver(f, t_span, y0, solver, options)
    bar = Wait_bar(t_span, sf.waitbar)
    t, y, stats, err_info = sf.run(bar)
    return t, y, stats, err_info

def ode3_sde_em(f: Callable, t_span: torch.Tensor, y0: torch.Tensor, options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict, Dict]:
    solver = 'EM'
    fix_step = bool((options or {}).get('fix_step', False))
    if fix_step:
        sf = sde3_fix_solver(f, t_span, y0, solver, options)
    else:
        sf = sde3_solver(f, t_span, y0, solver, options)
    bar = Wait_bar(t_span, sf.waitbar)
    t, y, stats, err_info = sf.run(bar)
    return t, y, stats, err_info

def ode3_sde_eh(f: Callable, t_span: torch.Tensor, y0: torch.Tensor, options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict, Dict]:
    solver = 'EH'
    sf = sde3_solver(f, t_span, y0, solver, options)
    bar = Wait_bar(t_span, sf.waitbar)
    t, y, stats, err_info = sf.run(bar)
    return t, y, stats, err_info