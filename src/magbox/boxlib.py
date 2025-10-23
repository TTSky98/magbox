import torch
from typing import Callable, Tuple, Dict, Any, Optional
from tqdm import tqdm
import time
import warnings
import math

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
def get_Jmtx(lattice_type,device=torch.device("cuda"),data_type=torch.float32) -> torch.Tensor:
    l_type=lattice_type.get("type")

    if l_type=="square":
        N=lattice_type.get("size")
        N_dim=len(N)
        if N_dim==1:
            N=N+[1,1]
        elif N_dim==2:
            N=N+[1]
            
        pd=lattice_type.get("periodic",True)
        if is_bool_or_single_bool_list(pd):
            pd=create_bool_list(pd, N)
            if N_dim==1:
                pd[1]=False
                pd[2]=False
            elif N_dim==2:
                pd[2]=False
        direction=lattice_type.get("J_direction",None)
        totalN=math.prod(N)
        N1=N[0]
        N2=N[1]
        N3=N[2]
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
def create_bool_list(x, y):
    """创建与y同长的布尔列表"""
    # 获取实际的布尔值
    if isinstance(x, bool):
        bool_val = x
    elif isinstance(x, list) and len(x) == 1 and isinstance(x[0], bool):
        bool_val = x[0]
    else:
        raise ValueError("x必须是布尔值或单元素布尔值列表")
    
    return [bool_val] * len(y)


def ode_rk45(odeFcn: Callable, tspan: torch.Tensor, y0: torch.Tensor, 
             options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict, Dict]:
    """
    Modified ode45 
    
    Parameters:
    -----------
    odeFcn : callable
        ODE function: f(t, y) or f(t, y, dh, rd) for thermal mode
    tspan : torch.Tensor
        Time span [t0, t1, ..., tfinal]
    y0 : torch.Tensor
        Initial conditions
    options : dict
        Options dictionary with keys:
        - rel_tol = 1e-3 ：relative tolerence
        - abs_tol = 1e-6 ：absolute tolerence
        - waitbar = True : whether to show progress
        - NormControl = 'off' : whether to use norm control
        - max_consecutive_failures = 10: Maximum number of consecutive step failures
        
    Returns:
    --------
    T : torch.Tensor
        Time points
    Y : torch.Tensor
        Solution values
    stats : dict
        Statistics (nfevals)
    errInfo : dict
        Error history and max step error
    """
    
    if options is None:
        options = {}
    
    # Initialize options
    waitbar = options.get('waitbar', True)
    
    # Extract odeset options
    rtol = options.get('rel_tol', 1e-3)
    atol = options.get('abs_tol', 1e-6)
    normcontrol = options.get('NormControl', 'off') == 'on'
    max_consecutive_failures = options.get('max_consecutive_failures', 10)
     # Initialize waitbar
    if waitbar:
        # Estimate total progress based on time span
        t0 = tspan[0].item()
        tfinal = tspan[-1].item()
        total_progress = tfinal - t0
        pbar = tqdm(total=total_progress, desc='ODE Integration', 
                   unit='time', ncols=100, bar_format='{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}]')
        last_update_time = time.time()
        update_interval = 0.1  # Update progress bar every 0.1 seconds
    
    # Initialize solution storage
    t0 = tspan[0]
    tfinal = tspan[-1]
    tdir = torch.sign(tfinal - t0)
    
    # Ensure y0 is 1D tensor
    original_shape = y0.shape
    y0 = y0.reshape(-1,1)
    neq = y0.shape[0]
    
    # Data type
    dtype = y0.dtype
    device = y0.device
    
    # Step size constraints
    hmin = 16 * torch.finfo(dtype).eps
    hmin=torch.tensor(hmin,dtype=dtype,device=device)
    safehmax = 16.0 * torch.finfo(dtype).eps * torch.max(torch.abs(t0), torch.abs(tfinal))
    defaulthmax = torch.max(0.1 * torch.abs(tfinal - t0), safehmax)
    hmax = torch.min(torch.abs(tfinal - t0), 
                    torch.tensor(options.get('MaxStep', defaulthmax.item()), dtype=dtype, device=device))
    threshold = torch.tensor(atol, dtype=dtype, device=device)
    if normcontrol:
        normy = torch.norm(y0)
    else:
        normy = torch.tensor(0.0, dtype=dtype, device=device)
    
    t = t0.clone()
    y = y0.clone()
    
    # Output configuration
    ntspan = tspan.shape[0]
    refine = options.get('Refine', 4)
    
    if ntspan > 2:
        outputAt = 1  # output only at tspan points
    elif refine <= 1:
        outputAt = 2  # computed points, no refinement
    else:
        outputAt = 3  # computed points, with refinement
        S = torch.linspace(1/refine, 1 - 1/refine, refine - 1, dtype=dtype, device=device)
    
    # Initialize output arrays
    if ntspan > 2:
        tout = torch.zeros(ntspan, dtype=dtype, device=device)
        yout = torch.zeros(neq, ntspan, dtype=dtype, device=device)
    else:
        chunk = min(max(100, 50 * refine), refine + (2**13) // neq)
        tout = torch.zeros(chunk, dtype=dtype, device=device)
        yout = torch.zeros(neq, chunk, dtype=dtype, device=device)
    
    nout = 0
    tout[nout] = t
    yout[:, nout] = y.view(-1)
    
    errHistory = []
    nfevals = 0
    nsteps = 0
    
    # Dormand-Prince coefficients
    a2 = torch.tensor(1/5, dtype=dtype, device=device)
    a3 = torch.tensor(3/10, dtype=dtype, device=device)
    a4 = torch.tensor(4/5, dtype=dtype, device=device)
    a5 = torch.tensor(8/9, dtype=dtype, device=device)
    
    b11 = torch.tensor(1/5, dtype=dtype, device=device)
    b21 = torch.tensor(3/40, dtype=dtype, device=device)
    b31 = torch.tensor(44/45, dtype=dtype, device=device)
    b41 = torch.tensor(19372/6561, dtype=dtype, device=device)
    b51 = torch.tensor(9017/3168, dtype=dtype, device=device)
    b61 = torch.tensor(35/384, dtype=dtype, device=device)
    b22 = torch.tensor(9/40, dtype=dtype, device=device)
    b32 = torch.tensor(-56/15, dtype=dtype, device=device)
    b42 = torch.tensor(-25360/2187, dtype=dtype, device=device)
    b52 = torch.tensor(-355/33, dtype=dtype, device=device)
    b33 = torch.tensor(32/9, dtype=dtype, device=device)
    b43 = torch.tensor(64448/6561, dtype=dtype, device=device)
    b53 = torch.tensor(46732/5247, dtype=dtype, device=device)
    b63 = torch.tensor(500/1113, dtype=dtype, device=device)
    b44 = torch.tensor(-212/729, dtype=dtype, device=device)
    b54 = torch.tensor(49/176, dtype=dtype, device=device)
    b64 = torch.tensor(125/192, dtype=dtype, device=device)
    b55 = torch.tensor(-5103/18656, dtype=dtype, device=device)
    b65 = torch.tensor(-2187/6784, dtype=dtype, device=device)
    b66 = torch.tensor(11/84, dtype=dtype, device=device)
    
    e1 = torch.tensor(71/57600, dtype=dtype, device=device)
    e3 = torch.tensor(-71/16695, dtype=dtype, device=device)
    e4 = torch.tensor(71/1920, dtype=dtype, device=device)
    e5 = torch.tensor(-17253/339200, dtype=dtype, device=device)
    e6 = torch.tensor(22/525, dtype=dtype, device=device)
    e7 = torch.tensor(-1/40, dtype=dtype, device=device)

    # Pi value
    t2pi=torch.tensor(2*math.pi,dtype=dtype,device=device)
    
    # Initial function evaluation
    f1 = odeFcn(t, y)
    nfevals += 1
    
    # Initial step size
    h = torch.min(hmax, torch.max(hmin, 0.1 * torch.abs(tfinal - t0)))
    absh = torch.abs(h)
    
    done = False
    next_idx = 1  # for tspan output
    
    # Main integration loop
    consecutive_failures = 0
    integration_failed= False
    while not done:
        # Step size control
        absh = torch.min(hmax, torch.max(hmin, absh))
        h = tdir * absh
        if 1.1 * absh >= torch.abs(tfinal - t):
            h = tfinal - t
            absh = torch.abs(h)
            done = True
        
        nofailed = True
        while True:
            # RK stages
            y2 = y + h * (b11 * f1)
            t2 = t + h * a2
            f2 = odeFcn(t2, y2)
            
            y3 = y + h * (b21 * f1 + b22 * f2)
            t3 = t + h * a3
            f3 = odeFcn(t3, y3)
            
            y4 = y + h * (b31 * f1 + b32 * f2 + b33 * f3)
            t4 = t + h * a4
            f4 = odeFcn(t4, y4)
            
            y5 = y + h * (b41 * f1 + b42 * f2 + b43 * f3 + b44 * f4)
            t5 = t + h * a5
            f5 = odeFcn(t5, y5)
            
            y6 = y + h * (b51 * f1 + b52 * f2 + b53 * f3 + b54 * f4 + b55 * f5)
            t6 = t + h
            f6 = odeFcn(t6, y6)
            
            tnew = t + h
            if done:
                tnew = tfinal
            # h = tnew - t
            
            ynew = y + h * (b61 * f1 + b63 * f3 + b64 * f4 + b65 * f5 + b66 * f6)
            f7 = odeFcn(tnew, ynew)
            
            nfevals += 6
            
            # Error estimation
            fE = f1 * e1 + f3 * e3 + f4 * e4 + f5 * e5 + f6 * e6 + f7 * e7
            
            if normcontrol:
                normynew = torch.norm(ynew)
                scalingFactor = torch.max(torch.max(normy, normynew), threshold)
                err = absh * torch.norm(fE) / scalingFactor
            else:
                scalingFactor = torch.max(torch.max(torch.abs(y), torch.abs(ynew)), threshold)
                err = fE / scalingFactor
                err = absh * torch.norm(err, p=float('inf'))
            
            err = err.item()  # Convert to scalar for comparison
            
            # Step acceptance
            if err > rtol:
                if torch.abs(absh - hmin) <  0.2 * hmin:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        if waitbar:
                            pbar.close()
                        warnings.warn(
                            f"Step size reached minimum hmin = {hmin.item():.2e} at t={t.item():.2e}, but still cannot satisfy tolerance. "
                            f"Current error: {err:.2e}, Required tolerance: {rtol:.2e}. "
                            f"This may indicate a stiff ODE or overly strict tolerances. "
                            f"Consider using a stiff solver or relaxing tolerances.",
                            RuntimeWarning,
                            stacklevel=2
                        )
                        done = True
                        integration_failed = True
                        break
                else:
                    consecutive_failures = 0  # Reset if we're still above hmin
                # Adaptive mode: shrink step and retry
                if nofailed:
                    nofailed = False
                    absh = torch.max(hmin, absh * max(0.1, 0.8 * (rtol / err) ** (1/5)))
                else:
                    absh = torch.max(hmin, 0.5 * absh)
                h = tdir * absh
                done = False
            else:
                # Accept step
                errHistory.append(err)
                consecutive_failures = 0
                break
        
        nsteps += 1
        if integration_failed:
            break

        # Update waitbar if enabled
        if waitbar:
            current_time = time.time()
            if current_time - last_update_time >= update_interval or done:
                progress = tnew.item() - t0.item()
                pbar.n = min(progress, total_progress)
                pbar.refresh()
                last_update_time = current_time
        
        # Output processing
        if outputAt == 2:  # computed points, no refinement
            nout_new = 1
            tout_new = tnew.unsqueeze(0)
            yout_new = ynew.unsqueeze(1)
        elif outputAt == 3:  # computed points, with refinement
            tref = t + (tnew - t) * S
            nout_new = refine
            tout_new = torch.cat([tref, tnew.unsqueeze(0)])
            yntrp45 = ntrp45split(tref, t, y, h, f1, f3, f4, f5, f6, f7)
            yout_new = torch.cat([yntrp45, ynew.unsqueeze(1)], dim=1) 
        else:  # output only at tspan points
            nout_new = 0
            tout_new = torch.tensor([], dtype=dtype, device=device)
            yout_new = torch.tensor([], dtype=dtype, device=device)
            
            while next_idx < ntspan:
                if tdir * (tnew - tspan[next_idx]) < 0:
                    break
                nout_new += 1
                tout_new = torch.cat([tout_new, tspan[next_idx].unsqueeze(0)])
                if tspan[next_idx] == tnew:
                    yout_new = torch.cat([yout_new, ynew], dim=1)
                else:
                    yntrp45 = ntrp45split(tspan[next_idx].unsqueeze(0), t, y, h, f1, f3, f4, f5, f6, f7)
                    yout_new = torch.cat([yout_new, yntrp45], dim=1)
                next_idx += 1
        yout_new = yout_new % t2pi
        # Store output
        if nout_new > 0:
            oldnout = nout
            nout = nout + nout_new
            
            # Expand arrays if needed
            if nout+1 > tout.shape[0]:
                extra = max(chunk, nout_new)
                tout_new_temp = torch.zeros(tout.shape[0] + extra, dtype=dtype, device=device)
                tout_new_temp[:tout.shape[0]] = tout
                tout = tout_new_temp
                
                yout_new_temp = torch.zeros(neq, yout.shape[1] + extra, dtype=dtype, device=device)
                yout_new_temp[:, :yout.shape[1]] = yout
                yout = yout_new_temp
            
            tout[oldnout+1:nout+1] = tout_new
            yout[:, oldnout+1:nout+1] = yout_new
        
        # Step size adjustment for adaptive mode
        if  nofailed:
            temp = 1.25 * (err / rtol) ** (1/5)
            if temp > 0.2:
                absh = absh / temp
            else:
                absh = 5.0 * absh
        
        # Advance integration
        t = tnew
        y = ynew % t2pi
        if normcontrol:
            normy = normynew
        f1 = f7  # Reuse last function evaluation

    # Close waitbar
    if waitbar:
        pbar.n = total_progress
        pbar.refresh()
        pbar.close()

    # Prepare outputs
    tout = tout[:nout+1]
    yout = yout[:, :nout+1]
    
    stats = {'n_fevals': nfevals,
             'n_steps': nsteps,
             'n_output': nout+1,
             'intergration': not integration_failed}
    errInfo = {
        'err_history': errHistory,
        'max_step_error': max(errHistory) if errHistory else 0.0
    }
    
    return tout, yout, stats, errInfo


def ntrp45split(tinterp: torch.Tensor, t: torch.Tensor, y: torch.Tensor, 
                h: torch.Tensor, f1: torch.Tensor, f3: torch.Tensor, 
                f4: torch.Tensor, f5: torch.Tensor, f6: torch.Tensor, 
                f7: torch.Tensor) -> torch.Tensor:
    """
    Interpolation function for Dormand-Prince method.
    """
    # Interpolation coefficients
    bi12 = -183/64;   bi13 = 37/12;     bi14 = -145/128
    bi32 = 1500/371;  bi33 = -1000/159; bi34 = 1000/371
    bi42 = -125/32;   bi43 = 125/12;    bi44 = -375/64
    bi52 = 9477/3392; bi53 = -729/106;  bi54 = 25515/6784
    bi62 = -11/7;     bi63 = 11/3;      bi64 = -55/28
    bi72 = 3/2;       bi73 = -4;        bi74 = 5/2
    
    dtype = y.dtype
    device = y.device
    
    s = (tinterp - t) / h
    s = s.reshape(-1)
    
    yinterp = torch.zeros(y.shape[0], s.shape[0], dtype=dtype, device=device)
    
    for jj in range(s.shape[0]):
        sj = s[jj]
        sj2 = sj * sj
        
        bs1 = sj + sj2 * (bi12 + sj * (bi13 + bi14 * sj))
        bs3 = sj2 * (bi32 + sj * (bi33 + bi34 * sj))
        bs4 = sj2 * (bi42 + sj * (bi43 + bi44 * sj))
        bs5 = sj2 * (bi52 + sj * (bi53 + bi54 * sj))
        bs6 = sj2 * (bi62 + sj * (bi63 + bi64 * sj))
        bs7 = sj2 * (bi72 + sj * (bi73 + bi74 * sj))
        
        yinterp[:, jj:jj+1] = y + h * (f1 * bs1 + f3 * bs3 + f4 * bs4 + 
                                 f5 * bs5 + f6 * bs6 + f7 * bs7)
    
    return yinterp

def ode_sde_em(f: Callable, # function
                tspan: torch.Tensor, 
                y0: torch.Tensor, 
                options: Optional[Dict[str, Any]] = None
        ) -> Tuple[torch.Tensor, torch.Tensor, Dict, Dict]:
    """
    Modified Euler-Maruyama method for SDEs.
    
    Parameters:
    -----------
    f : callable
        function: f0,g0 = f(t, y) with f0 the drift term and g0 the diffusion term
    tspan : torch.Tensor
        Time span for integration
    y0 : torch.Tensor
        Initial condition
    options : dict, optional
        Integration options

    Returns:
    --------
    tout : torch.Tensor
        Time points at which output is given
    yout : torch.Tensor 
        Solution at tout
    stats : dict
        Integration statistics
    errInfo : dict
        Error information
    """
    if options is None:
        options = {}

    # Initialize options
    waitbar = options.get('waitbar', True)
    
    # Extract odeset options
    rtol = options.get('rel_tol', 1e-3)
    atol = options.get('abs_tol', 1e-6)
    normcontrol = options.get('NormControl', 'off') == 'on'
    max_consecutive_failures = options.get('max_consecutive_failures', 10)
    # Initialize waitbar
    if waitbar:
        # Estimate total progress based on time span
        t0 = tspan[0].item()
        tfinal = tspan[-1].item()
        total_progress = tfinal - t0
        pbar = tqdm(total=total_progress, desc='ODE Integration', 
                unit='time', ncols=100, bar_format='{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}]')
        last_update_time = time.time()
        update_interval = 0.1  # Update progress bar every 0.1 seconds
    
    # Initialize solution storage
    t0 = tspan[0]
    tfinal = tspan[-1]
    tdir = torch.sign(tfinal - t0)
    
    # Ensure y0 is 1D tensor
    original_shape = y0.shape
    y0 = y0.reshape(-1,1)
    neq = y0.shape[0]
    
    # Data type
    dtype = y0.dtype
    device = y0.device
    
    # Step size constraints
    hmin = 16 * torch.finfo(dtype).eps
    hmin=torch.tensor(hmin,dtype=dtype,device=device)
    safehmax = 16.0 * torch.finfo(dtype).eps * torch.max(torch.abs(t0), torch.abs(tfinal))
    defaulthmax = torch.max(0.1 * torch.abs(tfinal - t0), safehmax)
    hmax = torch.min(torch.abs(tfinal - t0), 
                    torch.tensor(options.get('MaxStep', defaulthmax.item()), dtype=dtype, device=device))
    threshold = torch.tensor(atol, dtype=dtype, device=device)
    if normcontrol:
        normy = torch.norm(y0)
    else:
        normy = torch.tensor(0.0, dtype=dtype, device=device)
    
    t = t0.clone()
    y = y0.clone()
    
    # Output configuration
    ntspan = tspan.shape[0]
    refine = options.get('Refine', 4)
    
    if ntspan > 2:
        outputAt = 1  # output only at tspan points
    elif refine <= 1:
        outputAt = 2  # computed points, no refinement
    else:
        outputAt = 3  # computed points, with refinement
        S = torch.linspace(1/refine, 1 - 1/refine, refine - 1, dtype=dtype, device=device)
    
    # Initialize output arrays
    if ntspan > 2:
        tout = torch.zeros(ntspan, dtype=dtype, device=device)
        yout = torch.zeros(neq, ntspan, dtype=dtype, device=device)
    else:
        chunk = min(max(100, 50 * refine), refine + (2**13) // neq)
        tout = torch.zeros(chunk, dtype=dtype, device=device)
        yout = torch.zeros(neq, chunk, dtype=dtype, device=device)
    
    nout = 0
    tout[nout] = t
    yout[:, nout] = y.view(-1)
    
    errHistory = []
    nfevals = 0
    nsteps = 0

    # Pi value
    t2pi=torch.tensor(2*math.pi,dtype=dtype,device=device)
    # Initial step size
    h = torch.min(hmax, torch.max(hmin, 0.1 * torch.abs(tfinal - t0)))
    absh = torch.abs(h)
    # Initial function evaluation
    f1, g1 = f(t, y)
    noise_dim=g1.shape[1]
    nfevals += 1
    
 
    
    
    done = False
    next_idx = 1  # for tspan output
    
    # Main integration loop
    consecutive_failures = 0
    integration_failed= False

    while not done:
        absh = torch.min(hmax, torch.max(hmin, absh))
        h = tdir * absh
        if 1.1 * absh >= torch.abs(tfinal - t):
            h = tfinal - t
            absh = torch.abs(h)
            done = True
        
        nofailed = True
        W1=torch.randn(noise_dim,1,dtype=dtype,device=device)
        W2=torch.randn(noise_dim,1,dtype=dtype,device=device)
        W=W1+W2
        while True:
            y2 = y+ f1 * h/2 + g1 @ W1 * torch.sqrt(absh/2)
            t2 = t + h / 2
            f2, g2 = f(t2, y2)
            
            ynew = y + f2 *h/2 + g2 @ W2 * torch.sqrt(absh/2)
            tnew = t + h

            y_full = y + f1 * h + g1 @ W * torch.sqrt(absh)
            
            nfevals += 1
            
            fE= ynew - y_full
            if normcontrol:
                norm_y_new = torch.norm(ynew)
                scalingFactor = torch.max(torch.max(normy, norm_y_new), threshold)
                err = absh *torch.norm(fE) / scalingFactor
            else: 
                scalingFactor = torch.max(torch.max(torch.abs(y), torch.abs(ynew)), threshold)
                err = fE / scalingFactor
                err = absh * torch.norm(err, p=float('inf'))

            err=err.item()
            # Step acceptance
            if err > rtol:
                if torch.abs(absh - hmin) <  0.2 * hmin:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        if waitbar:
                            pbar.close()
                        warnings.warn(
                            f"Step size reached minimum hmin = {hmin.item():.2e} at t={t.item():.2e}, but still cannot satisfy tolerance. "
                            f"Current error: {err:.2e}, Required tolerance: {rtol:.2e}. "
                            f"This may indicate a stiff ODE or overly strict tolerances. "
                            f"Consider using a stiff solver or relaxing tolerances.",
                            RuntimeWarning,
                            stacklevel=2
                        )
                        done = True
                        integration_failed = True
                        break
                else:
                    consecutive_failures = 0  # Reset if we're still above hmin
                # Adaptive mode: shrink step and retry
                if nofailed:
                    nofailed = False
                    absh = torch.max(hmin, absh * max(0.1, 0.8 * (rtol / err) ** (1/1.5)))
                else:
                    absh = torch.max(hmin, 0.5 * absh)
                h = tdir * absh
                done = False
            else:
                # Accept step
                errHistory.append(err)
                consecutive_failures = 0
                break
        nsteps += 1
        if integration_failed:
            break

        # Update waitbar if enabled
        if waitbar:
            current_time = time.time()
            if current_time - last_update_time >= update_interval or done:
                progress = tnew.item() - t0.item()
                pbar.n = min(progress, total_progress)
                pbar.refresh()
                last_update_time = current_time
        
        # Output processing
        if outputAt == 2:  # computed points, no refinement
            nout_new = 1
            tout_new = tnew.unsqueeze(0)
            yout_new = ynew.unsqueeze(1)
        elif outputAt == 3:  # computed points, with refinement
            tref = t + (tnew - t) * S
            nout_new = refine
            tout_new = torch.cat([tref, tnew.unsqueeze(0)])
            y_ntrp = ntrp_em(tref, t, y, h, y2, ynew)
            yout_new = torch.cat([y_ntrp, ynew.unsqueeze(1)], dim=1) 
        else:  # output only at tspan points
            nout_new = 0
            tout_new = torch.tensor([], dtype=dtype, device=device)
            yout_new = torch.tensor([], dtype=dtype, device=device)
            
            while next_idx < ntspan:
                if tdir * (tnew - tspan[next_idx]) < 0:
                    break
                nout_new += 1
                tout_new = torch.cat([tout_new, tspan[next_idx].unsqueeze(0)])
                if tspan[next_idx] == tnew:
                    yout_new = torch.cat([yout_new, ynew], dim=1)
                else:
                    y_ntrp = ntrp_em(tspan[next_idx].unsqueeze(0), t, y, h, y2, ynew)
                    yout_new = torch.cat([yout_new, y_ntrp], dim=1)
                next_idx += 1
        yout_new = yout_new % t2pi
        # Store output
        if nout_new > 0:
            oldnout = nout
            nout = nout + nout_new
            
            # Expand arrays if needed
            if nout+1 > tout.shape[0]:
                extra = max(chunk, nout_new)
                tout_new_temp = torch.zeros(tout.shape[0] + extra, dtype=dtype, device=device)
                tout_new_temp[:tout.shape[0]] = tout
                tout = tout_new_temp
                
                yout_new_temp = torch.zeros(neq, yout.shape[1] + extra, dtype=dtype, device=device)
                yout_new_temp[:, :yout.shape[1]] = yout
                yout = yout_new_temp
            
            tout[oldnout+1:nout+1] = tout_new
            yout[:, oldnout+1:nout+1] = yout_new
        
        # Step size adjustment for adaptive mode
        if  nofailed:
            temp = 1.25 * (err / rtol) ** (1/1.5)
            if temp > 0.2:
                absh = absh / temp
            else:
                absh = 5.0 * absh
        
        # Advance integration
        t = tnew
        y = ynew % t2pi
        if normcontrol:
            normy = norm_y_new
        f1, g1 = f(t,y)
        nfevals += 1
    # Close waitbar
    if waitbar:
        pbar.n = total_progress
        pbar.refresh()
        pbar.close()

    # Prepare outputs
    tout = tout[:nout+1]
    yout = yout[:, :nout+1]
    
    stats = {'n_fevals': nfevals,
             'n_steps': nsteps,
             'n_output': nout+1,
             'intergration': not integration_failed}
    err_info = {
        'err_history': errHistory,
        'max_step_error': max(errHistory) if errHistory else 0.0
    }
    return tout,yout,stats,err_info

def ntrp_em(tinterp: torch.Tensor, t: torch.Tensor, y: torch.Tensor, 
            h: torch.Tensor, y_mid: torch.Tensor, y_end: torch.Tensor) -> torch.Tensor:
    """
    2nd order Interpolation for Euler-Maruyama method.
    """
    dtype = y.dtype
    device = y.device
    
    s = (tinterp - t) / h
    s = s.reshape(-1)

    l0 = (2.0*s-1.0)*(s-1.0)
    l_mid = -4.0*s*(s-1.0)
    l_end = s*(2.0*s-1.0)
    
    yinterp = l0*y + l_mid*y_mid + l_end*y_end
    
    return yinterp