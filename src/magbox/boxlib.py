import torch
from typing import Callable, Tuple, Dict, Any, Optional
import warnings
import math
from .Wait_bar import Wait_bar
from .initial import Lattice

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
            
        pd0=lattice_type.periodic
        if is_bool_or_single_bool_list(pd0):
            pd=create_bool_list(pd0, N)
            if N_dim==1:
                pd[1]=False
                pd[2]=False
            elif N_dim==2:
                pd[2]=False
        direction=lattice_type.J_direction
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
class eq_solver:
    def __init__(self,odeFcn, t_span, y0: torch.Tensor, solver_name, options):
        self.device=y0.device
        self.dtype=y0.dtype
        self._ode_options(options)
        self._ode_initial(odeFcn, t_span, y0)
        self._tableau(solver_name)
    def _ode_options(self, options):
        device=self.device
        dtype=self.dtype
        if options is None:
            options = {}
        
        # Initialize options
        self.waitbar = options.get('waitbar', True)
        
        # Extract odeset options
        rtol = options.get('rel_tol', torch.tensor(1e-3,device=device,dtype=dtype))
        atol = options.get('abs_tol', torch.tensor(1e-6,device=device,dtype=dtype))
        max_failures = options.get('max_consecutive_failures', torch.tensor(10,device=device,dtype=torch.int64))
        refine = options.get('refine', 4)
        max_step = options.get('max_step', torch.tensor(2**32-1,device=device, dtype=torch.int64))

        self.rtol=torch.as_tensor(rtol, device=device, dtype=dtype)
        self.atol=torch.as_tensor(atol, device=device, dtype=dtype)
        self.max_failures=torch.as_tensor(max_failures, device=device, dtype=torch.int64)
        self.refine=refine
        self.max_step = torch.as_tensor(max_step, device=device, dtype=torch.int64)

    def _ode_initial(self, odeFcn, t_span, y0):
        device=self.device
        dtype=self.dtype
        self.t0 = t_span[0]
        self.t_final = t_span[-1]
        self.t_dir = torch.sign(self.t_final - self.t0)
        # step size constraints
        h_min = 16 * torch.finfo(dtype).eps
        h_min=torch.tensor(h_min, dtype=dtype,device=device)
        safe_h_max = 16.0 * torch.finfo(dtype).eps * torch.max(torch.abs(self.t0), torch.abs(self.t_final))
        default_h_max = torch.max(0.1 * torch.abs(self.t_final - self.t0), safe_h_max)
        h_max = torch.min(torch.abs(self.t_final - self.t0), self.max_step)

        t = self.t0.clone()
        y = y0.clone()
        y=y.view(-1,1)


        n_t_span=t_span.shape[0]
        n_eq=y0.shape[0]
        S=torch.tensor(0,device=device,dtype=dtype)
        chunk=0
        refine=self.refine
        if n_t_span > 2:
            output_pos = 1  # output at t_span points
        elif self.refine <= 1:
            output_pos = 2  # computed points
        else:
            output_pos = 3  # computed points, with refinement
            S = torch.linspace(1/refine, 1 - 1/refine, refine - 1, dtype=dtype, device=device)
        # Initialize output arrays
        if n_t_span > 2:
            t_out = torch.zeros(n_t_span, dtype=dtype, device=device)
            y_out = torch.zeros(n_eq, n_t_span, dtype=dtype, device=device)
        else:
            chunk = min(max(100, 50 * refine), refine + (2**13) // n_eq)
            t_out = torch.zeros(chunk, dtype=dtype, device=device)
            y_out = torch.zeros(n_eq, int(chunk), dtype=dtype, device=device)

        self.S=S
        self.chunk=chunk
        self.h_min=h_min
        self.h_max=h_max
        self.t=t
        self.y=y
        self.n_t_span=n_t_span
        self.n_eq=n_eq
        self.t_span=t_span
        self.output_pos=output_pos
        self.t_out=t_out
        self.y_out=y_out
        self.ode_fcn=odeFcn

        # return t0, t_final, t_dir, y0, h_min, h_max, t, y0, output_pos, t_out, y_out, chunk, S
    def _tableau(self, solver_name):
        device=self.device
        dtype=self.dtype
        if solver_name == 'RK45':
            self.alpha=torch.tensor([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1.], 
                                    dtype=dtype,device=device)
            self.beta=[
                torch.tensor([1 / 5], dtype=dtype,device=device).view(-1,1),
                torch.tensor([3 / 40, 9 / 40], dtype=dtype,device=device).view(-1,1),
                torch.tensor([44 / 45, -56 / 15, 32 / 9], dtype=dtype,device=device).view(-1,1),
                torch.tensor([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729], dtype=dtype,device=device).view(-1,1),
                torch.tensor([9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656], dtype=dtype,device=device).view(-1,1),
                torch.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84], dtype=dtype,device=device).view(-1,1)
            ]
            self.c_error=torch.tensor([71/57600, 0.0, -71/16695, 71/1920, -17253/339200, 22/525, -1/40]
                # [71 / 86400, 0, -142 / 50085, 71 / 2880, -5751 / 169600, 44 / 1575, -1 / 60],
                                      , dtype=dtype,device=device).view(-1,1)
            self.order=torch.tensor(5, dtype=dtype,device=device)
            self.interp_coeff=torch.tensor([
                [1, -183/64, 37/12, -145/128],
                [0,0,0,0],
                [0,1500/371, -1000/159, 1000/371],
                [0, -125/32, 125/12, -375/64],
                [0, 9477/3392, -729/106, 25515/6784],
                [0,-11/7, 11/3, -55/28],
                [0, 3/2, -4, 5/2]
            ], dtype=dtype,device=device)
        else:
            raise ValueError(f'Unknown solver: {solver_name}')
    def run(self,bar: Wait_bar):
        finished = False

        next_idx = 1 # for t_span output
        n_failures = 0
        integration_failed= False
        err_history=[]
        n_steps = 0

        n_out=0

        t_out=self.t_out
        y_out=self.y_out
        t=self.t
        y=self.y
        dtype=self.dtype
        device=self.device
        h_max=self.h_max
        h_min=self.h_min
        rtol=self.rtol
        atol=self.atol
        max_failures=self.max_failures
        t_final=self.t_final
        t0=self.t0
        t_dir=self.t_dir
        t_span=self.t_span
        waitbar=self.waitbar
        alpha=self.alpha
        beta=self.beta
        c_error=self.c_error
        ode_fcn=self.ode_fcn
        S=self.S
        chunk=self.chunk
        output_pos=self.output_pos
        n_t_span=self.n_t_span
        n_eq=self.n_eq
        refine=self.refine
        order=self.order
        interp_coeff=self.interp_coeff

        f1=ode_fcn(t,y).view(-1,1)
        
        n_calls=1

        t_out[n_out]=t
        y_out[:,n_out]=y.view(-1)

        t2pi=torch.tensor(2*math.pi,dtype=dtype,device=device)
        h = torch.min(h_max, torch.max(h_min, 0.1 * torch.abs(t_final -t0)))
        h_abs=torch.abs(h)

        while not finished:
            h_abs=torch.min(h_max, torch.max(h_min, h_abs))
            h = t_dir*h_abs
            if h_abs > torch.abs(t_final-t):
                h = t_final - t
                h_abs = torch.abs(h)
                finished = True
            failed = False
            y_list=y.clone()
            f_list=f1.clone()
            while True:
                t_new = t + h
                for i, (alpha_i, beta_i) in enumerate(zip(alpha,beta)):
                    y_list=torch.cat([y_list, y + h * f_list @ beta_i],dim=1)
                    if alpha_i == 1. :
                        ti=t_new
                    else:
                        ti=t+alpha_i*h
                    f_list=torch.cat([f_list,ode_fcn(ti,y_list[:,-1:])],dim=1)
                n_calls += 6
                y_new = y_list[:,-1:]
                err = f_list @ c_error

                err /= torch.max(torch.max(y.abs(),y_new.abs()), atol)
                err = h_abs *  torch.max(err.abs())
                err=err.item()
                # step acceptance
                accept_step=err <= rtol
                if accept_step:
                    n_failures = 0
                if h_abs <= h_min:
                    accept_step = True
                    n_failures +=1
                    failed=True
                    if n_failures >= max_failures:
                        bar.close(waitbar)
                        warnings.warn(
                            f"Step size reached minimum hmin = {h_min.item():.2e} at t={t.item():.2e}, but still cannot satisfy tolerance. "
                            f"Current error: {err:.2e}, Required tolerance: {rtol:.2e}. "
                            f"This may indicate a stiff ODE or overly strict tolerances. "
                            f"Consider using a stiff solver or relaxing tolerances.",
                            RuntimeWarning,
                            stacklevel=2
                        )
                        finished = True
                        integration_failed = True
                        break
                else:
                    n_failures = 0 # Reset if we're still above hmin
                if accept_step:
                    err_history.append(err)
                    break
                else:
                    if failed:
                        h_abs = torch.max(h_min, 0.5* h_abs)
                    else:
                        failed = True 
                        h_abs =step_after_nofailed(h_min,h_abs,rtol,err,order)
                    h = t_dir * h_abs
                    y_list=y.clone()
                    f_list=f1.clone()
                    finished = False
            n_steps += 1
            if integration_failed:
                break
            # Update waitbar if enabled
            bar.update(t_new, h, waitbar, finished)
            # output
            if output_pos ==2: # computed points
                nout_new = 1
                t_out_new = t_new.unsqueeze(0)
                y_out_new = y_new.view(-1,1)
            elif output_pos ==3: # computed points, with refinement
                t_ref = t + (t_new - t) * S
                nout_new = refine
                t_out_new = torch.cat([t_ref, t_new.unsqueeze(0)])
                y_interp = interp_fun(t_ref, t, y, h, f_list,interp_coeff)
                y_out_new = torch.cat([y_interp, y_new.view(-1,1)], dim=1) 
            else:
                nout_new = 0
                t_out_new = torch.tensor([], dtype=dtype, device=device)
                y_out_new = torch.tensor([], dtype=dtype, device=device)
                
                while next_idx < n_t_span:
                    if t_dir * (t_new - t_span[next_idx]) < 0:
                        break
                    nout_new += 1
                    t_out_new = torch.cat([t_out_new, t_span[next_idx].unsqueeze(0)])
                    if t_span[next_idx] == t_new:
                        y_out_new = torch.cat([y_out_new, y_new], dim=1)
                    else:
                        y_interp = interp_fun(t_span[next_idx], t, y, h, f_list, interp_coeff)
                        y_out_new = torch.cat([y_out_new, y_interp], dim=1)
                    next_idx += 1
            y_out_new = y_out_new % t2pi
            # Store output
            if nout_new > 0:
                old_nout = n_out
                n_out += nout_new

                if n_out+1 > t_out.shape[0]:
                    extra = max(chunk, nout_new)
                    tout_new_temp = torch.zeros(t_out.shape[0] + extra, dtype=dtype, device=device)
                    tout_new_temp[:t_out.shape[0]] = t_out
                    t_out = tout_new_temp
                    
                    yout_new_temp = torch.zeros(n_eq, y_out.shape[1] + extra, dtype=dtype, device=device)
                    yout_new_temp[:, :y_out.shape[1]] = y_out
                    y_out = yout_new_temp
            
                t_out[old_nout+1:n_out+1] = t_out_new
                y_out[:, old_nout+1:n_out+1] = y_out_new
            h_abs = _optimal_step_size(h_abs, err/rtol, order, failed)
        
            t=t_new
            y=y_new
            y=y % t2pi
            f1 = f_list[:,-1:]

        bar.close(waitbar)

        t_out= t_out[:n_out+1]
        y_out = y_out[:,:n_out+1]
        stats = {'n_calls': n_calls,
            'n_steps': n_steps,
            'n_output': n_out+1,
            'intergration': not integration_failed}
        err_info = {
            'err_history': err_history,
            'max_step_error': max(err_history) if err_history else 0.0
        }
        return t_out, y_out, stats, err_info
    
class eq3_solver(eq_solver):
    def run(self,bar: Wait_bar):
        finished = False

        next_idx = 1 # for t_span output
        n_failures = 0
        integration_failed= False
        err_history=[]
        n_steps = 0

        n_out=0

        t_out=self.t_out
        y_out=self.y_out
        t=self.t
        y=self.y
        dtype=self.dtype
        device=self.device
        h_max=self.h_max
        h_min=self.h_min
        rtol=self.rtol
        atol=self.atol
        max_failures=self.max_failures
        t_final=self.t_final
        t0=self.t0
        t_dir=self.t_dir
        t_span=self.t_span
        waitbar=self.waitbar
        alpha=self.alpha
        beta=self.beta
        c_error=self.c_error
        ode_fcn=self.ode_fcn
        S=self.S
        chunk=self.chunk
        output_pos=self.output_pos
        n_t_span=self.n_t_span
        n_eq=self.n_eq
        refine=self.refine
        order=self.order
        interp_coeff=self.interp_coeff

        f1=ode_fcn(t,y).view(-1,1)
        
        n_calls=1

        t_out[n_out]=t
        y_out[:,n_out]=y.view(-1)

        t2pi=torch.tensor(2*math.pi,dtype=dtype,device=device)
        h = torch.min(h_max, torch.max(h_min, 0.1 * torch.abs(t_final -t0)))
        h_abs=torch.abs(h)

        while not finished:
            h_abs=torch.min(h_max, torch.max(h_min, h_abs))
            h = t_dir*h_abs
            if h_abs > torch.abs(t_final-t):
                h = t_final - t
                h_abs = torch.abs(h)
                finished = True
            failed = False
            y_list=y.clone()
            f_list=f1.clone()
            while True:
                t_new = t + h
                for i, (alpha_i, beta_i) in enumerate(zip(alpha,beta)):
                    y_list=torch.cat([y_list, y + h * f_list @ beta_i],dim=1)
                    if alpha_i == 1. :
                        ti=t_new
                    else:
                        ti=t+alpha_i*h
                    f_list=torch.cat([f_list,ode_fcn(ti,y_list[:,-1:])],dim=1)
                n_calls += 6
                y_new = y_list[:,-1:]
                err = f_list @ c_error

                err /= torch.max(torch.max(y.abs(),y_new.abs()), atol)
                err = h_abs *  torch.max(err.abs())
                err=err.item()
                # step acceptance
                accept_step=err <= rtol
                if accept_step:
                    n_failures = 0
                if h_abs <= h_min:
                    accept_step = True
                    n_failures +=1
                    failed=True
                    if n_failures >= max_failures:
                        bar.close(waitbar)
                        warnings.warn(
                            f"Step size reached minimum hmin = {h_min.item():.2e} at t={t.item():.2e}, but still cannot satisfy tolerance. "
                            f"Current error: {err:.2e}, Required tolerance: {rtol:.2e}. "
                            f"This may indicate a stiff ODE or overly strict tolerances. "
                            f"Consider using a stiff solver or relaxing tolerances.",
                            RuntimeWarning,
                            stacklevel=2
                        )
                        finished = True
                        integration_failed = True
                        break
                else:
                    n_failures = 0 # Reset if we're still above hmin
                if accept_step:
                    err_history.append(err)
                    break
                else:
                    if failed:
                        h_abs = torch.max(h_min, 0.5* h_abs)
                    else:
                        failed = True 
                        h_abs =step_after_nofailed(h_min,h_abs,rtol,err,order)
                    h = t_dir * h_abs
                    y_list=y.clone()
                    f_list=f1.clone()
                    finished = False
            n_steps += 1
            if integration_failed:
                break
            # Update waitbar if enabled
            bar.update(t_new, h, waitbar, finished)
            # output
            if output_pos ==2: # computed points
                nout_new = 1
                t_out_new = t_new.unsqueeze(0)
                y_out_new = y_new.view(-1,1)
            elif output_pos ==3: # computed points, with refinement
                t_ref = t + (t_new - t) * S
                nout_new = refine
                t_out_new = torch.cat([t_ref, t_new.unsqueeze(0)])
                y_interp = interp_fun(t_ref, t, y, h, f_list,interp_coeff)
                y_out_new = torch.cat([y_interp, y_new.view(-1,1)], dim=1) 
            else:
                nout_new = 0
                t_out_new = torch.tensor([], dtype=dtype, device=device)
                y_out_new = torch.tensor([], dtype=dtype, device=device)
                
                while next_idx < n_t_span:
                    if t_dir * (t_new - t_span[next_idx]) < 0:
                        break
                    nout_new += 1
                    t_out_new = torch.cat([t_out_new, t_span[next_idx].unsqueeze(0)])
                    if t_span[next_idx] == t_new:
                        y_out_new = torch.cat([y_out_new, y_new], dim=1)
                    else:
                        y_interp = interp_fun(t_span[next_idx], t, y, h, f_list, interp_coeff)
                        y_out_new = torch.cat([y_out_new, y_interp], dim=1)
                    next_idx += 1
            y_out_new = _vec_normaliza(y_out_new)
            # Store output
            if nout_new > 0:
                old_nout = n_out
                n_out += nout_new

                if n_out+1 > t_out.shape[0]:
                    extra = max(chunk, nout_new)
                    tout_new_temp = torch.zeros(t_out.shape[0] + extra, dtype=dtype, device=device)
                    tout_new_temp[:t_out.shape[0]] = t_out
                    t_out = tout_new_temp
                    
                    yout_new_temp = torch.zeros(n_eq, y_out.shape[1] + extra, dtype=dtype, device=device)
                    yout_new_temp[:, :y_out.shape[1]] = y_out
                    y_out = yout_new_temp
            
                t_out[old_nout+1:n_out+1] = t_out_new
                y_out[:, old_nout+1:n_out+1] = y_out_new
            h_abs = _optimal_step_size(h_abs, err/rtol, order, failed)
        
            t=t_new
            y=y_new
            y=_vec_normaliza(y)
            f1 = f_list[:,-1:]

        bar.close(waitbar)

        t_out= t_out[:n_out+1]
        y_out = y_out[:,:n_out+1]
        stats = {'n_calls': n_calls,
            'n_steps': n_steps,
            'n_output': n_out+1,
            'intergration': not integration_failed}
        err_info = {
            'err_history': err_history,
            'max_step_error': max(err_history) if err_history else 0.0
        }
        return t_out, y_out, stats, err_info

def  _vec_normaliza(y:torch.Tensor):
    if y.shape[0] == 0: 
        return y
    else:
        return torch.nn.functional.normalize(y.view(-1,3,y.shape[-1]), dim=1,p=2).view_as(y)

def interp_fun(t_interp: torch.Tensor, t: torch.Tensor, y: torch.Tensor,h: torch.Tensor ,f_list: torch.Tensor, interp_coeff)-> torch.Tensor:
    """
    Interpolation function for Dormand-Prince method.
    """
    max_order = interp_coeff.shape[1] # interp_coeff.shape = [f_order, t_order]
    s = (t_interp - t) / h
    s = s.reshape(1,-1) #[1,t_l]
    s_list = s 
    if max_order > 1:
        for jj in range(max_order-1):
            s_list=torch.cat([s_list,s**(jj+2)],dim=0) # [t_order,t_l]
    s_coeff = interp_coeff @ s_list #[f_order, t_order] @ [t_order,t_l] = [f_order, t_l]
    y_interp = y + h* f_list@ s_coeff  # [y_d, f_order] @ [f_order, t_l] =[y_d,t_l]
    return y_interp

def step_after_nofailed(h_min,h_abs,rtol,err,order):
    return torch.max(h_min, h_abs * max(0.1, 0.8 * (rtol / err) ** (1/order)))

def _optimal_step_size(last_step, error_ratio, order, failedflag, safety=0.8, ifactor=4.0, dfactor=0.2):
    """Calculate the optimal size for the next step."""
    if failedflag:
        return last_step
    else: 
        ifactor=torch.as_tensor(ifactor,dtype=last_step.dtype,device=last_step.device)
        dfactor=torch.as_tensor(dfactor,dtype=last_step.dtype,device=last_step.device)
        if error_ratio == 0:
            return last_step * ifactor
        if error_ratio < 1:
            dfactor = torch.ones((), dtype=last_step.dtype, device=last_step.device)
        error_ratio = error_ratio.type_as(last_step)
        exponent = torch.as_tensor(order, dtype=last_step.dtype, device=last_step.device).reciprocal()
        factor = torch.min(ifactor, torch.max(safety / error_ratio ** exponent, dfactor))
        return last_step * factor

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
    sf=eq3_solver(ode_fun, t_span, y0, solver, options)               
    bar=Wait_bar(t_span, sf.waitbar)  # Initialize the progress bar
    t, ang, stats, error_info =sf.run(bar)
    return t, ang, stats, error_info

def ode_sde_em(f: Callable, # function
                t_span: torch.Tensor, 
                y0: torch.Tensor, 
                options: Optional[Dict[str, Any]] = None
        ) -> Tuple[torch.Tensor, torch.Tensor, Dict, Dict]:
    """
    Modified Euler-Maruyama method for SDEs.
    
    Parameters:
    -----------
    f : callable
        function: f0, g0 = f(t, y) with f0 the drift term and g0 the diffusion term
    t_span : torch.Tensor
        Time span for integration
    y0 : torch.Tensor
        Initial condition
    options : dict, optional
        Integration options

    Returns:
    --------
    t : torch.Tensor
        Time points at which output is given
    y : torch.Tensor 
        Solution at tout
    stats : dict
        Integration statistics
    err_info : dict
        Error information
    """
    if options is None:
        options = {}

    # Initialize options
    waitbar = options.get('waitbar', True)
    
    # Extract odeset options
    rtol = options.get('rel_tol', 1e-3)
    atol = options.get('abs_tol', 1e-6)
    norm_control = options.get('NormControl', 'off') == 'on'
    max_failures = options.get('max_consecutive_failures', 10)
    # Initialize waitbar
    bar=Wait_bar(t_span, waitbar)  # Initialize the progress bar
    
    # Initialize solution storage
    t0 = t_span[0]
    t_final = t_span[-1]
    t_dir = torch.sign(t_final - t0)
    
    # Ensure y0 is 1D tensor
    original_shape = y0.shape
    y0 = y0.reshape(-1,1)
    n_eq = y0.shape[0]
    
    # Data type
    dtype = y0.dtype
    device = y0.device

    m1_sqrt2=torch.tensor(1-math.sqrt(2),dtype=dtype,device=device)
    sqrt2=torch.tensor(math.sqrt(2),dtype=dtype,device=device)
    
    # Step size constraints
    h_min = 16 * torch.finfo(dtype).eps
    h_min=torch.tensor(h_min,dtype=dtype,device=device)
    safe_h_max = 16.0 * torch.finfo(dtype).eps * torch.max(torch.abs(t0), torch.abs(t_final))
    default_h_max = torch.max(torch.abs(t_final - t0), safe_h_max)
    h_max = torch.min(torch.abs(t_final - t0), 
                    torch.tensor(options.get('MaxStep', default_h_max.item()), dtype=dtype, device=device))
    threshold = torch.tensor(atol, dtype=dtype, device=device)
    if norm_control:
        norm_y = torch.norm(y0)
    else:
        norm_y = torch.tensor(0.0, dtype=dtype, device=device)
    
    t = t0.clone()
    y = y0.clone()
    
    # Output configuration
    n_t_span = t_span.shape[0]
    refine = options.get('Refine', 4)
    
    if n_t_span > 2:
        output_pos = 1  # output only at tspan points
    elif refine <= 1:
        output_pos = 2  # computed points, no refinement
    else:
        output_pos = 3  # computed points, with refinement
        S = torch.linspace(1/refine, 1 - 1/refine, refine - 1, dtype=dtype, device=device)
    
    # Initialize output arrays
    if n_t_span > 2:
        t_out = torch.zeros(n_t_span, dtype=dtype, device=device)
        y_out = torch.zeros(n_eq, n_t_span, dtype=dtype, device=device)
    else:
        chunk = min(max(100, 50 * refine), refine + (2**13) // n_eq)
        t_out = torch.zeros(chunk, dtype=dtype, device=device)
        y_out = torch.zeros(n_eq, chunk, dtype=dtype, device=device)
    
    n_out = 0
    t_out[n_out] = t
    y_out[:, n_out] = y.view(-1)
    
    error_history = []
    n_calls = 0
    n_steps = 0

    # Pi value
    t2pi=torch.tensor(2*math.pi,dtype=dtype,device=device)
    # Initial step size
    h = torch.min(h_max, torch.max(h_min, 0.1 * torch.abs(t_final - t0)))
    h_abs = torch.abs(h)
    # Initial function evaluation
    f1, g1 = f(t, y)
    noise_dim=g1.shape[1]
    n_calls += 1

    finished = False
    next_idx = 1  # for tspan output
    
    # Main integration loop
    n_failures = 0
    integration_failed= False

    while not finished:
        h_abs = torch.min(h_max, torch.max(h_min, h_abs))
        h = t_dir * h_abs
        if 1.1 * h_abs >= torch.abs(t_final - t):
            h = t_final - t
            h_abs = torch.abs(h)
            finished = True
        
        no_failed = True
        W1=torch.randn(noise_dim,1,dtype=dtype,device=device)
        W2=torch.randn(noise_dim,1,dtype=dtype,device=device)
        # W=W1+W2
        while True:
            gw11 = g1 @ W1
            h_absinv2_sqrt=torch.sqrt(h_abs/2)
            y2 = y+ f1 * h/2 + gw11 * h_absinv2_sqrt
            t2 = t + h / 2
            f2, g2 = f(t2, y2)
            

            gw22 = g2 @ W2
            y_new = y2 + f2 *h/2 + gw22 * h_absinv2_sqrt
            t_new = t + h

            gw12 = g1 @ W2
            # y_full = y + f1 * h + gw * torch.sqrt(h_abs)
            
            n_calls += 1
            
            fE= h/2*(f1-f2) + h_absinv2_sqrt*(m1_sqrt2*gw11+gw22-sqrt2*gw12)
            if norm_control:
                norm_y_new = torch.norm(y_new)
                scaling_Factor = torch.max(torch.max(norm_y, norm_y_new), threshold)
                err = h_abs *torch.norm(fE) / scaling_Factor
            else: 
                scaling_Factor = torch.max(torch.max(torch.abs(y), torch.abs(y_new)), threshold)
                err = fE / scaling_Factor
                err = h_abs * torch.max(err.abs())

            err=err.item()
            # Step acceptance
            accept_step= err <= rtol
            if accept_step:
                    n_failures = 0
            if h_abs <= h_min: # accept the step when h reaches h_min
                    accept_step = True
                    n_failures +=1
                    failed=True
                    if n_failures >= max_failures: # Stop integration when h is too small for too many  consecutive times
                        bar.close(waitbar)
                        warnings.warn(
                            f"Step size reached minimum hmin = {h_min.item():.2e} at t={t.item():.2e}, but still cannot satisfy tolerance. "
                            f"Current error: {err:.2e}, Required tolerance: {rtol:.2e}. "
                            f"This may indicate a stiff ODE or overly strict tolerances. "
                            f"Consider using a stiff solver or relaxing tolerances.",
                            RuntimeWarning,
                            stacklevel=2
                        )
                        finished = True
                        integration_failed = True
                        break
            else:
                n_failures = 0 # Reset if we're still above hmin
            if accept_step:
                error_history.append(err)
                break
            else:
                # Adaptive mode: shrink step and retry
                if no_failed:
                    no_failed = False
                    h_abs = torch.max(h_min, h_abs * max(0.1, 0.8 * (rtol / err) ** (1/1.5)))
                else:
                    h_abs = torch.max(h_min, 0.5 * h_abs)
                h = t_dir * h_abs
                finished = False
        n_steps += 1
        if integration_failed:
            break

        # Update waitbar if enabled
        bar.update(t_new, h, waitbar, finished)
        
        # Output processing
        if output_pos == 2:  # computed points, no refinement
            n_out_new = 1
            t_out_new = t_new.unsqueeze(0)
            y_out_new = y_new.unsqueeze(1)
        elif output_pos == 3:  # computed points, with refinement
            t_ref = t + (t_new - t) * S
            n_out_new = refine
            t_out_new = torch.cat([t_ref, t_new.unsqueeze(0)])
            y_ntrp = ntrp_em(t_ref, t, y, h, y2, y_new)
            y_out_new = torch.cat([y_ntrp, y_new.unsqueeze(1)], dim=1) 
        else:  # output only at tspan points
            n_out_new = 0
            t_out_new = torch.tensor([], dtype=dtype, device=device)
            y_out_new = torch.tensor([], dtype=dtype, device=device)
            
            while next_idx < n_t_span:
                if t_dir * (t_new - t_span[next_idx]) < 0:
                    break
                n_out_new += 1
                t_out_new = torch.cat([t_out_new, t_span[next_idx].unsqueeze(0)])
                if t_span[next_idx] == t_new:
                    y_out_new = torch.cat([y_out_new, y_new], dim=1)
                else:
                    y_ntrp = ntrp_em(t_span[next_idx].unsqueeze(0), t, y, h, y2, y_new)
                    y_out_new = torch.cat([y_out_new, y_ntrp], dim=1)
                next_idx += 1
        y_out_new = y_out_new % t2pi
        # Store output
        if n_out_new > 0:
            old_n_out = n_out
            n_out = n_out + n_out_new
            
            # Expand arrays if needed
            if n_out+1 > t_out.shape[0]:
                extra = max(chunk, n_out_new)
                t_out_new_temp = torch.zeros(t_out.shape[0] + extra, dtype=dtype, device=device)
                t_out_new_temp[:t_out.shape[0]] = t_out
                t_out = t_out_new_temp
                
                y_out_new_temp = torch.zeros(n_eq, y_out.shape[1] + extra, dtype=dtype, device=device)
                y_out_new_temp[:, :y_out.shape[1]] = y_out
                y_out = y_out_new_temp
            
            t_out[old_n_out+1:n_out+1] = t_out_new
            y_out[:, old_n_out+1:n_out+1] = y_out_new
        
        # Step size adjustment for adaptive mode
        if  no_failed:
            temp = 1.25 * (err / rtol) ** (1/1.5)
            if temp > 0.2:
                h_abs = h_abs / temp
            else:
                h_abs = 5.0 * h_abs
        
        # Advance integration
        t = t_new
        y = y_new % t2pi
        if norm_control:
            norm_y = norm_y_new
        f1, g1 = f(t,y)
        n_calls += 1
    # Close waitbar
    bar.close(waitbar)

    # Prepare outputs
    t_out = t_out[:n_out+1]
    y_out = y_out[:, :n_out+1]
    
    stats = {'n_calls': n_calls,
             'n_steps': n_steps,
             'n_output': n_out+1,
             'intergration': not integration_failed}
    err_info = {
        'err_history': error_history,
        'max_step_error': max(error_history) if error_history else 0.0
    }
    return t_out,y_out,stats,err_info
def ode3_sde_em(f: Callable, # function
                t_span: torch.Tensor, 
                y0: torch.Tensor, 
                options: Optional[Dict[str, Any]] = None
        ) -> Tuple[torch.Tensor, torch.Tensor, Dict, Dict]:
    """
    Modified Euler-Maruyama method for SDEs.
    
    Parameters:
    -----------
    f : callable
        function: f0, g0 = f(t, y) with f0 the drift term and g0 the diffusion term
    t_span : torch.Tensor
        Time span for integration
    y0 : torch.Tensor
        Initial condition
    options : dict, optional
        Integration options

    Returns:
    --------
    t : torch.Tensor
        Time points at which output is given
    y : torch.Tensor 
        Solution at tout
    stats : dict
        Integration statistics
    err_info : dict
        Error information
    """
    if options is None:
        options = {}

    # Initialize options
    waitbar = options.get('waitbar', True)
    
    # Extract odeset options
    rtol = options.get('rel_tol', 1e-3)
    atol = options.get('abs_tol', 1e-6)
    norm_control = options.get('NormControl', 'off') == 'on'
    max_failures = options.get('max_consecutive_failures', 10)
    # Initialize waitbar
    bar=Wait_bar(t_span, waitbar)  # Initialize the progress bar
    
    # Initialize solution storage
    t0 = t_span[0]
    t_final = t_span[-1]
    t_dir = torch.sign(t_final - t0)
    
    # Ensure y0 is 1D tensor
    original_shape = y0.shape
    y0 = y0.reshape(-1,1)
    n_eq = y0.shape[0]
    
    # Data type
    dtype = y0.dtype
    device = y0.device

    m1_sqrt2=torch.tensor(1-math.sqrt(2),dtype=dtype,device=device)
    sqrt2=torch.tensor(math.sqrt(2),dtype=dtype,device=device)
    
    # Step size constraints
    h_min = 16 * torch.finfo(dtype).eps
    h_min=torch.tensor(h_min,dtype=dtype,device=device)
    safe_h_max = 16.0 * torch.finfo(dtype).eps * torch.max(torch.abs(t0), torch.abs(t_final))
    default_h_max = torch.max(torch.abs(t_final - t0), safe_h_max)
    h_max = torch.min(torch.abs(t_final - t0), 
                    torch.tensor(options.get('MaxStep', default_h_max.item()), dtype=dtype, device=device))
    threshold = torch.tensor(atol, dtype=dtype, device=device)
    if norm_control:
        norm_y = torch.norm(y0)
    else:
        norm_y = torch.tensor(0.0, dtype=dtype, device=device)
    
    t = t0.clone()
    y = y0.clone()
    
    # Output configuration
    n_t_span = t_span.shape[0]
    refine = options.get('Refine', 4)
    
    if n_t_span > 2:
        output_pos = 1  # output only at tspan points
    elif refine <= 1:
        output_pos = 2  # computed points, no refinement
    else:
        output_pos = 3  # computed points, with refinement
        S = torch.linspace(1/refine, 1 - 1/refine, refine - 1, dtype=dtype, device=device)
    
    # Initialize output arrays
    if n_t_span > 2:
        t_out = torch.zeros(n_t_span, dtype=dtype, device=device)
        y_out = torch.zeros(n_eq, n_t_span, dtype=dtype, device=device)
    else:
        chunk = min(max(100, 50 * refine), refine + (2**13) // n_eq)
        t_out = torch.zeros(chunk, dtype=dtype, device=device)
        y_out = torch.zeros(n_eq, chunk, dtype=dtype, device=device)
    
    n_out = 0
    t_out[n_out] = t
    y_out[:, n_out] = y.view(-1)
    
    error_history = []
    n_calls = 0
    n_steps = 0

    # Pi value
    t2pi=torch.tensor(2*math.pi,dtype=dtype,device=device)
    # Initial step size
    h = torch.min(h_max, torch.max(h_min, 0.1 * torch.abs(t_final - t0)))
    h_abs = torch.abs(h)
    # Initial function evaluation
    f1, g1 = f(t, y)
    noise_dim=g1.shape[1]
    n_calls += 1

    finished = False
    next_idx = 1  # for tspan output
    
    # Main integration loop
    n_failures = 0
    integration_failed= False

    while not finished:
        h_abs = torch.min(h_max, torch.max(h_min, h_abs))
        h = t_dir * h_abs
        if 1.1 * h_abs >= torch.abs(t_final - t):
            h = t_final - t
            h_abs = torch.abs(h)
            finished = True
        
        no_failed = True
        W1=torch.randn(noise_dim,1,dtype=dtype,device=device)
        W2=torch.randn(noise_dim,1,dtype=dtype,device=device)
        W=W1+W2
        while True:
            gw11 = g1 @ W1
            h_absinv2_sqrt=torch.sqrt(h_abs/2)
            y2 = y+ f1 * h/2 + gw11 * h_absinv2_sqrt
            t2 = t + h / 2
            f2, g2 = f(t2, y2)
            

            gw22 = g2 @ W2
            y_new = y2 + f2 *h/2 + gw22 * h_absinv2_sqrt
            t_new = t + h

            gw12 = g1 @ W2
            # y_full = y + f1 * h + gw * torch.sqrt(h_abs)
            
            n_calls += 1
            
            fE= h/2*(f1-f2) + h_absinv2_sqrt*(m1_sqrt2*gw11+gw22-sqrt2*gw12)
            if norm_control:
                norm_y_new = torch.norm(y_new)
                scaling_Factor = torch.max(torch.max(norm_y, norm_y_new), threshold)
                err = h_abs *torch.norm(fE) / scaling_Factor
            else: 
                scaling_Factor = torch.max(torch.max(torch.abs(y), torch.abs(y_new)), threshold)
                err = fE / scaling_Factor
                err = h_abs * torch.max(err.abs())

            err=err.item()
            # Step acceptance
            accept_step= err <= rtol
            if accept_step:
                    n_failures = 0
            if h_abs <= h_min: # accept the step when h reaches h_min
                    accept_step = True
                    n_failures +=1
                    failed=True
                    if n_failures >= max_failures: # Stop integration when h is too small for too many  consecutive times
                        bar.close(waitbar)
                        warnings.warn(
                            f"Step size reached minimum hmin = {h_min.item():.2e} at t={t.item():.2e}, but still cannot satisfy tolerance. "
                            f"Current error: {err:.2e}, Required tolerance: {rtol:.2e}. "
                            f"This may indicate a stiff ODE or overly strict tolerances. "
                            f"Consider using a stiff solver or relaxing tolerances.",
                            RuntimeWarning,
                            stacklevel=2
                        )
                        finished = True
                        integration_failed = True
                        break
            else:
                n_failures = 0 # Reset if we're still above hmin
            if accept_step:
                error_history.append(err)
                break
            else:
                # Adaptive mode: shrink step and retry
                if no_failed:
                    no_failed = False
                    h_abs = torch.max(h_min, h_abs * max(0.1, 0.8 * (rtol / err) ** (1/1.5)))
                else:
                    h_abs = torch.max(h_min, 0.5 * h_abs)
                h = t_dir * h_abs
                finished = False
        n_steps += 1
        if integration_failed:
            break

        # Update waitbar if enabled
        bar.update(t_new, h, waitbar, finished)
        
        # Output processing
        if output_pos == 2:  # computed points, no refinement
            n_out_new = 1
            t_out_new = t_new.unsqueeze(0)
            y_out_new = y_new.unsqueeze(1)
        elif output_pos == 3:  # computed points, with refinement
            t_ref = t + (t_new - t) * S
            n_out_new = refine
            t_out_new = torch.cat([t_ref, t_new.unsqueeze(0)])
            y_ntrp = ntrp_em(t_ref, t, y, h, y2, y_new)
            y_out_new = torch.cat([y_ntrp, y_new.unsqueeze(1)], dim=1) 
        else:  # output only at tspan points
            n_out_new = 0
            t_out_new = torch.tensor([], dtype=dtype, device=device)
            y_out_new = torch.tensor([], dtype=dtype, device=device)
            
            while next_idx < n_t_span:
                if t_dir * (t_new - t_span[next_idx]) < 0:
                    break
                n_out_new += 1
                t_out_new = torch.cat([t_out_new, t_span[next_idx].unsqueeze(0)])
                if t_span[next_idx] == t_new:
                    y_out_new = torch.cat([y_out_new, y_new], dim=1)
                else:
                    y_ntrp = ntrp_em(t_span[next_idx].unsqueeze(0), t, y, h, y2, y_new)
                    y_out_new = torch.cat([y_out_new, y_ntrp], dim=1)
                next_idx += 1
        y_out_new = _vec_normaliza(y_out_new)
        # Store output
        if n_out_new > 0:
            old_n_out = n_out
            n_out = n_out + n_out_new
            
            # Expand arrays if needed
            if n_out+1 > t_out.shape[0]:
                extra = max(chunk, n_out_new)
                t_out_new_temp = torch.zeros(t_out.shape[0] + extra, dtype=dtype, device=device)
                t_out_new_temp[:t_out.shape[0]] = t_out
                t_out = t_out_new_temp
                
                y_out_new_temp = torch.zeros(n_eq, y_out.shape[1] + extra, dtype=dtype, device=device)
                y_out_new_temp[:, :y_out.shape[1]] = y_out
                y_out = y_out_new_temp
            
            t_out[old_n_out+1:n_out+1] = t_out_new
            y_out[:, old_n_out+1:n_out+1] = y_out_new
        
        # Step size adjustment for adaptive mode
        if  no_failed:
            temp = 1.25 * (err / rtol) ** (1/1.5)
            if temp > 0.2:
                h_abs = h_abs / temp
            else:
                h_abs = 5.0 * h_abs
        
        # Advance integration
        t = t_new
        y = y_new
        y = _vec_normaliza(y)
        if norm_control:
            norm_y = norm_y_new
        f1, g1 = f(t,y)
        n_calls += 1
    # Close waitbar
    bar.close(waitbar)

    # Prepare outputs
    t_out = t_out[:n_out+1]
    y_out = y_out[:, :n_out+1]
    
    stats = {'n_calls': n_calls,
             'n_steps': n_steps,
             'n_output': n_out+1,
             'intergration': not integration_failed}
    err_info = {
        'err_history': error_history,
        'max_step_error': max(error_history) if error_history else 0.0
    }
    return t_out,y_out,stats,err_info

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