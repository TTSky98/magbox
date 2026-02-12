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
class eq_solver:
    def __init__(self,odeFcn, t_span, y0: torch.Tensor, solver_name, options):
        self.device=y0.device
        self.dtype=y0.dtype
        self.t2pi=torch.tensor(2*math.pi,dtype=self.dtype,device=self.device)
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
        h_max = torch.min(torch.abs(self.t_final - self.t0), torch.min(self.max_step, default_h_max))

        t = self.t0.clone()
        y = y0.clone()


        n_t_span=t_span.shape[0]
        n_eq=y0.shape
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
            y_out = torch.zeros((n_t_span,)+n_eq, dtype=dtype, device=device)
        else:
            chunk = min(max(100, 50 * refine), refine + (2**13) // n_eq)
            t_out = torch.zeros(chunk, dtype=dtype, device=device)
            y_out = torch.zeros((int(chunk),)+n_eq, dtype=dtype, device=device)
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
    def _get_parameters(self):
        return self.t_out, self.y_out, self.t, self.y, self.dtype, self.device, self.h_max, self.h_min, self.rtol, self.atol, self.max_failures, self.t_final, self.t0, self.t_dir, self.t_span, self.waitbar, self.alpha, self.beta, self.c_error, self.ode_fcn, self.S, self.chunk, self.output_pos, self.n_t_span, self.n_eq, self.refine, self.order, self.interp_coeff
    @staticmethod
    def _one_step(f_list: torch.Tensor, y_list: torch.Tensor, t, y, h, ode_fcn, alpha,beta):
        t_new = t + h
        # f_list shape: [b, N, ...]  y_list shape: [7, N, ...] 
        # f_list[0] has been filled with k1
        for i, (alpha_i, beta_i) in enumerate(zip(alpha,beta)):
            # Use data up to the current stage i+1 (indices 0 to i) to compute calculating stage
            # beta_i has size i+1
            # Efficiently compute dy using pre-allocated f_list slice
            dy = torch.einsum('s...,sj->...', f_list[:i+1], beta_i)
            
            y_next = y + h * dy
            y_list[i+1] = y_next

            if alpha_i == 1. :
                ti=t_new
            else:
                ti=t+alpha_i*h

            k_next = ode_fcn(ti, y_next)
            f_list[i+1] = k_next

        return t_new, y_list, f_list
    @staticmethod
    def _get_err(f_list: torch.Tensor, c_error: torch.Tensor, h, y, y_new, atol):
        err = torch.einsum('s...,si->...', f_list, c_error)
        err /= torch.max(torch.max(y.abs(),y_new.abs()), atol)
        err = h *  torch.max(err.abs())
        return err.item()
    
    @staticmethod
    def _get_outputs(output_pos, interp_fun, t, S, refine, t_new, y_new, y, h, f_list, interp_coeff, dtype, device, n_t_span, t_span, next_idx, t_dir):
        if output_pos ==2: # computed points
                nout_new = 1
                t_out_new = t_new.unsqueeze(0)
                y_out_new = y_new
        elif output_pos ==3: # computed points, with refinement
            t_ref = t + (t_new - t) * S
            nout_new = refine
            t_out_new = torch.cat([t_ref, t_new.unsqueeze(0)])
            y_interp = interp_fun(t_ref, t, y, h, f_list, interp_coeff)
            y_out_new = torch.cat([y_interp, y_new.unsqueeze(0)], dim=0) 
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
                    y_out_new = torch.cat([y_out_new, y_new.unsqueeze(0)], dim=0)
                else:
                    y_interp = interp_fun(t_span[next_idx], t, y, h, f_list, interp_coeff)
                    y_out_new = torch.cat([y_out_new, y_interp], dim=0)
                next_idx += 1
        return nout_new, t_out_new, y_out_new, next_idx
    @staticmethod
    def _interp_fun(t_interp, t, y, h, y_list, interp_coeff):
        y_interp = interp_fun(t_interp, t, y, h, y_list, interp_coeff)
        return y_interp
    
    def _after_process(self,y: torch.Tensor) -> torch.Tensor:
        return y/self.t2pi
    
    def run(self,bar: Wait_bar):
        finished = False

        next_idx = 1 # for t_span output
        n_failures = 0
        integration_failed= False
        err_history=[]
        n_steps = 0

        n_out=0

        t_out, y_out, t, y, dtype, device, h_max, h_min, rtol, atol, max_failures, t_final, t0, t_dir, t_span, waitbar, alpha, beta, c_error, ode_fcn, S, chunk, output_pos, n_t_span, n_eq, refine, order, interp_coeff = self._get_parameters()

        n_beta=len(beta)
        f1=ode_fcn(t,y)
        
        n_calls=1

        t_out[n_out]=t
        y_out[n_out,...]=y

        h = torch.min(h_max, torch.max(h_min, 0.1 * torch.abs(t_final -t0)))
        h_abs=torch.abs(h)

        # Pre-allocate buffers for RK45 stages (7 stages)
        stage_shape = (n_beta+1,) + y.shape
        f_list_buffer = torch.zeros(stage_shape, dtype=dtype, device=device)
        y_list_buffer = torch.zeros(stage_shape, dtype=dtype, device=device)

        while not finished:
            h_abs=torch.min(h_max, torch.max(h_min, h_abs))
            h = t_dir*h_abs
            if h_abs > torch.abs(t_final-t):
                h = t_final - t
                h_abs = torch.abs(h)
                finished = True
            failed = False
            y_list_buffer[0] = y
            f_list_buffer[0] = f1
            while True:
                t_new, y_list_buffer, f_list_buffer = self._one_step(f_list_buffer, y_list_buffer, t, y, h, ode_fcn, alpha, beta)
                n_calls += n_beta
                y_new = y_list_buffer[n_beta]
                err = self._get_err(f_list_buffer, c_error, h_abs, y, y_new, atol)
                # step acceptance
                accept_step = err <= rtol
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
                    y_list_buffer[0] = y
                    f_list_buffer[0] = f1
                    finished = False
            n_steps += 1
            if integration_failed:
                break
            # Update waitbar if enabled
            bar.update(t_new, h, waitbar, finished)
            # output
            nout_new, t_out_new, y_out_new, next_idx = self._get_outputs(output_pos,self._interp_fun, t, S, refine, t_new, y_new, y, h, f_list_buffer, interp_coeff, dtype, device, n_t_span, t_span, next_idx,t_dir)
            y_out_new = self._after_process(y_out_new)
            # Store output
            if nout_new > 0:
                old_nout = n_out
                n_out += nout_new

                if n_out+1 > t_out.shape[0]:
                    extra = max(chunk, nout_new)
                    tout_new_temp = torch.zeros(t_out.shape[0] + extra, dtype=dtype, device=device)
                    tout_new_temp[:t_out.shape[0]] = t_out
                    t_out = tout_new_temp
                    
                    yout_new_temp = torch.zeros((y_out.shape[0] + extra,)+n_eq, dtype=dtype, device=device)
                    yout_new_temp[:y_out.shape[0],...] = y_out
                    y_out = yout_new_temp
            
                t_out[old_nout+1:n_out+1] = t_out_new
                y_out[old_nout+1:n_out+1,...] = y_out_new
            h_abs = _optimal_step_size(h_abs, err/rtol, order, failed)
        
            t=t_new
            y=y_new
            y=self._after_process(y)
            f1 = f_list_buffer[6]

        bar.close(waitbar)

        t_out= t_out[:n_out+1]
        y_out = y_out[:n_out+1,...]
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
    def _after_process(self, y: torch.Tensor) -> torch.Tensor:
        return _vec_normaliza(y)

class sde_solver(eq_solver):
    def _tableau(self, solver_name):
        device=self.device
        dtype=self.dtype
        if solver_name == 'EM':
            self.order=2
            self.adaptive_order = 1.5
            m1_sqrt2=torch.tensor(1-math.sqrt(2),dtype=dtype,device=device)
            sqrt2=torch.tensor(math.sqrt(2),dtype=dtype,device=device)
            self.alpha=torch.tensor([],
                                    dtype=dtype,device=device)
            self.beta=torch.tensor([],
                                   dtype=dtype,device=device)
            self.c_error=[torch.tensor([1/2, -1/2],dtype=dtype,device=device).view(-1,1),
                        torch.tensor([-sqrt2/2, -sqrt2/2, sqrt2/2],dtype=dtype,device=device).view(-1,1)]
            self.interp_coeff=torch.tensor([[-3,2],
                                            [4,-4],
                                            [-1,2]],
                                    dtype=dtype,device=device)  
            pass
        else:
            super()._tableau(solver_name)
    @staticmethod
    def _one_step(f1, g1, t, y, h, W, sde_fcn, order):
        gw11 = g1(W[0])
        h_absinv2_sqrt = torch.sqrt(h/2)
        y2 = y + f1 * h/2 + gw11 * h_absinv2_sqrt
        t2 = t + h / 2
        f2, g2, _ =sde_fcn(t2, y2)

        gw22 = g2(W[1])
        y_new = y2 + f2 * h/2 + gw22 * h_absinv2_sqrt
        t_new = t + h

        gw12 = g1(W[0]+W[1])

        y_list=torch.stack([y, y2, y_new], dim=0)
        f_list=torch.stack([f1, f2], dim=0)
        gw_list=torch.stack([gw11, gw22, gw12], dim=0)

        return t_new, y_list, f_list, gw_list
    @staticmethod
    def _get_err(f_list, gw_list, c_error, h, y, y_new, atol):
        err = h * torch.einsum('s...,si->...', f_list, c_error[0])
        err += torch.sqrt(h) * torch.einsum('s...,si->...', gw_list, c_error[1])
        err /= torch.max(torch.max(y.abs(), y_new.abs()), atol)
        err = torch.max(err.abs())
        return err.item()
    
    @staticmethod
    def _interp_fun(t_interp, t, y, h, y_list, interp_coeff):
        y_interp = interp_fun(t_interp, t, y, h, y_list, interp_coeff,is_dy=False)
        return y_interp
    
    def run(self, bar: Wait_bar):
        t_out, y_out, t, y, dtype, device, h_max, h_min, rtol, atol, max_failures, t_final, t0, t_dir, t_span, waitbar, alpha, beta, c_error, sde_fcn, S, chunk, output_pos, n_t_span, n_eq, refine, order, interp_coeff = self._get_parameters()
        if hasattr(self, 'adaptive_order'):
            adaptive_order = self.adaptive_order
        else:
            adaptive_order = order - 0.5

        n_out = 0
        t_out[n_out] = t
        y_out[n_out, ...] = y

        # Pre-allocate error history as tensor for better performance
        max_steps_estimate = max(1000, int(torch.abs(t_final - t0) / h_min) if h_min > 0 else 1000)
        error_history = torch.zeros(max_steps_estimate, dtype=dtype, device=device)
        error_idx = 0
        n_calls = 0
        n_steps = 0
        
        h = torch.min(h_max, torch.max(h_min, 0.1 * torch.abs(t_final - t0)))
        h_abs = torch.abs(h)

        f1, g1, noise_dim = sde_fcn(t, y)
        n_calls += 1

        finished = False
        next_idx = 1
        n_failures = 0
        integration_failed = False

        while not finished:
            h_abs = torch.min(h_max, torch.max(h_min, h_abs))
            h = t_dir * h_abs
            if 1.1 * h_abs >= torch.abs(t_final - t):
                h = t_final - t
                h_abs = torch.abs(h)
                finished = True
            
            no_failed = True

            W=torch.randn((order,)+noise_dim,dtype=dtype,device=device)

            while True:
                t_new, y_list, f_list, gw_list = self._one_step(f1, g1, t, y, h, W, sde_fcn, order)
                
                n_calls += 1
                y_new= y_list[order]
                
                err = self._get_err(f_list, gw_list, c_error, h_abs, y, y_new, atol)
                accept_step = err <= rtol

                if accept_step:
                    n_failures = 0
                
                if h_abs <= h_min:
                    accept_step = True
                    n_failures += 1
                    no_failed = False
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
                    n_failures = 0
                
                if accept_step:
                    # Store error in pre-allocated tensor
                    if error_idx >= error_history.shape[0]:
                        # Expand if needed (rare case)
                        new_size = error_history.shape[0] * 2
                        error_history_new = torch.zeros(new_size, dtype=dtype, device=device)
                        error_history_new[:error_idx] = error_history
                        error_history = error_history_new
                    error_history[error_idx] = err
                    error_idx += 1
                    break
                else:
                    if no_failed:
                        no_failed = False
                        h_abs = torch.max(h_min, h_abs * max(0.1, 0.8 * (rtol / err) ** (1/adaptive_order)))
                    else:
                        h_abs = torch.max(h_min, 0.5 * h_abs)
                    h = t_dir * h_abs
                    finished = False
            
            n_steps += 1
            if integration_failed:
                break
            
            bar.update(t_new, h, waitbar, finished)

            # Output processing using ntrp
            n_out_new, t_out_new, y_out_new, next_idx = self._get_outputs(output_pos, self._interp_fun, t, S, refine, t_new, y_new, y, h, y_list, interp_coeff, dtype, device, n_t_span, t_span, next_idx, t_dir)
            y_out_new = self._after_process(y_out_new)
            
            if n_out_new > 0:
                old_n_out = n_out
                n_out = n_out + n_out_new
                
                if n_out + 1 > t_out.shape[0]:
                    # Use 1.5x growth strategy for better amortized performance
                    extra = max(chunk, n_out_new, int(t_out.shape[0] * 0.5))
                    t_out_new_temp = torch.zeros(t_out.shape[0] + extra, dtype=dtype, device=device)
                    t_out_new_temp[:t_out.shape[0]] = t_out
                    t_out = t_out_new_temp
                    
                    y_out_new_temp = torch.zeros((y_out.shape[0] + extra,)+n_eq, dtype=dtype, device=device)
                    y_out_new_temp[:y_out.shape[0],...] = y_out
                    y_out = y_out_new_temp
                
                t_out[old_n_out+1:n_out+1] = t_out_new
                y_out[old_n_out+1:n_out+1,...] = y_out_new

            h_abs = _optimal_step_size(h_abs, err/rtol, adaptive_order, no_failed)
            
            t = t_new
            y = y_new

            y = self._after_process(y) 

            f1, g1, _ = sde_fcn(t, y)
            n_calls += 1

        bar.close(waitbar)
        
        t_out = t_out[:n_out+1]
        y_out = y_out[:n_out+1, ...]

        # Trim error_history to actual size
        error_history = error_history[:error_idx]
        
        stats = {'n_calls': n_calls,
             'n_steps': n_steps,
             'n_output': n_out+1,
             'intergration': not integration_failed}
        err_info = {
            'err_history': error_history.cpu().tolist(),  # Convert to list for compatibility
            'max_step_error': error_history.max().item() if error_idx > 0 else 0.0
        }
        return t_out, y_out, stats, err_info

class sde3_solver(sde_solver):
    def _after_process(self, y: torch.Tensor) -> torch.Tensor:
        return _vec_normaliza(y)

def  _vec_normaliza(y:torch.Tensor):
    if y.shape[0] == 0: 
        return y
    else:
        return torch.nn.functional.normalize(y.view(-1,3), dim=1,p=2).view_as(y)

def interp_fun(t_interp: torch.Tensor, t: torch.Tensor, y: torch.Tensor,h: torch.Tensor ,f_list: torch.Tensor, interp_coeff, is_dy: bool=True)-> torch.Tensor:
    """
    Interpolation function for Dormand-Prince method.
    """
    max_order = interp_coeff.shape[1] # interp_coeff.shape = [f_order, t_order]
    s = (t_interp - t) / h
    s = s.reshape(1,-1) #[1,t_l]
    s_list = torch.zeros((max_order, s.shape[1]), dtype=f_list.dtype, device=f_list.device) #[t_order,t_l]
    s_list[0] = s 
    if max_order > 1:
        for jj in range(max_order-1):
            s_list[jj+1] = s**(jj+2) # [t_order,t_l]
    s_coeff = interp_coeff @ s_list #[f_order, t_order] @ [t_order,t_l] = [f_order, t_l]
    dy = torch.einsum('s...,st->t...', f_list, s_coeff)  # [f_order,...] @ [f_order, t_l] = [t_l,...]
    if is_dy:
        y_interp = y.unsqueeze(0) + h * dy
    else:
        y_interp = y.unsqueeze(0) + dy  # [1,...] + [t_l,...] = [t_l,...]
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

def ode_sde_em(f: Callable, t_span: torch.Tensor, y0: torch.Tensor, options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict, Dict]:
    solver = 'EM'
    sf = sde_solver(f, t_span, y0, solver, options)
    bar = Wait_bar(t_span, sf.waitbar)
    t, y, stats, err_info = sf.run(bar)
    return t, y, stats, err_info

def ode3_sde_em(f: Callable, t_span: torch.Tensor, y0: torch.Tensor, options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict, Dict]:
    solver = 'EM'
    sf = sde3_solver(f, t_span, y0, solver, options)
    bar = Wait_bar(t_span, sf.waitbar)
    t, y, stats, err_info = sf.run(bar)
    return t, y, stats, err_info