import torch
import math
from magbox.Wait_bar import Wait_bar
import warnings

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
            self.order=5
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
    def _interp_fun(t_interp: torch.Tensor, t: torch.Tensor, y: torch.Tensor,h: torch.Tensor ,f_list: torch.Tensor, interp_coeff, is_dy: bool=True)-> torch.Tensor:
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
    
    @staticmethod
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
    @staticmethod
    def _step_after_nofailed(h_min,h_abs,rtol,err,order):
        return torch.max(h_min, h_abs * max(0.1, 0.8 * (rtol / err) ** (1/order)))
    
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
                        h_abs =self._step_after_nofailed(h_min,h_abs,rtol,err,order)
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
            h_abs = self._optimal_step_size(h_abs, err/rtol, order, failed)
        
            t=t_new
            y=y_new
            y=self._after_process(y)
            f1 = f_list_buffer[order+1]

        bar.close(waitbar)

        t_out= t_out[:n_out+1]
        y_out = y_out[:n_out+1,...]
        stats = {'n_calls': n_calls,
            'n_steps': n_steps,
            'n_output': n_out+1,
            'integration': not integration_failed}
        err_info = {
            'err_history': err_history,
            'max_step_error': max(err_history) if err_history else 0.0
        }
        return t_out, y_out, stats, err_info
