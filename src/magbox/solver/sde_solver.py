from magbox.Wait_bar import Wait_bar
import torch
import math
import warnings
from .eq_solver import eq_solver

class sde_solver(eq_solver):
    def _tableau(self, solver_name):
        device=self.device
        dtype=self.dtype
        if solver_name == 'EM':
            self.order=2
            self.adaptive_order = 1.5
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
    def _one_step(f1, g1, t, y, h, W, sde_fcn, order, y_list,f_list,gw_list):
        # EM method one step
        h_abs = torch.abs(h)
        # Drift uses signed h, diffusion uses |h|.
        half_h = h * 0.5
        sqrt_half_abs_h = torch.sqrt(h_abs * 0.5)

        # stage 1
        y_list[0] = y
        f_list[0] = f1
        gw11 = g1(W[0])
        gw_list[0] = gw11

        y2 = y + f1 * half_h + gw11 * sqrt_half_abs_h
        y_list[1] = y2
        t2 = t + half_h
        f2, g2, _ = sde_fcn(t2, y2)
        f_list[1] = f2

        # stage 2
        gw22 = g2(W[1])
        gw_list[1] = gw22
        y_new = y2 + f2 * half_h + gw22 * sqrt_half_abs_h
        y_list[2] = y_new
        t_new = t + h

        # combined noise term for error estimate
        gw_list[2] = g1(W[0] + W[1])
        return t_new, y_list, f_list, gw_list
    @staticmethod
    def _get_err(f_list, gw_list, c_error, h, y, y_new, atol):
        err = h * torch.einsum('s...,si->...', f_list, c_error[0])
        err += torch.sqrt(h) * torch.einsum('s...,si->...', gw_list, c_error[1])
        err /= torch.max(torch.max(y.abs(), y_new.abs()), atol)
        err = torch.max(err.abs())
        return err
    
    @staticmethod
    def _interp_fun(t_interp, t, y, h, y_list, interp_coeff):
        y_interp = eq_solver._interp_fun(t_interp, t, y, h, y_list, interp_coeff,is_dy=False)
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

        error_history = []
        n_calls = 0
        n_steps = 0
        
        h = torch.min(h_max, torch.max(h_min, 0.1 * torch.abs(t_final - t0)))
        h_abs = torch.abs(h)

        f1, g1, noise_dim = sde_fcn(t, y)
        n_calls += 1

        # Pre-allocate buffers for EM stages to reduce per-step allocations.
        # y_list: (order+1, ...)  f_list: (order, ...)  gw_list: (order+1, ...)
        y_list = torch.empty((order + 1,) + y.shape, dtype=dtype, device=device)
        f_list = torch.empty((order,) + y.shape, dtype=dtype, device=device)
        gw_list = torch.empty((order + 1,) + y.shape, dtype=dtype, device=device)

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
                t_new, y_list, f_list, gw_list = self._one_step(f1, g1, t, y, h, W, sde_fcn, order, y_list, f_list, gw_list)
                
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
                    error_history.append(err)
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
                    extra = max(chunk, n_out_new)
                    t_out_new_temp = torch.zeros(t_out.shape[0] + extra, dtype=dtype, device=device)
                    t_out_new_temp[:t_out.shape[0]] = t_out
                    t_out = t_out_new_temp
                    
                    y_out_new_temp = torch.zeros((y_out.shape[0] + extra,)+n_eq, dtype=dtype, device=device)
                    y_out_new_temp[:y_out.shape[0],...] = y_out
                    y_out = y_out_new_temp
                
                t_out[old_n_out+1:n_out+1] = t_out_new
                y_out[old_n_out+1:n_out+1,...] = y_out_new

            h_abs = self._optimal_step_size(h_abs, err/rtol, adaptive_order, no_failed)
            
            t = t_new
            y = y_new

            y = self._after_process(y) 

            f1, g1, _ = sde_fcn(t, y)
            n_calls += 1

        bar.close(waitbar)
        
        t_out = t_out[:n_out+1]
        y_out = y_out[:n_out+1, ...]

        stats = {'n_calls': n_calls,
             'n_steps': n_steps,
             'n_output': n_out+1,
             'intergration': not integration_failed}
        err_info = {
            'err_history': error_history,
            'max_step_error': max(error_history) if error_history else 0.0
        }
        return t_out, y_out, stats, err_info