import torch

from magbox.Wait_bar import Wait_bar
from magbox.solver.sde3_solver import sde3_solver


class sde3_fix_solver(sde3_solver):
    """Fixed-step SDE solver for 3D spin dynamics.

    Uses the same EM stepping kernel as `sde_solver`, but advances with a fixed step `dh`.
    `dh` is read from `options['dh']` and defaults to `(t_span[1]-t_span[0]) / 10` when possible.
    """

    def _ode_options(self, options):
        super()._ode_options(options)
        if options is None:
            options = {}
        dh = options.get('dh', None)
        if dh is None:
            self.dh = None
        else:
            self.dh = torch.as_tensor(dh, device=self.device, dtype=self.dtype)
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

        ## combined noise term for error estimate
        ## not used in fixed step mode
        # gw_list[2] = g1(W[0] + W[1])
        return t_new, y_list, f_list, gw_list

    def run(self, bar: Wait_bar):
        (
            t_out,
            y_out,
            t,
            y,
            dtype,
            device,
            h_max,
            h_min,
            rtol,
            atol,
            max_failures,
            t_final,
            t0,
            t_dir,
            t_span,
            waitbar,
            alpha,
            beta,
            c_error,
            sde_fcn,
            S,
            chunk,
            output_pos,
            n_t_span,
            n_eq,
            refine,
            order,
            interp_coeff,
        ) = self._get_parameters()

        # Fixed step size (magnitude), clamped to [h_min, h_max]
        if self.dh is None:
            if isinstance(t_span, torch.Tensor) and t_span.numel() >= 2:
                dt = torch.abs(t_span[1] - t_span[0])
            else:
                dt = torch.abs(t_final - t0)
            dh_abs = dt / 10
        else:
            dh_abs = torch.abs(self.dh)
        if dh_abs.numel() != 1:
            dh_abs = dh_abs.reshape(())
        dh_abs = torch.min(h_max, torch.max(h_min, dh_abs))
        dh_abs = torch.as_tensor(dh_abs, dtype=dtype, device=device)

        # Outputs
        n_out = 0
        t_out[n_out] = t
        y_out[n_out, ...] = self._after_process(y)

        # Output pointer for t_span output
        next_idx = 1

        # Initial drift/diffusion
        f1, g1, noise_dim = sde_fcn(t, y)
        n_calls = 1
        n_steps = 0

        # Pre-allocate buffers for EM stages
        # y_list: (order+1, ...)  f_list: (order, ...)  gw_list: (order+1, ...)
        y_list = torch.empty((order + 1,) + y.shape, dtype=dtype, device=device)
        f_list = torch.empty((order,) + y.shape, dtype=dtype, device=device)
        gw_list = torch.empty((order + 1,) + y.shape, dtype=dtype, device=device)

        err_history: list[float] = []

        finished = False
        while not finished:
            if dh_abs > torch.abs(t_final-t):
                dh_abs = torch.abs(t_final-t)
                finished = True

            h = t_dir * dh_abs

            # One fixed EM step
            W = torch.randn((order,) + noise_dim, dtype=dtype, device=device)
            t_new, y_list, f_list, gw_list = self._one_step(
                f1, g1, t, y, h, W, sde_fcn, order, y_list, f_list, gw_list
            )
            n_calls += 1
            y_new = y_list[order]

            finished = bool(t_dir * (t_final - t_new) <= 0)
            bar.update(t_new, h, waitbar, finished)

            # outputs
            nout_new, t_out_new, y_out_new, next_idx = self._get_outputs(output_pos,self._interp_fun, t, S, refine, t_new, y_new, y, h, y_list, interp_coeff, dtype, device, n_t_span, t_span, next_idx,t_dir)
            y_out_new = self._after_process(y_out_new)

            if nout_new > 0:
                old_nout = n_out
                n_out += nout_new

                if n_out + 1 > t_out.shape[0]:
                    extra = max(chunk, nout_new)
                    tout_new_temp = torch.zeros(t_out.shape[0] + extra, dtype=dtype, device=device)
                    tout_new_temp[: t_out.shape[0]] = t_out
                    t_out = tout_new_temp

                    yout_new_temp = torch.zeros((y_out.shape[0] + extra,) + n_eq, dtype=dtype, device=device)
                    yout_new_temp[: y_out.shape[0], ...] = y_out
                    y_out = yout_new_temp

                t_out[old_nout + 1 : n_out + 1] = t_out_new
                y_out[old_nout + 1 : n_out + 1, ...] = y_out_new

            # Advance
            n_steps += 1
            t = t_new
            y = self._after_process(y_new)

            f1, g1, _ = sde_fcn(t, y)
            n_calls += 1

        bar.close(waitbar)

        if output_pos == 1:
            if n_t_span > 0:
                final_len = max(1, min(next_idx, n_t_span))
                t_out = t_out[:final_len]
                y_out = y_out[:final_len, ...]
        else:
            t_out = t_out[: n_out + 1]
            y_out = y_out[: n_out + 1, ...]

        stats = {
            'n_calls': n_calls,
            'n_steps': n_steps,
            'n_output': y_out.shape[0],
            'integration': True,
            'fix_step': True,
            'dh': float(dh_abs.item()),
        }
        err_info = {
            'err_history': err_history,
            'max_step_error': 0.0,
        }
        return t_out, y_out, stats, err_info
