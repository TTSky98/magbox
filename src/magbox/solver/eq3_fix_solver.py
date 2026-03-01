import torch

from magbox.Wait_bar import Wait_bar
from magbox.solver.eq3_solver import eq3_solver


class eq3_fix_solver(eq3_solver):
    """Fixed-step RK solver for 3D spin ODEs.

    Notes
    -----
    - Uses the same Dormand–Prince (RK45) tableau as `eq_solver`, but advances with a fixed step `dh`.
    - `dh` is read from `options['dh']` and defaults to `(t_span[1]-t_span[0]) / 10` when possible.
    """

    def _ode_options(self, options):
        super()._ode_options(options)
        # NOTE: eq_solver.__init__ calls _ode_options() before _ode_initial(), so `t_span` is
        # not available here. We only record the user-provided `dh` and compute defaults later.
        if options is None:
            options = {}
        dh = options.get('dh', None)
        if dh is None:
            self.dh = None
        else:
            self.dh = torch.as_tensor(dh, device=self.device, dtype=self.dtype)

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
            ode_fcn,
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

        n_beta = len(beta)

        # Pre-allocate RK45 stage buffers (7 stages).
        stage_shape = (n_beta + 1,) + y.shape
        f_list_buffer = torch.zeros(stage_shape, dtype=dtype, device=device)
        y_list_buffer = torch.zeros(stage_shape, dtype=dtype, device=device)

        # Initial derivative
        f1 = ode_fcn(t, y)
        n_calls = 1

        n_steps = 0
        # Keep outputs consistent with the post-processing convention.
        n_out = 0
        t_out[n_out] = t
        y_out[n_out, ...] = self._after_process(y)

        # Output pointer for t_span mode
        next_idx = 1

        err_history: list[float] = []

        finished = False

        # Main loop
        while not finished:
            if dh_abs > torch.abs(t_final-t):
                dh_abs = torch.abs(t_final-t)
                finished = True

            h = t_dir * dh_abs

            # One fixed step
            y_list_buffer[0] = y
            f_list_buffer[0] = f1
            t_new, y_list_buffer, f_list_buffer = self._one_step(
                f_list_buffer, y_list_buffer, t, y, h, ode_fcn, alpha, beta
            )
            n_calls += n_beta

            y_new = y_list_buffer[n_beta]

            # Update progress bar
            finished = bool(t_dir * (t_final - t_new) <= 0)
            bar.update(t_new, h, waitbar, finished)

            # outputs
            nout_new, t_out_new, y_out_new, next_idx = self._get_outputs(output_pos,self._interp_fun, t, S, refine, t_new, y_new, y, h, f_list_buffer, interp_coeff, dtype, device, n_t_span, t_span, next_idx,t_dir)
            y_out_new = self._after_process(y_out_new)
            # Store output
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

            # Advance state
            n_steps += 1
            t = t_new
            y = self._after_process(y_new)

            # After post-processing (normalization), recompute derivative.
            f1 = f_list_buffer[order+1]

        bar.close(waitbar)

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
