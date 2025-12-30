from tqdm import tqdm
import time
import sys

class Wait_bar:
    def __init__(self, t_span, flag):
        self.t0=0
        if flag:
            h=0
            self.t0 = t_span[0].item()
            self.t_final = t_span[-1].item()
            self.total_progress = self.t_final - self.t0
            
            # Check if running in a TTY (terminal)
            is_tty = sys.stdout.isatty()
            # General purpose setting:
            # TTY: 0.2s for smooth animation.
            # Non-TTY (Slurm/Log): 5.0s. 
            #   - Short task (20s): ~4-5 updates (sufficient heartbeat).
            #   - Long task (24h): ~1.7MB log (very manageable).
            self.update_interval = 0.2 if is_tty else 5.0
            
            self.pbar = tqdm(total=self.total_progress, desc='ODE Integration', 
                    unit='time', ncols=100, bar_format='{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}{postfix}]')
            self.pbar.set_postfix(dt=format(h,'.2e'))
            self.last_update_time = time.time()
    def update(self, t, h, flag, done):
        if flag:
            current_time = time.time()
            if current_time - self.last_update_time >= self.update_interval or done:
                progress = t.item() - self.t0
                self.pbar.n = min(progress,self.total_progress)
                self.pbar.set_postfix(dt=format(h,'.2e'))
                self.pbar.refresh()
                self.last_update_time = current_time
    def close(self,flag):
        if flag: 
            self.pbar.close()