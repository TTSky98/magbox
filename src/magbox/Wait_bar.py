from tqdm import tqdm
import time

class Wait_bar:
    def __init__(self, t_span, flag):
        self.t0=0
        if flag:
            self.t0 = t_span[0].item()
            self.t_final = t_span[-1].item()
            self.total_progress = self.t_final - self.t0
            self.pbar = tqdm(total=self.total_progress, desc='ODE Integration', 
                    unit='time', ncols=100, bar_format='{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}]')
            self.last_update_time = time.time()
            self.update_interval = 0.1  # Update progress bar every 0.1 seconds
    def update(self, t, flag, done):
        if flag:
            current_time = time.time()
            if current_time - self.last_update_time >= self.update_interval or done:
                progress = t.item() - self.t0
                self.pbar.n = min(progress,self.total_progress)
                self.pbar.refresh()
                self.last_update_time = current_time
    def close(self,flag):
        if flag: 
            self.pbar.close()