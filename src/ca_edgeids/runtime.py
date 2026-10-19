"""Reproducible seeds, synchronized timings, and phase-local sampled RSS."""
import ctypes
import os
import platform
import random
import threading
import time
import numpy as np
import torch


def seed_all(seed,threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def current_rss():
    if os.name == "nt":
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in ["PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage","QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage"]]
        c = Counters()
        c.cb = ctypes.sizeof(c)
        kernel = ctypes.WinDLL("kernel32",use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi",use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(c),c.cb):
            raise OSError(ctypes.get_last_error())
        return int(c.WorkingSetSize)
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1])*os.sysconf("SC_PAGE_SIZE")
    except FileNotFoundError:
        raise RuntimeError("RSS measurement currently supports Windows and Linux")


class PeakRSS:
    def __enter__(self):
        self.baseline = current_rss()
        self.peak = self.baseline
        self.stop = threading.Event()
        def sample():
            while not self.stop.wait(0.005):
                self.peak = max(self.peak,current_rss())
        self.thread = threading.Thread(target=sample,daemon=True)
        self.thread.start()
        return self

    def __exit__(self,*args):
        self.peak = max(self.peak,current_rss())
        self.stop.set()
        self.thread.join()


def environment():
    import sklearn, scipy, pandas, matplotlib
    return {"python":platform.python_version(),"platform":platform.platform(),"processor":platform.processor(),
            "torch":torch.__version__,"numpy":np.__version__,"pandas":pandas.__version__,
            "sklearn":sklearn.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__,
            "cuda_available":torch.cuda.is_available(),"cuda":torch.version.cuda,
            "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
