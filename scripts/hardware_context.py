"""Non-identifying Linux hardware context for the processing-study manifest."""
import os
from pathlib import Path
import platform
import re


def hardware_context():
    cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),platform.machine())
    memory={line.split(':',1)[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','SwapTotal:'))}
    return dict(cpu_model=cpu,logical_cpus=os.cpu_count(),linux_memory_kib=memory['MemTotal'],
        linux_visible_memory_gib=memory['MemTotal']/1024**2,linux_swap_kib=memory['SwapTotal'],
        windows_mounted_workspace=bool(re.match(r'^/mnt/[a-z]/',Path.cwd().as_posix())),
        thread_environment={name:os.environ.get(name) for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},
        scope='Linux-visible VM/process context; not whole-host available memory or a deployment capacity bound')
