import os

try:
    import psutil
    physical = psutil.cpu_count(logical=False)  # physical cores
    logical = psutil.cpu_count(logical=True)    # logical threads
except ImportError:
    # fallback if psutil not installed
    physical = os.cpu_count() // 2 if os.cpu_count() else None
    logical = os.cpu_count()

print(f"CPU Physical cores: {physical}")
print(f"CPU Logical threads: {logical}")
