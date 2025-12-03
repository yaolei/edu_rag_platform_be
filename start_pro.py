# start_pro.py
import subprocess
import signal
import sys

def start_pro(name, port):
    cmd = [
        sys.executable, "-m", "uvicorn",
        f"{name}.app.main:app",
        "--reload", "--host", "0.0.0.0", "--port", str(port)
    ]
    return subprocess.Popen(cmd)

def stop(*_):
    # 只发终止信号，不 wait
    for p in procs:
        p.terminate()
    sys.exit(0)

if __name__ == '__main__':
    procs = [
        start_pro('gateway', 8000),
        start_pro('service_rag', 8001),
    ]
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    for p in procs:
        p.wait()