import os
import sys
import time
import traceback

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open("probe_out.txt", "w") as _f:
    _f.write("START\n")
    _f.flush()
    try:
        from brain import Brain

        _f.write("Brain imported\n")
        _f.flush()
        b = Brain()
        _f.write("Brain() OK\n")
        _f.flush()
        b.start_headless()
        _f.write("start_headless() OK\n")
        _f.flush()
        time.sleep(0.5)
        b.stop()
        _f.write("stop() OK\n")
        _f.flush()
    except BaseException as e:
        _f.write(f"CRASH: {type(e).__name__}: {e}\n")
        traceback.print_exc(file=_f)
        _f.flush()
        sys.exit(1)
