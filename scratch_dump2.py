import faulthandler
import threading
import time
import sys

faulthandler.enable()

def dump():
    time.sleep(15)
    faulthandler.dump_traceback(sys.stderr)
    sys.exit(1)

threading.Thread(target=dump).start()

print("importing stage2_retrieval")
import stage2_retrieval
print("done")
