import faulthandler
import threading
import time
import sys

faulthandler.enable()

def dump():
    time.sleep(2)
    faulthandler.dump_traceback(sys.stderr)
    sys.exit(1)

threading.Thread(target=dump).start()

print("importing sklearn")
import sklearn.feature_extraction.text
print("done")
