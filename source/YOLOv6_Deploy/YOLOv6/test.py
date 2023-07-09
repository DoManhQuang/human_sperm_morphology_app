import os,sys
ROOT = os.getcwd()
print(ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

print(ROOT)