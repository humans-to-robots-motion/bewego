import os
import sys
driectory = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, driectory)
sys.path.insert(0, driectory + os.sep + "../pybewego")
import _pybewego as m
assert m.__version__ == '0.0.1'
assert m.add(1, 2) == 3
assert m.subtract(1, 2) == -1
assert m.test_identity(10) == True
assert m.test_identity(1) == True
print("All OK!")
