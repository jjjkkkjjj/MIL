import sys, os
#os.chdir(os.path.abspath('__file__'))
sys.path.append(os.path.realpath('..'))
from MIL.simpleMIL import SimpleMIL

simpleMIL = SimpleMIL()
simpleMIL.fit()