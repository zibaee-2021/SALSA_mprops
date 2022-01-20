import os

"""
Programmatically resolves absolute path of SALSA_mprops root directory.  
"""
abspath_root = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))