from root_path import abspath_root
import os
import pandas as pd


def read_xls_and_write_csv(xls_path: str):
    read_file = pd.read_excel(xls_path)
    csv_path = xls_path.replace('xls', 'csv')
    csv_path = os.path.join(csv_path)
    read_file.to_csv(csv_path, header=True, index=False)