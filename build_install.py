import os

os.system("python setup.py sdist bdist_wheel")
os.system("pip install -U dist/UCTB-0.3.5-py3-none-any.whl")