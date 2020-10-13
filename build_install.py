import os

os.system("python setup.py sdist bdist_wheel")
os.system("pip install -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com"
          " -U dist/UCTB-0.3.0-py3-none-any.whl")