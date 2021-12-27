import os

# working directory = repository root folder
os.environ['PYTHONPATH'] = os.path.abspath('src')
path = os.path.join('tests', '_test_c.pyx')
os.system(r'cythonize -f -i ' + path)
