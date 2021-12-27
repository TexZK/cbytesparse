import os

# working directory = repository root folder
path = os.path.join('src', 'cbytesparse', '_c.pyx')
os.system(r'cythonize -f -i ' + path)
