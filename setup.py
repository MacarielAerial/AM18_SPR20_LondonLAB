from distutils.core import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.txt'), 'r', encoding = 'utf-8') as fh:
	long_description = fh.read()

setup(
  name = 'Kami',
  packages = ['Kami'],
  version = '0.3.9',
  license='MIT',
  description = 'Forecast sales with Entity Embedding LSTM',
  long_description = long_description,
  author = 'Yifei Yu',
  author_email = 'yyu.mam2020@london.edu',
  url = 'https://github.com/MacarielAerial',
  download_url = 'https://github.com/MacarielAerial/AM18_SPR20_LondonLAB/archive/V_0.3.9.tar.gz',
  keywords = ['SALES', 'FORECAST', 'LSTM', 'EMBEDDING'],
  install_requires=[
          'numpy',
          'pandas',
          'matplotlib',
          'sklearn',
          'tensorflow',
          'pydot',
          'graphviz'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
  ],
  python_requires = '>= 3.6'
)
