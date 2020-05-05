from distutils.core import setup

with open('README.md', 'r') as fh
	long_description = fh.read()

setup(
  name = 'Kami',
  packages = ['Kami'],
  version = '0.1',
  license='MIT',
  description = 'Forecast sales with entity embedding LSTM neural network based on past POS record',
  long_description = long_description,
  long_description_content_type = 'text/markdown',
  author = 'Yifei Yu',
  author_email = 'yyu.mam2020@london.edu',
  url = 'https://github.com/MacarielAerial',
  download_url = 'https://github.com/MacarielAerial/AM18_SPR20_LondonLAB/archive/v_0.1.tar.gz',
  keywords = ['SALES', 'FORECAST', 'LSTM', 'EMBEDDING'],
  install_requires=[
          'numpy',
          'pandas',
          'matplotlib',
          'sklearn',
          'tensorflow'
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
