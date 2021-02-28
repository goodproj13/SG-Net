from setuptools import setup, find_packages


setup(name='DenserNet',
      version='0.1.0',
      description='Open-source toolbox for Image-based Localization (Place Recognition)',
      author_email='liu2538@purdue.edu',
      url='https://github.com/goodproj13/DenserNet',
      license='MIT',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn'],
      packages=find_packages(),
      keywords=[
          'Image Localization',
          'Image Retrieval',
          'Place Recognition'
      ])
