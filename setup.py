"""Uses setuptools to install the ignite_simple module"""
import setuptools
import os

setuptools.setup(
    name='ignite_simple',
    version='0.0.11',
    author='Timothy Moore',
    author_email='mtimothy984@gmail.com',
    description='Easily train pytorch models with automatic LR and BS tuning',
    license='CC0',
    keywords='torch pytorch models machine-learning learning-rate batch-size hyperparameters',
    url='https://github.com/tjstretchalot/ignite-simple',
    packages=['ignite_simple', 'ignite_simple.gen_sweep'],
    package_data={'ignite_simple': ['html/*']},
    long_description=open(
        os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    long_description_content_type='text/markdown',
    install_requires=['pca3dvis', 'torch', 'numpy', 'matplotlib', 'scipy',
                      'pyzmq', 'beautifulsoup4', 'pytorch-ignite', 'psutil',
                      'sortedcontainers'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication',
        'Topic :: Utilities'],
    python_requires='>=3.6',
)