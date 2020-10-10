import setuptools
from os import path

file_dir = path.abspath(path.dirname(__file__))

with open(path.join(file_dir, 'README.md'), "r") as f:
	long_description = f.read()


version = dict()
with open(path.join(file_dir, 'BiModNeuroCNN/version.py'), "r") as (version_file):
	exec(version_file.read(), version)


setuptools.setup(

	name = "BiModNeuroCNN",
	version = version['__version__'],

	description = "Tools for bimodal training of CNNs, i.e. concurrent training with two data types",
	long_description = long_description,
	long_description_content_type = "text/markdown",

	url = "https://github.com/cfcooney",

	author = "Ciaran Cooney",

	license='MIT License',

	install_requires=['braindecode==0.4.85', 'mne', 'numpy',
	                  'pandas', 'scipy', 'matplotlib',],

	packages = setuptools.find_packages(),

	classifiers = [
	
		"Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        'Topic :: Software Development :: Build Tools',
        "Topic :: Scientific/Engineering :: Artificial Intelligence",

		'Programming Language :: Python :: 3.6',
	]

	)