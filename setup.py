import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ltron",
    version="0.0.6",
    author="Aaron Walsman",
    author_email="aaronwalsman@gmail.com",
    description='LEGO interactive machine learning environment.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aaronwalsman/ltron",
    install_requires = [
        'gym',
        'numpy',
        'scipy',
        'pyquaternion',
        'gdown',
        'tqdm',
        'splendor-render',
    ],
    packages=setuptools.find_packages(),
    #scripts=[],
    entry_points = {
        'console_scripts' : [
            'ltron_viewer=ltron.scripts.ltron_viewer:main',
            'ltron_asset_installer=ltron.scripts.ltron_asset_installer:main',
            'ltron_make_license=ltron.scripts.ltron_make_license:main',
        ]
    },
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ]
)
