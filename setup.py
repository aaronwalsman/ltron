import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ltron",
    version="0.0.2",
    install_requires = ['gym', 'tqdm', 'numpy', 'pyquaternion'],
    author="Aaron Walsman",
    author_email="aaronwalsman@gmail.com",
    description="Lego interactive machine learning environment.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aaronwalsman/ltron",
    packages=setuptools.find_packages(),
    scripts=[
            "bin/ltron_viewer",
    ],
)
