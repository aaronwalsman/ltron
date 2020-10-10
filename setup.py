import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="brick-gym",
    version="0.0.1",
    author="Aaron Walsman",
    author_email="aaronwalsman@gmail.com",
    description="Machine learning environment for Lego bricks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aaronwalsman/brick_gym",
    packages=setuptools.find_packages(),
    scripts=[]
)
