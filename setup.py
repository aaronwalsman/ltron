import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="brick-gym",
    version="0.0.1",
    install_requires = ['gym'],
    author="Aaron Walsman",
    author_email="aaronwalsman@gmail.com",
    description="Machine learning environment for LDraw bricks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aaronwalsman/brick-gym",
    packages=setuptools.find_packages(),
    scripts=[
            "bin/brick_viewer",
            "bin/make_dataset_metadata",
            "bin/train_semantic_segmentation",
            "bin/train_graph",
            "bin/train_graph_b"]
)
