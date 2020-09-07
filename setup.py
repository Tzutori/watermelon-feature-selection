import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="watermelon-feature-selection",
    version="0.0.3",
    author="Xiang Xie",
    author_email="xiang.xie.china@gmail.com",
    description="A python package for watermelon feature selection method",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tzutori/watermelon-feature-selection",
    packages=setuptools.find_packages(),
    install_requires=['pandas>=1.0.4','numpy>=1.18.5','scipy>=1.4.1','scikit-learn>=0.23.1'],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: Microsoft :: Windows :: Windows 10",
    ],
    python_requires='>=3.7',
)