import setuptools

with open("README.md", mode="r", encoding="utf-8") as fh:
    long_description = fh.read()

REQUIRED_PACKAGES = ["flair>=0.13.0", "torch>=2.5.0", "country_converter>=1.0.0"]

setuptools.setup(
    name="name2nat",  # should propably rename.
    version="1.0.0",
    author="Kyubyong Park / Jimmy Engelbrecht",
    author_email="jimmy@plero.se",
    description="Nationality Prediction from Name",
    install_requires=REQUIRED_PACKAGES,
    license="Apache License 2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jimmy927/name2nat",
    packages=setuptools.find_packages(),
    package_data={"name2nat": ["name2nat/best-model.pt", "name2nat/fix_path.py"]},
    python_requires=">=3.6",
    include_package_data=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
