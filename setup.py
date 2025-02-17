from setuptools import setup, find_packages

setup(
    name="RMNet",
    version="1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "RMNet = RMNet.detect:main", 
        ],
    },
    install_requires=[],  
)