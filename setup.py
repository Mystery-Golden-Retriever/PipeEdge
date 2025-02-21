from setuptools import setup, find_packages

setup(
    name="pipeedge",
    version="0.1.1",
    description="PipeEdge",
    author="Yang Hu, Connor Imes, Haonan Wang",
    author_email="yhu210003@usc.edu, cimes@isi.edu, haonanwa@usc.edu",
    readme="README.md",
    python_requires=">=3.7",
    license="LICENSE",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "apphb>=0.1.0",
        "energymon>=0.1.0",
        "networkx>=2.6",
        "numpy>=1.15.0",
        "PyYAML",
        "requests",
        "scipy",
        "timm>=0.3.2",
        "torch>=1.8.0",
        "transformers>=4.6.0",
    ],
    extras_require={
        "runtime": [
            "datasets>=2.0.0",
            "Pillow",
            "psutil",
            "torchvision>=0.3.0",
        ],
    },
)
