from setuptools import setup, find_packages

setup(
    name="shard_llms",
    version="0.2.13",
    author="Yzm0034",
    author_email="yash.mahajan50@gmail.com",
    description="Shard Large Language Models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yzm1205/Shard-Any-LLMs",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "transformers",
        "torch",
    ],
    entry_points={
        "console_scripts": [
            "shard_model=shard_llms.sharding_model:main",
        ],
    },
)
