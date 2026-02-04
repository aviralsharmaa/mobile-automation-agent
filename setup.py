"""
Setup script for mobile automation agent
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mobile-automation-agent",
    version="0.1.0",
    author="Your Name",
    description="AI-powered mobile automation agent for blind users",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mobile-automation-agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "mobile-agent=src.agent.orchestrator:main",
        ],
    },
)
