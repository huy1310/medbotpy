from setuptools import setup, find_packages

setup(
    name="medbot_api",  # Name of the package
    version="0.1.0",  # Initial version number
    description="A simple example package",  # A brief description of the package
    author="Huy",  # Replace with your name
    author_email="your.email@example.com",  # Replace with your email
    url="https://github.com/yourusername/abc",  # Replace with your repository URL if applicable
    packages=find_packages(),  # Automatically find all packages and sub-packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Specify Python version compatibility
    install_requires=[
        # Add your dependencies here
        # Example: 'numpy>=1.21.0'
    ],
    entry_points={
        'console_scripts': [
            # Example for creating command-line tools
            # 'abc-cli=abc.cli:main',
        ],
    },
)
