import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="llmonitor",
    version="0.0.1",
    author="llmonitor",
    author_email="hello@llmonitor.com",
    description="Open-source observability for AI apps.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://llmonitor.com",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'requests',  
    ],
    keywords='git tag git-tagup tagup tag-up version autotag auto-tag commit message',
    project_urls={
        'Homepage': 'https://github.com/llmonitor/llmonitor-py',
    },
)