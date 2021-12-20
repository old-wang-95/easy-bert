from setuptools import setup, find_packages

setup(
    name='easy-bert',
    version='0.2.0',
    author='waking95',
    author_email="wang0624@foxmail.com",
    description="easy-bert是一个中文NLP工具，提供诸多bert变体调用和调参方法，极速上手；清晰的设计和代码注释，也很适合学习",
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/waking95/easy-bert",
    project_urls={
        "Bug Tracker": "https://github.com/waking95/easy-bert/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(where='easy_bert'),
    install_requires=open('requirements.txt').read().strip().split('\n')
)
