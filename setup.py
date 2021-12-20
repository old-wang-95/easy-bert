from setuptools import setup, find_packages

setup(
    name='easy-zh-bert',
    version='0.4.0',
    author='waking95',
    author_email="wang0624@foxmail.com",
    license='MIT',
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
    packages=find_packages(),
    install_requires=[
        'torch == 1.4.0',
        'transformers == 3.1.0',
        'scikit-learn == 0.24.0',
        'numpy == 1.17.0',
        'datasets == 1.16.0',
        'textbrewer == 0.2.0',
        # 'onnxruntime==1.4.0',
        'onnxruntime-gpu == 1.4.0',
    ],
)
