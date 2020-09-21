from setuptools import setup, find_packages

setup(
    name='keyext_100',
    version='1.0.0',
    author='Junhyun Kim',
    author_email='me@junbread.win',
    url='https://github.com/2020swchallenge-moai/keyword-extraction',
    description='Keyword extraction based on TF-IDF',
    install_requires=['scikit-learn>=0.23', 'krwordrank>=1.0', 'konlpy>=0.5'],
    packages=find_packages()
)
