import keyext
from setuptools import setup, find_packages

setup(
    name='keyext',
    version='0.0.1',
    author='Junhyun Kim',
    author_email='me@junbread.win',
    url='https://github.com/2020swchallenge-moai/keyword-extraction',
    description='Keyword extraction based on TF-IDF',
    install_requires=['scikit-learn>=0.23', 'krwordrank', 'konlpy'],
    packages=find_packages()
)
