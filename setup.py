from setuptools import setup, find_packages
import json

with open('metadata.json', 'r') as f:
    metadata = json.load(f)

    setup(
        name=metadata['name'],
        version=metadata['version'],
        author=metadata['author'],
        author_email=metadata['author_email'],
        url=metadata['url'],
        description=metadata['description'],
        install_requires=metadata['install_requires'],
        packages=find_packages()
    )
