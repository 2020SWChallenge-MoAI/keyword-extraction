from setuptools import setup
import json
import os

with open('metadata.json', 'r') as f:
    metadata = json.load(f)
    name_versioned = f'{metadata["name"]}_{metadata["version"].replace(".","")}'

    os.symlink(metadata['name'], name_versioned, True)

    setup(
        name=name_versioned,
        version=metadata['version'],
        author=metadata['author'],
        author_email=metadata['author_email'],
        url=metadata['url'],
        description=metadata['description'],
        install_requires=metadata['install_requires'],
        packages=[name_versioned]
    )

    os.remove(name_versioned)
