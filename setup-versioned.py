from setuptools import setup, find_packages
import json
import os
import glob

with open('metadata.json', 'r') as f:
    metadata = json.load(f)
    name_versioned = f'{metadata["name"]}_{metadata["version"].replace(".","")}'

    os.symlink(metadata['name'], name_versioned)
    for cachefile in glob.glob(f'{metadata["name"]}/**/*.pyc', recursive=True):
        os.remove(cachefile)

    setup(
        name=name_versioned,
        version=metadata['version'],
        author=metadata['author'],
        author_email=metadata['author_email'],
        url=metadata['url'],
        description=metadata['description'],
        install_requires=metadata['install_requires'],
        packages=find_packages(exclude=[metadata["name"], f'{metadata["name"]}.*'])
    )

    os.remove(name_versioned)
