from setuptools import setup
import json

with open('metadata.json', 'r') as f:
    metadata = json.load(f)
    name_versioned = f'{metadata["name"]}_{metadata["version"].replace(".","")}'

    setup(
        name=name_versioned,
        version=metadata['version'],
        author=metadata['author'],
        author_email=metadata['author_email'],
        url=metadata['url'],
        description=metadata['description'],
        install_requires=metadata['install_requires'],
        packages=[name_versioned],
        package_dir={name_versioned: metadata['name']}
    )