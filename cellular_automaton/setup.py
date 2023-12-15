# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(
    name="cellular_automaton",
    version="1.0.8",
    author="Richard Feistenauer",
    author_email="r.feistenauer@web.de",
    packages=find_packages(exclude=('tests', 'docs', 'examples')),
    url="https://gitlab.com/DamKoVosh/cellular_automaton",
    license="Apache License 2.0",
    description="N dimensional cellular automaton.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    requires=["Python (>3.6.1)", "recordclass"]
)
