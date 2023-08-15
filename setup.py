from setuptools import setup, find_packages
from setuptools.command.install import install


__version__ = "1.0.0"

with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()


class CustomInstall(install):
    def run(self):
        install.run(self)

setup(
    name='rlgym-ppo',
    packages=find_packages(),
    version=__version__,
    description='A multi-processed implementation of PPO for use with RLGym.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Matthew Allen',
    url='https://github.com/AechPro/rlgym-ppo',
    install_requires=[
        'cloudpickle==2.2.1',
        'filelock==3.12.2',
        'gym==0.26.2',
        'gym-notices==0.0.8',
        'Jinja2==3.1.2',
        'mpmath==1.3.0',
        'networkx==3.1',
        'numpy==1.25.2',
        'rlgym-sim @ git+https://github.com/AechPro/rocket-league-gym-sim@1aabb0449bdfeca7ba454dced1f6238c33706819',
        'RocketSim @ git+https://github.com/mtheall/RocketSim@5fee384d4ae1d88e8d025a5b00bb855e14465d8c',
        'sympy==1.12',
        'torch==2.0.1',
        'typing_extensions==4.7.1'
    ],
    python_requires='>=3.7',
    cmdclass={'install': CustomInstall},
    license='Apache 2.0',
    license_file='LICENSE',
    keywords=['rocket-league', 'gym', 'reinforcement-learning', 'simulation', 'ppo', 'rlgym', 'rocketsim']
)