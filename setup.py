"""
KTB ML Framework - 통합 머신러닝 실험 프레임워크
"""

from setuptools import setup, find_packages
import os

# Read version from __init__.py
def get_version():
    init_file = os.path.join('research', '__init__.py')
    with open(init_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip("'\"")
    return '0.1.0'

# Read long description from README
def get_long_description():
    readme_file = 'README.md'
    if os.path.exists(readme_file):
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Core dependencies
INSTALL_REQUIRES = [
    'torch>=2.0.0',
    'torchvision>=0.15.0',
    'numpy>=1.24.0',
    'matplotlib>=3.7.0',
    'scikit-learn>=1.3.0',
    'pandas>=1.5.0',
    'tqdm>=4.65.0',
]

# Optional dependencies
EXTRAS_REQUIRE = {
    'wandb': [
        'wandb>=0.15.0',
    ],
    'dev': [
        'pytest>=7.0.0',
        'pytest-cov>=4.0.0',
        'black>=23.0.0',
        'flake8>=6.0.0',
        'mypy>=1.0.0',
        'isort>=5.12.0',
    ],
    'notebook': [
        'jupyter>=1.0.0',
        'ipykernel>=6.0.0',
        'ipywidgets>=8.0.0',
    ],
    'docs': [
        'sphinx>=5.0.0',
        'sphinx-rtd-theme>=1.2.0',
        'sphinx-autodoc-typehints>=1.23.0',
    ],
}

# Add 'all' extras that includes everything
EXTRAS_REQUIRE['all'] = sum(EXTRAS_REQUIRE.values(), [])

setup(
    # Basic metadata
    name='research',
    version=get_version(),
    author='KTB AI Research Team',
    author_email='ai-research@ktb.com',
    description='통합 머신러닝 실험 프레임워크 - 전이학습과 일반 ML 태스크를 위한 범용 프레임워크',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/ktb-ai/ktb-ml-framework',
    project_urls={
        'Bug Tracker': 'https://github.com/ktb-ai/ktb-ml-framework/issues',
        'Documentation': 'https://ktb-ml-framework.readthedocs.io',
        'Source Code': 'https://github.com/ktb-ai/ktb-ml-framework',
    },
    
    # Package configuration
    packages=find_packages(exclude=['tests', 'tests.*', 'examples', 'examples.*']),
    include_package_data=True,
    python_requires='>=3.8',
    
    # Dependencies
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    
    # Classification
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    
    # Keywords
    keywords=[
        'machine-learning',
        'deep-learning',
        'transfer-learning',
        'pytorch',
        'experiment-framework',
        'model-comparison',
        'computer-vision',
        'audio-processing',
    ],
    
    # License
    license='MIT',
    
    # Entry points (optional CLI commands)
    entry_points={
        'console_scripts': [
            'ktb-ml-info=research:print_info',
        ],
    },
    
    # Zip safe
    zip_safe=False,
)
