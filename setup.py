from setuptools import setup, find_packages

setup(
    name="cl_califa_pipeline",
    version="0.1.0",
    description="A pipeline for processing and modeling data from the CL and CALIFA surveys",
    author="Ginés Martínez Solaeche",
    author_email="gimarso@iaa.es",
    url="https://github.com/gimarso/CALIFA_CL_pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tensorflow>=2.4.0",
        "numpy>=1.19.0",
        "astropy>=4.0.0",
        "matplotlib>=3.3.0",
        "scipy>=1.5.0",
        "scikit-image>=0.17.0",
        "joblib>=1.0.0",
        "scikit-learn>=0.24.0",
        "statsmodels>=0.12.0",
        "h5py>=2.10.0"
    ],
    entry_points={
        'console_scripts': [
            'train= train:main',
            'generate_galaxy_pairs= generate_galaxy_pairs:main',
            'create_latent_space= create_latent_space:main',
            'merge_trecords= merge_trecords:main',
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
