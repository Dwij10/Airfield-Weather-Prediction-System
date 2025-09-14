from setuptools import setup, find_packages

setup(
    name="airfield-weather-predictor",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'tensorflow',
        'dash',
        'plotly',
        'netCDF4',
        'xarray',
        'metpy',
        'boto3',
        'requests',
        'pytz',
        'schedule',
        'opencv-python',
        'joblib',
        'apscheduler',
        'dash-bootstrap-components'
    ]
)
