import os


PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets")
CONFIG_ROOT = os.path.join(PROJECT_ROOT, "config")
