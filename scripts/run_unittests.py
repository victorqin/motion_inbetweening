import os
import sys
import unittest


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
package_root = os.path.join(project_root, "packages")
sys.path.append(package_root)


if __name__ == "__main__":
    search_dir = os.path.join(project_root, "tests")
    suite = unittest.defaultTestLoader.discover(search_dir)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
