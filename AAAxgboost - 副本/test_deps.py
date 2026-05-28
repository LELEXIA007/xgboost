import sys
import os

# 添加本地包目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'local_packages', 'Lib', 'site-packages'))

print("Testing dependencies...")

try:
    import numpy
    print(f"numpy: {numpy.__version__}")
except ImportError as e:
    print(f"numpy import error: {e}")

try:
    import pandas
    print(f"pandas: {pandas.__version__}")
except ImportError as e:
    print(f"pandas import error: {e}")

try:
    import xgboost
    print(f"xgboost: {xgboost.__version__}")
except ImportError as e:
    print(f"xgboost import error: {e}")

try:
    import shap
    print(f"shap: {shap.__version__}")
except ImportError as e:
    print(f"shap import error: {e}")

try:
    import matplotlib
    print(f"matplotlib: {matplotlib.__version__}")
except ImportError as e:
    print(f"matplotlib import error: {e}")

try:
    import seaborn
    print(f"seaborn: {seaborn.__version__}")
except ImportError as e:
    print(f"seaborn import error: {e}")

try:
    import sklearn
    print(f"scikit-learn: {sklearn.__version__}")
except ImportError as e:
    print(f"scikit-learn import error: {e}")

print("Dependency test completed.")