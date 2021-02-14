import sys,os
import cx_Freeze
import sklearn
base = None

if sys.platform == 'win32':
    base = 'Win32GUI'

# buildOptions = dict(packages = ["osgeo._gdal","os","numpy","six","matplotlib"], excludes = [], \
#     include_files = [(r"C:\Users\User\.conda\envs\maptorenv\Lib\site-packages\mpl_toolkits","data")])

excludes = ['multiprocessing.Pool']
packages = ["sklearn","os","pandas","numpy","six","matplotlib","multiprocessing","reportlab","logging","sklearn.cross_decomposition","sklearn.metrics",
            'sklearn.model_selection','joblib','sklearn.tree._criterion','operator','traceback',#'sklearn.utils.sparsetools._graph_validation',
            #'sklearn.utils.sparsetools._graph_tools',
            #'sklearn.utils.lgamma',
            'sklearn.utils.weight_vector','sklearn.utils.fixes',
            'sklearn.utils.extmath','sklearn.metrics.ranking','sklearn.neighbors','sklearn.neighbors.typedefs','sklearn.neighbors.quad_tree'
            ]


executables = [cx_Freeze.Executable("MainWindow.py")]

includefiles = [r"C:\Users\User\.conda\envs\maptorenv\Lib\site-packages", r"D:\Maptor-xx\maptor\venv\UI\proj",
                r"D:\Maptor-xx\maptor\venv\UI\Images"]
               #  r"flob.png",r"Mapto.png",r"unirostock.png",
               #  "rsz_1rsz_1europaeischer-sozialfonds-vektor-farbig-rgb.png","rsz_1wetscapes_logo.png","rsz_11rsz_euro_fonds_quer.png",
               # ]


cx_Freeze.setup(
        name = "mainwindow",
        version = "0.1",
        description = "Contact <myworkemail> with questions",
        options = {"build_exe" :{"packages":packages,"excludes":excludes,"include_files":includefiles}},
        #option = buildOptions,
        executables = executables
    )