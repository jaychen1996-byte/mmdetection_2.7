from mmcv.utils import Config

flag_1 = False  # Config
flag_2 = False  # ProgressBar
flag_3 = True  # Timer
flag_4 = False

if flag_1:
    # Config
    """
    Config class is used for manipulating config and config files. 
    It supports loading configs from multiple file formats including python, json and yaml.
     It provides dict-like apis to get and set values.
    """
    cfg = Config.fromfile('asset/test.py')
    print(cfg)

    # predefined variables are supported.
    """
    对于所有格式配置，都支持一些预定义的变量。它将使用其实际值转换变量。{{ var }}
    当前，它支持四个预定义变量：
    {{ fileDirname }} -当前打开的文件的目录名，例如/ home / your-username / your-project / folder
    {{ fileBasename }} -当前打开的文件的基本名称，例如file.ext
    {{ fileBasenameNoExtension }} -当前打开的文件的基本名称，没有文件扩展名，例如file
    {{ fileExtname }} -当前打开的文件的扩展名，例如.ext
    """
    cfg = Config.fromfile('asset/config_a.py')  # 详见config_a.py的内容
    print(cfg)

    # inheritance is supported.
    """
    For all format configs, inheritance is supported. 
    To reuse fields in other config files, specify _base_='./config_a.py' or a list of configs _base_=['./config_a.py', './config_b.py']. 
    Here are 4 examples of config inheritance.
    """
    cfg = Config.fromfile('asset/config_2.py')
    print(cfg)

    """
    Inherit from base config with overlaped keys.
    """
    cfg = Config.fromfile('asset/config_3.py')
    print(cfg)  # b.b2=None in config_1 is replaced with b.b2=1 in config_3.py.

    """
    Inherit from base config with ignored fields.
    You may also set _delete_=True to ignore some fields in base configs. 
    All old keys b1, b2, b3 in b are replaced with new keys b2, b3.
    """
    cfg = Config.fromfile('asset/config_4.py')
    print(cfg)

    """
    Inherit from multiple base configs (the base configs should not contain the same keys).
    """
    cfg = Config.fromfile("asset/config_6.py")
    print(cfg)

if flag_2:
    """
    If you want to apply a method to a list of items and track the progress, track_progress is a good choice. 
    It will display a progress bar to tell the progress and ETA.
    """

    """模版
    import mmcv
    
    def func(item):
        # do something
        pass
    
    tasks = [item_1, item_2, ..., item_n]
    
    mmcv.track_progress(func, tasks)
    """
    import mmcv
    import time


    def plus_one(n):
        time.sleep(0.5)
        return n + 1


    tasks = list(range(10))

    # result = mmcv.track_progress(plus_one, tasks)
    # print(result)

    """
    There is another method track_parallel_progress, which wraps multiprocessing and progress visualization.
    """

    # result = mmcv.track_parallel_progress(plus_one, tasks, 4)  # 8 workers
    # print(result)

    """
    If you want to iterate or enumerate a list of items and track the progress, track_iter_progress is a good choice. 
    It will display a progress bar to tell the progress and ETA.
    """

    for task in mmcv.track_iter_progress(tasks):
        # do something like print
        print(task)

    # for i, task in enumerate(mmcv.track_iter_progress(tasks)):
    # do something like print
    # print(i)
    # print(task)

if flag_3:
    # 计时
    """
    It is convinient to compute the runtime of a code block with Timer.
    """
    import time
    import mmcv

    with mmcv.Timer():
        # simulate some code block
        # time.sleep(1)
        for _ in range(1000):
            a = 1
            b = 2
            c = a + b
            d = c ^ 9
            e = int(d / 888)

    """
    or try with since_start() and since_last_check(). 
    This former can return the runtime since the timer starts and the latter will return the time since the last time checked.
    """
    timer = mmcv.Timer()
    # code block 1 here
    time.sleep(2)
    print(timer.since_start())  # 从timer被创建到现在过去多少时间
    # code block 2 here
    time.sleep(1)
    print(timer.since_last_check())  # 从上一次被统计的时间过去多久
    time.sleep(1)
    print(timer.since_start())
    time.sleep(1)
    print(timer.since_last_check())
