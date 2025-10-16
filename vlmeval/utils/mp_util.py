from multiprocessing import Pool
import os
from typing import Callable, Iterable, Sized

from rich.progress import (BarColumn, MofNCompleteColumn, Progress, Task,
                           TaskProgressColumn, TextColumn, TimeRemainingColumn)
from rich.text import Text
import os.path as osp
import time
import portalocker
from ..smp import load, dump


def track_progress_rich(
        func: Callable,
        tasks: Iterable = tuple(),
        nproc: int = 1,
        save=None,
        keys=None,
        save_time=None,
        **kwargs) -> list:

    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    if save is not None:
        assert osp.exists(osp.dirname(save)) or osp.dirname(save) == ''
        if not osp.exists(save):
            dump({}, save)
    if save_time is not None:
        assert osp.exists(osp.dirname(save_time)) or osp.dirname(save_time) == ''
        if not osp.exists(save_time):
            dump({}, save_time)
    if keys is not None:
        assert len(keys) == len(tasks)
    if not callable(func):
        raise TypeError('func must be a callable object')
    if not isinstance(tasks, Iterable):
        raise TypeError(
            f'tasks must be an iterable object, but got {type(tasks)}')
    assert nproc > 0, 'nproc must be a positive number'
    res = load(save) if save is not None else {}
    time_res = load(save_time) if save_time is not None else {}
    results = [None for _ in range(len(tasks))]
    
    # 记录每个任务的开始时间
    task_start_times = {}

    with ThreadPoolExecutor(max_workers=nproc) as executor:
        futures = []

        for inputs in tasks:
            if not isinstance(inputs, (tuple, list, dict)):
                inputs = (inputs, )
            if isinstance(inputs, dict):
                future = executor.submit(func, **inputs)
            else:
                future = executor.submit(func, *inputs)
            futures.append(future)
            # 记录提交时间作为开始时间
            task_start_times[len(futures) - 1] = time.time()

        unfinished = set(range(len(tasks)))
        pbar = tqdm(total=len(unfinished))
        while len(unfinished):
            new_finished = set()
            for idx in unfinished:
                if futures[idx].done():
                    # 计算执行时间
                    end_time = time.time()
                    inference_time = end_time - task_start_times[idx]
                    
                    results[idx] = futures[idx].result()
                    new_finished.add(idx)
                    if keys is not None:
                        res[keys[idx]] = results[idx]
                        # 保存时间信息
                        if save_time is not None:
                            time_res[keys[idx]] = inference_time
            if len(new_finished):
                if save is not None:
                    dump(res, save)
                if save_time is not None:
                    dump(time_res, save_time)
                pbar.update(len(new_finished))
                for k in new_finished:
                    unfinished.remove(k)
            time.sleep(0.1)
        pbar.close()

    if save is not None:
        dump(res, save)
    if save_time is not None:
        dump(time_res, save_time)
    return results
