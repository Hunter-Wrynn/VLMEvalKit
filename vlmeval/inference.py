import torch
import torch.distributed as dist
import time
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *

FAIL_MSG = 'Failed to obtain answer via API.'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(model, work_dir, model_name, dataset, index_set=None, api_nproc=4, ignore_failed=False):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)
    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    lt, indices = len(data), list(data['index'])

    structs = []
    for i in range(lt):
        item = data.iloc[i]
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)

    out_file = f'{work_dir}/{model_name}_{dataset_name}_supp.pkl'
    out_time_file = f'{work_dir}/{model_name}_{dataset_name}_supp_TIME.pkl'

    # To reuse records in MMBench_V11
    if dataset_name in ['MMBench', 'MMBench_CN']:
        pred_format = get_pred_file_format()
        v11_pred = f'{work_dir}/{model_name}_{dataset_name}_V11.{pred_format}'
        if osp.exists(v11_pred):
            try:
                reuse_inds = load('http://opencompass.openxlab.space/utils/mmb_reuse.pkl')
                data = load(v11_pred)
                ans_map = {x: y for x, y in zip(data['index'], data['prediction']) if x in reuse_inds}
                dump(ans_map, out_file)
            except Exception as err:
                print(type(err), err)

    res = {}
    time_res = {}
    token_stats_res = {}
    if osp.exists(out_file):
        res = load(out_file)
        if ignore_failed:
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}
    if osp.exists(out_time_file):
        time_res = load(out_time_file)
    
    # 加载token统计文件
    out_token_file = f'{work_dir}/{model_name}_{dataset_name}_supp_TOKEN.pkl'
    if osp.exists(out_token_file):
        token_stats_res = load(out_token_file)

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    # 包装 generate 函数以记录时间和token统计
    def gen_func_with_time(**kwargs):
        start_time = time.time()
        response = model.generate(**kwargs)
        end_time = time.time()
        inference_time = end_time - start_time
        
        # 获取token统计信息
        token_stats = getattr(model, 'last_token_stats', {})
        
        return {
            'response': response, 
            'time': inference_time,
            'token_stats': token_stats
        }

    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        results = track_progress_rich(gen_func_with_time, structs, nproc=api_nproc, chunksize=api_nproc, save=None, keys=None)
        
        # 分离响应、时间和token统计
        for idx, result in zip(indices, results):
            res[idx] = result['response']
            time_res[idx] = result['time']
            token_stats_res[idx] = result['token_stats']
        
        # 保存结果
        dump(res, out_file)
        dump(time_res, out_time_file)
        dump(token_stats_res, out_token_file)

    res = load(out_file)
    time_res = load(out_time_file) if osp.exists(out_time_file) else {}
    token_stats_res = load(out_token_file) if osp.exists(out_token_file) else {}
    
    if index_set is not None:
        res = {k: v for k, v in res.items() if k in index_set}
        time_res = {k: v for k, v in time_res.items() if k in index_set}
        token_stats_res = {k: v for k, v in token_stats_res.items() if k in index_set}
    
    os.remove(out_file)
    if osp.exists(out_time_file):
        os.remove(out_time_file)
    if osp.exists(out_token_file):
        os.remove(out_token_file)
    
    return res, time_res, token_stats_res


def infer_data(model, model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4, use_vllm=False):
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    prev_time_file = f'{work_dir}/{model_name}_{dataset_name}_PREV_TIME.pkl'
    prev_token_file = f'{work_dir}/{model_name}_{dataset_name}_PREV_TOKEN.pkl'
    
    res = load(prev_file) if osp.exists(prev_file) else {}
    time_dict = load(prev_time_file) if osp.exists(prev_time_file) else {}
    token_dict = load(prev_token_file) if osp.exists(prev_token_file) else {}
    
    if osp.exists(out_file):
        res.update(load(out_file))
    
    # 加载对应的时间文件和token统计文件
    out_time_file = out_file.replace('.pkl', '_TIME.pkl')
    out_token_file = out_file.replace('.pkl', '_TOKEN.pkl')
    if osp.exists(out_time_file):
        time_dict.update(load(out_time_file))
    if osp.exists(out_token_file):
        token_dict.update(load(out_token_file))

    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        # 保存时间信息
        time_dict_filtered = {k: time_dict.get(k, 0.0) for k in data_indices}
        dump(time_dict_filtered, out_time_file)
        return model

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)

    kwargs = {}
    if model_name is not None and (
        'Llama-4' in model_name
        or 'Qwen2-VL' in model_name
        or 'Qwen2.5-VL' in model_name
    ):
        kwargs = {'use_vllm': use_vllm}

    # (25.06.05) In newer version of transformers (after 4.50), with device_map='auto' and torchrun launcher,
    # Transformers automatically adopt TP parallelism, which leads to compatibility problems with VLMEvalKit
    # (In VLMEvalKit, we use torchrun to launch multiple model instances on a single node).
    # To bypass this problem, we unset `WORLD_SIZE` before building the model to not use TP parallel.
    ws_bak = os.environ.pop('WORLD_SIZE', None)
    model = supported_VLM[model_name](**kwargs) if isinstance(model, str) else model
    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak

    is_api = getattr(model, 'is_api', False)
    if is_api:
        lt, indices = len(data), list(data['index'])
        supp, supp_time, supp_token = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            index_set=set(indices),
            api_nproc=api_nproc)
        for idx in indices:
            assert idx in supp
        res.update(supp)
        time_dict.update(supp_time)
        token_dict.update(supp_token)
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        # 保存 API 模型的推理时间和token统计
        time_dict_filtered = {k: time_dict.get(k, 0.0) for k in data_indices}
        token_dict_filtered = {k: token_dict.get(k, {}) for k in data_indices}
        dump(time_dict_filtered, out_time_file)
        dump(token_dict_filtered, out_token_file)
        return model
    else:
        model.set_dump_image(dataset.dump_image)

    for i in tqdm(range(lt), desc=f'Infer {model_name}/{dataset_name}, Rank {rank}/{world_size}'):
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(data.iloc[i])

        # 记录推理开始时间
        start_time = time.time()
        
        # If `SKIP_ERR` flag is set, the model will skip the generation if error is encountered
        if os.environ.get('SKIP_ERR', False) == '1':
            FAIL_MSG = 'Failed to obtain answer'
            try:
                response = model.generate(message=struct, dataset=dataset_name)
            except RuntimeError as err:
                torch.cuda.synchronize()
                warnings.warn(f'{type(err)} {str(err)}')
                response = f'{FAIL_MSG}: {type(err)} {str(err)}'
        else:
            response = model.generate(message=struct, dataset=dataset_name)
        
        # 记录推理结束时间
        end_time = time.time()
        inference_time = end_time - start_time
        
        # 获取token统计信息
        token_stats = getattr(model, 'last_token_stats', {})
        
        torch.cuda.empty_cache()

        if verbose:
            print(f'{response} (Time: {inference_time:.2f}s)', flush=True)

        res[idx] = response
        time_dict[idx] = inference_time
        token_dict[idx] = token_stats
        
        if (i + 1) % 10 == 0:
            dump(res, out_file)
            dump(time_dict, out_time_file)
            dump(token_dict, out_token_file)

    res = {k: res[k] for k in data_indices}
    time_dict_filtered = {k: time_dict.get(k, 0.0) for k in data_indices}
    token_dict_filtered = {k: token_dict.get(k, {}) for k in data_indices}
    dump(res, out_file)
    dump(time_dict_filtered, out_time_file)
    dump(token_dict_filtered, out_token_file)
    return model


# A wrapper for infer_data, do the pre & post processing
def infer_data_job(
    model, work_dir, model_name, dataset, verbose=False, api_nproc=4, ignore_failed=False, use_vllm=False
):
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name
    # 使用环境变量控制的文件格式
    result_file = get_pred_file_path(work_dir, model_name, dataset_name, use_env_format=True)

    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    prev_time_file = f'{work_dir}/{model_name}_{dataset_name}_PREV_TIME.pkl'
    prev_token_file = f'{work_dir}/{model_name}_{dataset_name}_PREV_TOKEN.pkl'
    
    if osp.exists(result_file):
        if rank == 0:
            data = load(result_file)
            # breakpoint()
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
            if not ignore_failed:
                results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}
            dump(results, prev_file)
            
            # 如果结果文件中有时间信息，也保存到 PREV_TIME
            if 'inference_time' in data:
                time_results = {k: v for k, v in zip(data['index'], data['inference_time'])}
                dump(time_results, prev_time_file)
            
            # 如果结果文件中有token统计信息，也保存到 PREV_TOKEN
            if 'token_stats' in data:
                token_results = {k: v for k, v in zip(data['index'], data['token_stats'])}
                dump(token_results, prev_token_file)
        if world_size > 1:
            dist.barrier()

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}.pkl')
    out_file = tmpl.format(rank)

    model = infer_data(
        model=model, work_dir=work_dir, model_name=model_name, dataset=dataset,
        out_file=out_file, verbose=verbose, api_nproc=api_nproc, use_vllm=use_vllm)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        time_all = {}
        token_all = {}
        
        # 合并所有进程的预测结果
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))
        
        # 合并所有进程的时间数据
        time_tmpl = tmpl.replace('.pkl', '_TIME.pkl')
        for i in range(world_size):
            time_file = time_tmpl.format(i)
            if osp.exists(time_file):
                time_all.update(load(time_file))
        
        # 合并所有进程的token统计数据
        token_tmpl = tmpl.replace('.pkl', '_TOKEN.pkl')
        for i in range(world_size):
            token_file = token_tmpl.format(i)
            if osp.exists(token_file):
                token_all.update(load(token_file))

        data = dataset.data
        for x in data['index']:
            assert x in data_all
        if os.getenv('SPLIT_THINK', False):
            prediction = [str(data_all[x]) for x in data['index']]

            def split_thinking(s):
                if '</think>' in s:
                    splits = s.split('</think>')
                    prediction = splits[-1].strip()
                    if len(splits) == 2 and '<think>' in splits[0]:
                        thinking = splits[0].split('<think>')[1].strip()
                    else:
                        thinking = '</think>'.join(splits[:-1])
                        thinking += '</think>'
                        warnings.warn('Failed to parse thinking, multiple </think> tags or missing <think> tag.')
                else:
                    thinking = ''
                    prediction = s
                return (prediction, thinking)
            split_func = model.split_thinking if hasattr(model, 'split_thinking') else split_thinking
            print(f'Prediction format: {os.getenv("SPLIT_THINK")},splitting func: {split_func}')
            tups = [split_func(x) for x in prediction]
            data['prediction'] = [x[0] for x in tups]
            data['thinking'] = [x[1] for x in tups]
        else:
            data['prediction'] = [str(data_all[x]) for x in data['index']]
        
        # 添加推理时间列
        data['inference_time'] = [time_all.get(x, 0.0) for x in data['index']]
        
        # 添加token统计列
        data['prompt_token_count'] = [token_all.get(x, {}).get('prompt_token_count', 0) for x in data['index']]
        data['candidates_token_count'] = [token_all.get(x, {}).get('candidates_token_count', 0) for x in data['index']]
        data['completion_token_count'] = [token_all.get(x, {}).get('completion_token_count', 0) for x in data['index']]
        data['total_token_count'] = [token_all.get(x, {}).get('total_token_count', 0) for x in data['index']]
        data['text_prompt_token_count'] = [token_all.get(x, {}).get('text_prompt_token_count', 0) for x in data['index']]
        data['image_prompt_token_count'] = [token_all.get(x, {}).get('image_prompt_token_count', 0) for x in data['index']]
        data['thoughts_token_count'] = [token_all.get(x, {}).get('thoughts_token_count', 0) for x in data['index']]
        data['reasoning_token_count'] = [token_all.get(x, {}).get('reasoning_token_count', 0) for x in data['index']]
        
        if 'image' in data:
            data.pop('image')

        dump(data, result_file)
        
        # 清理临时文件
        for i in range(world_size):
            os.remove(tmpl.format(i))
            time_file = time_tmpl.format(i)
            if osp.exists(time_file):
                os.remove(time_file)
            token_file = token_tmpl.format(i)
            if osp.exists(token_file):
                os.remove(token_file)
    if world_size > 1:
        dist.barrier()
    return model
