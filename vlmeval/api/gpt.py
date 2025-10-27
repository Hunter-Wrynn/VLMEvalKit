from ..smp import *
import os
import sys
from .base import BaseAPI

APIBASES = {
    'OFFICIAL': 'https://api.openai.com/v1/chat/completions',
}


def GPT_context_window(model):
    length_map = {
        'gpt-4': 8192,
        'gpt-4-0613': 8192,
        'gpt-4-turbo-preview': 128000,
        'gpt-4-1106-preview': 128000,
        'gpt-4-0125-preview': 128000,
        'gpt-4-vision-preview': 128000,
        'gpt-4-turbo': 128000,
        'gpt-4-turbo-2024-04-09': 128000,
        'gpt-3.5-turbo': 16385,
        'gpt-3.5-turbo-0125': 16385,
        'gpt-3.5-turbo-1106': 16385,
        'gpt-3.5-turbo-instruct': 4096,
    }
    if model in length_map:
        return length_map[model]
    else:
        return 128000


class OpenAIWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'gpt-3.5-turbo-0613',
                 retry: int = 5,
                 key: str = None,
                 verbose: bool = False,
                 system_prompt: str = None,
                 temperature: float = 0,
                 timeout: int = 300,
                 api_base: str = None,
                 max_tokens: int = 2048,
                 img_size: int = -1,
                 img_detail: str = 'low',
                 use_azure: bool = False,
                 **kwargs):

        self.model = model
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_azure = use_azure

        if 'step' in model:
            env_key = os.environ.get('STEPAI_API_KEY', '')
            if key is None:
                key = env_key
        elif 'yi-vision' in model:
            env_key = os.environ.get('YI_API_KEY', '')
            if key is None:
                key = env_key
        elif 'internvl2-pro' in model:
            env_key = os.environ.get('InternVL2_PRO_KEY', '')
            if key is None:
                key = env_key
        elif 'abab' in model:
            env_key = os.environ.get('MiniMax_API_KEY', '')
            if key is None:
                key = env_key
        elif 'moonshot' in model:
            env_key = os.environ.get('MOONSHOT_API_KEY', '')
            if key is None:
                key = env_key
        elif 'grok' in model:
            env_key = os.environ.get('XAI_API_KEY', '')
            if key is None:
                key = env_key
        elif 'gemini' in model and 'preview' in model:
            # Will only handle preview models
            env_key = os.environ.get('GOOGLE_API_KEY', '')
            if key is None:
                key = env_key
            api_base = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        elif 'ernie' in model:
            env_key = os.environ.get('BAIDU_API_KEY', '')
            if key is None:
                key = env_key
            api_base = 'https://qianfan.baidubce.com/v2/chat/completions'
            self.baidu_appid = os.environ.get('BAIDU_APP_ID', None)
        else:
            if use_azure:
                env_key = os.environ.get('AZURE_OPENAI_API_KEY', None)
                assert env_key is not None, 'Please set the environment variable AZURE_OPENAI_API_KEY. '

                if key is None:
                    key = env_key
                assert isinstance(key, str), (
                    'Please set the environment variable AZURE_OPENAI_API_KEY to your openai key. '
                )
            else:
                env_key = os.environ.get('OPENAI_API_KEY', '')
                if key is None:
                    key = env_key
                assert isinstance(key, str) and key.startswith('sk-'), (
                    f'Illegal openai_key {key}. '
                    'Please set the environment variable OPENAI_API_KEY to your openai key. '
                )

        self.key = key
        assert img_size > 0 or img_size == -1
        self.img_size = img_size
        assert img_detail in ['high', 'low']
        self.img_detail = img_detail
        self.timeout = timeout
        self.is_max_completion_tokens = ('o1' in model) or ('o3' in model) or ('o4' in model) or ('gpt-5' in model)
        self.is_o_model = ('o1' in model) or ('o3' in model) or ('o4' in model)
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        if use_azure:
            api_base_template = (
                '{endpoint}openai/deployments/{deployment_name}/chat/completions?api-version={api_version}'
            )
            endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', None)
            assert endpoint is not None, 'Please set the environment variable AZURE_OPENAI_ENDPOINT. '
            deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', None)
            assert deployment_name is not None, 'Please set the environment variable AZURE_OPENAI_DEPLOYMENT_NAME. '
            api_version = os.getenv('OPENAI_API_VERSION', None)
            assert api_version is not None, 'Please set the environment variable OPENAI_API_VERSION. '

            self.api_base = api_base_template.format(
                endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
                api_version=os.getenv('OPENAI_API_VERSION')
            )
        else:
            if api_base is None:
                if 'OPENAI_API_BASE' in os.environ and os.environ['OPENAI_API_BASE'] != '':
                    self.logger.info('Environment variable OPENAI_API_BASE is set. Will use it as api_base. ')
                    api_base = os.environ['OPENAI_API_BASE']
                else:
                    api_base = 'OFFICIAL'

            assert api_base is not None

            if api_base in APIBASES:
                self.api_base = APIBASES[api_base]
            elif api_base.startswith('http'):
                self.api_base = api_base
            else:
                self.logger.error('Unknown API Base. ')
                raise NotImplementedError
            if os.environ.get('BOYUE', None):
                self.api_base = os.environ.get('BOYUE_API_BASE')
                self.key = os.environ.get('BOYUE_API_KEY')

        self.logger.info(f'Using API Base: {self.api_base}; API Key: {self.key}')

    # inputs can be a lvl-2 nested list: [content1, content2, content3, ...]
    # content can be a string or a list of image & text
    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    from PIL import Image
                    img = Image.open(msg['value'])
                    b64 = encode_image_to_base64(img, target_size=self.img_size)
                    img_struct = dict(url=f'data:image/jpeg;base64,{b64}', detail=self.img_detail)
                    content_list.append(dict(type='image_url', image_url=img_struct))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(type='text', text=text)]
        return content_list

    def prepare_inputs(self, inputs):
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                input_msgs.append(dict(role=item['role'], content=self.prepare_itlist(item['content'])))
        else:
            input_msgs.append(dict(role='user', content=self.prepare_itlist(inputs)))
        return input_msgs

    def generate_inner(self, inputs, **kwargs) -> str:
        input_msgs = self.prepare_inputs(inputs)
        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)

        # Will send request if use Azure, dk how to use openai client for it
        if self.use_azure:
            headers = {'Content-Type': 'application/json', 'api-key': self.key}
        elif 'internvl2-pro' in self.model:
            headers = {'Content-Type': 'application/json', 'Authorization': self.key}
        else:
            headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
        if hasattr(self, 'baidu_appid'):
            headers['appid'] = self.baidu_appid

        payload = dict(
            model=self.model,
            messages=input_msgs,
            n=1,
            temperature=temperature,
            **kwargs)

        if self.is_max_completion_tokens:
            payload['max_completion_tokens'] = max_tokens
            payload.pop('temperature')
        else:
            payload['max_tokens'] = max_tokens

        if 'gemini' in self.model:
            payload.pop('max_tokens')
            payload.pop('n')
            payload['reasoning_effort'] = 'high'

        response = requests.post(
            self.api_base,
            headers=headers, data=json.dumps(payload), timeout=self.timeout * 1.1)
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        usage_info = {}
        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct['choices'][0]['message']['content'].strip()
            
            # 提取usage信息
            if 'usage' in resp_struct:
                usage_info = resp_struct['usage']
                if self.verbose:
                    self.logger.info(f'OpenAI usage info: {usage_info}')
                    # 详细打印token统计信息
                    self.print_token_stats(usage_info)
                
                # 提取简化的token统计信息
                simplified_tokens = self.extract_simplified_tokens(usage_info)
                usage_info = simplified_tokens
            
        except Exception as err:
            if self.verbose:
                self.logger.error(f'{type(err)}: {err}')
                self.logger.error(response.text if hasattr(response, 'text') else response)

        return ret_code, answer, response, usage_info

    def print_token_stats(self, usage_info):
        """打印详细的token统计信息"""
        print("\n" + "="*60)
        print(f"🔢 GPT-5 Token使用统计 ({self.model})")
        print("="*60)
        
        # 基本token统计
        print(f"📝 输入Token数量: {usage_info.get('prompt_tokens', 0)}")
        print(f"📤 输出Token数量: {usage_info.get('completion_tokens', 0)}")
        print(f"📊 总Token数量: {usage_info.get('total_tokens', 0)}")
        
        # 输入token详情
        if 'prompt_tokens_details' in usage_info:
            prompt_details = usage_info['prompt_tokens_details']
            print(f"\n📥 输入Token详情:")
            print(f"  • 缓存Token: {prompt_details.get('cached_tokens', 0)}")
            print(f"  • 音频Token: {prompt_details.get('audio_tokens', 0)}")
        
        # 输出token详情
        if 'completion_tokens_details' in usage_info:
            completion_details = usage_info['completion_tokens_details']
            print(f"\n📤 输出Token详情:")
            print(f"  • 推理Token: {completion_details.get('reasoning_tokens', 0)}")
            print(f"  • 音频Token: {completion_details.get('audio_tokens', 0)}")
            print(f"  • 接受预测Token: {completion_details.get('accepted_prediction_tokens', 0)}")
            print(f"  • 拒绝预测Token: {completion_details.get('rejected_prediction_tokens', 0)}")
        
        print("="*60)

    def extract_simplified_tokens(self, usage_info):
        """提取简化的token统计信息，只包含输入和输出token"""
        simplified = {
            'prompt_token_count': usage_info.get('prompt_tokens', 0),
            'completion_token_count': usage_info.get('completion_tokens', 0),
            'total_token_count': usage_info.get('total_tokens', 0)
        }
        
        # 提取推理token数量（如果存在）
        if 'completion_tokens_details' in usage_info:
            completion_details = usage_info['completion_tokens_details']
            simplified['reasoning_token_count'] = completion_details.get('reasoning_tokens', 0)
        else:
            simplified['reasoning_token_count'] = 0
            
        return simplified

    def get_image_token_len(self, img_path, detail='low'):
        import math
        if detail == 'low':
            return 85

        im = Image.open(img_path)
        height, width = im.size
        if width > 1024 or height > 1024:
            if width > height:
                height = int(height * 1024 / width)
                width = 1024
            else:
                width = int(width * 1024 / height)
                height = 1024

        h = math.ceil(height / 512)
        w = math.ceil(width / 512)
        total = 85 + 170 * h * w
        return total

    def get_token_len(self, inputs) -> int:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(self.model)
        except Exception as err:
            if 'gpt' in self.model.lower():
                if self.verbose:
                    self.logger.warning(f'{type(err)}: {err}')
                enc = tiktoken.encoding_for_model('gpt-4')
            else:
                return 0
        assert isinstance(inputs, list)
        tot = 0
        for item in inputs:
            if 'role' in item:
                tot += self.get_token_len(item['content'])
            elif item['type'] == 'text':
                tot += len(enc.encode(item['value']))
            elif item['type'] == 'image':
                tot += self.get_image_token_len(item['value'], detail=self.img_detail)
        return tot


class GPT4V(OpenAIWrapper):

    def generate(self, message, dataset=None):
        return super(GPT4V, self).generate(message)
