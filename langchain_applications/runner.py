import os
import argparse

from tqdm import tqdm
from langchain_applications.models.models import LLMRunner


class Runner:
    def __init__(self, args):
        # Initialize Configs
        self.init_cfg(args)

        # Large Language Model
        self.llm_runner = LLMRunner.load(
            self.llmname, input_info=self.input_info, memory_mode=self.memory_mode, chain_mode=self.chain_mode,
            agent_mode=self.agent_mode)

    @staticmethod
    def save_file(text, save_file_name='', write='w'):
        # save
        if save_file_name:

            # build folder
            if os.path.dirname(save_file_name):
                os.makedirs(os.path.dirname(save_file_name), exist_ok=True)

            # save summary
            with open(save_file_name, write, encoding="utf-8") as file:
                file.write(text)
            return True
        else:
            return False

    def init_cfg(self, args):
        # Variables
        self.llmname = args.llmname
        self.memory_mode = args.memory_mode
        self.chain_mode = args.chain_mode
        self.agent_mode = args.agent_mode

        # File
        self.file = args.file
        self.infolder = args.infolder
        self.outfolder = args.outfolder
        self.name = '.'.join(self.file.split('.')[:-1])
        self.input_info = os.path.join(self.infolder, self.file)
        self.summary_file = os.path.join(self.outfolder, f'{self.name}-{self.llmname}-{self.chain_mode}_summary.md')

        # Check Name
        self.restart = args.restart
        self.output_json = args.output_json
        self.check_file = args.check_prompt
        self.input_key = args.input_key
        self.output_key = args.output_key
        self.output_json_file = os.path.join(self.outfolder, self.output_json)

    def chatbot(self):
        if self.llm_runner is not None:
            text = ''
            while text != 'exit':
                text = input("USER > ")
                if text != 'exit':
                    result = self.llm_runner.chat(text, text, use_history=True)
                    print(result.replace(text, ''))

    def summary(self, new_file_path=None, save_file_name=''):
        if self.llm_runner is not None:
            # summary
            summary = self.llm_runner.summary(new_file_path)
            Runner.save_file(summary, save_file_name=save_file_name, write='w')
            print(summary)

    def check_name(self):
        if os.path.exists(self.input_info):
            # Prompt
            with open(self.check_file, 'r') as f:
                prompt = f.read()

            # Read
            with open(self.input_info, 'r') as f:
                json_data = f.readlines()

            # Update
            import json
            for i, d in tqdm(enumerate(json_data), total=len(json_data)):
                try:
                    data = json.loads(d)
                    if self.input_key in data:
                        if self.restart or self.output_key not in data:
                            # pre-processing
                            print(data[self.input_key])
                            text = prompt.replace('{question}', data[self.input_key])

                            # inference
                            result = self.llm_runner.chat(text, text, use_history=True)

                            # post-processing
                            try:
                                result = result.rstrip(',')[:-2].rstrip(',') + result.rstrip(',')[-2:]  # json.loads(
                                result = result.replace('\n', '').replace('{', '').replace('}', '').replace(
                                   '"', '').replace("'", "").replace(' ', '').split(',')
                                result_json, key = {}, []
                                for r in result:
                                    if r:
                                        if ':' not in r:
                                            result_json[key] += str(',' + r)
                                        else:
                                            key = r.split(':')[0]
                                            result_json[key] = r.split(':')[1]

                                # final
                                data[self.output_key] = result_json
                                json_data[i] = str(data).replace("'", '"') + '\n'
                                print(data[self.output_key])

                                # Write
                                with open(self.output_json_file, 'w') as f:
                                    f.write(''.join(json_data))

                            except Exception as e:
                                print(e)  # import pdb; pdb.set_trace()
                                print('Unexcept output format %s' % result)
                        # else:
                        #     print('already infered %s' % d)
                except Exception as e:
                    print(e)
                    print('Unexcept input format %s' % d)

    def run(self, mode=''):
        # Name
        if 'name' in mode:
            self.check_name()

        # Summary: new_file_path=os.path.join(self.infolder, self.file),
        if 'summary' in mode:
            self.summary(new_file_path=None, save_file_name=self.summary_file)

        # Chat
        if 'chat' in mode:
            self.chatbot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Large Language Model
    parser.add_argument('--llmname', type=str, default='Gemini', help=LLMRunner.LLM_CHOICES)
    parser.add_argument('--memory_mode', type=str, default='', help=LLMRunner.MEMORY_MODE_CHOICES)
    parser.add_argument('--chain_mode', type=str, default='', help=LLMRunner.CHAIN_MODE_CHOICES)
    parser.add_argument('--agent_mode', type=str, default='', help=LLMRunner.AGENT_MODE_CHOICE)
    # Runner
    parser.add_argument('--run_mode', type=str, default='chat', help="'chat' or 'summary' or 'chat_summary")
    parser.add_argument('--file', type=str, default='', help="kd_sharing-1h.txt, RKD.pdf")
    parser.add_argument('--infolder', type=str, default='', help="input folder")
    parser.add_argument('--outfolder', type=str, default=os.path.join(os.path.dirname(__file__), 'examples'), help="")
    # Check Name
    parser.add_argument('-r', '--restart', action="store_true", help="restart to inference whole json")
    parser.add_argument('--check_prompt', type=str, default='examples/name_template.txt', help="")
    parser.add_argument('-ik', '--input_key', type=str, default='sentence', help="example: 'sentence'")
    parser.add_argument('-ok', '--output_key', type=str, default='analysis', help="example: 'analysis'")
    parser.add_argument('--output_json', type=str, default='name.json', help="")
    args = parser.parse_args()

    # main
    runner = Runner(args)
    runner.run(mode=args.run_mode)
