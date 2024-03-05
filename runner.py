import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain_applications.models.models import LLMRunner  # noqa: E402


class Runner:
    def __init__(self, args):
        # Initialize Large Language Model
        self.init_llm_cfg(args)
        self.llm_runner = LLMRunner.load(self.llmname, self.chain_mode, self.agent_mode)

        # Initialize Summary Configs
        self.init_summary_cfg(args)

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

    def init_llm_cfg(self, args):
        self.llmname = args.llmname
        self.chain_mode = args.chain_mode
        self.agent_mode = args.agent_mode

    def init_summary_cfg(self, args):
        self.file = args.file
        self.infolder = args.infolder
        self.outfolder = args.outfolder
        self.name = '.'.join(self.file.split('.')[:-1])
        self.summary_file = os.path.join(self.outfolder, f'{self.name}-{self.llmname}-{self.chain_mode}_summary.md')

    def chatbot(self):
        print('Todo ...')
        return
        if self.llm_runner is not None:
            text = ''
            while text != 'exit':
                text = input("USER > ")
                if text != 'exit':
                    # [summarizer]
                    result = self.llm_runner.run(text, text)
                    print(result.replace(text, ''))

    def summary(self, file_path, save_file_name=''):
        if self.llm_runner is not None:
            # load
            text, input_info = '', None
            if '.txt' in file_path:
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
            else:
                input_info = file_path

            # summary
            summary = self.llm_runner.run(text, text, input_info=input_info)
            Runner.save_file(summary, save_file_name=save_file_name, write='w')
            print(summary)

    def run(self, mode=''):
        # APP
        if mode == 'chat':
            # Chat
            self.chatbot()
        elif mode == 'summary':
            # Summary
            self.summary(os.path.join(self.infolder, self.file), save_file_name=self.summary_file)
        else:
            print('Unknown mode: %s' % mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Large Language Model
    parser.add_argument('--llmname', type=str, default='Gemini', help=LLMRunner.LLM_CHOICES)
    parser.add_argument('--chain_mode', type=str, default='rag_map_reduce-stuff', help=LLMRunner.CHAIN_MODE_CHOICES)
    parser.add_argument('--agent_mode', type=str, default='', help=LLMRunner.AGENT_MODE_CHOICE)
    # Runner
    parser.add_argument('--run_mode', type=str, default='summary', help="'chat' or 'summary'")
    parser.add_argument('--file', type=str, default='RKD.pdf', help="kd_sharing-1h.txt, RKD.pdf")
    parser.add_argument('--infolder', type=str, default=os.path.join(os.path.dirname(__file__), 'examples'), help="")
    parser.add_argument('--outfolder', type=str, default=os.path.join(os.path.dirname(__file__), 'examples'), help="")
    args = parser.parse_args()

    # main
    runner = Runner(args)
    runner.run(mode=args.run_mode)
