import os
import argparse

from langchain_applications.models.models import LLMRunner


class Runner:
    def __init__(self, args):
        # Initialize Configs
        self.init_cfg(args)

        # Large Language Model
        self.llm_runner = LLMRunner.load(
            self.llmname, self.input_info, self.memory_mode, self.chain_mode, self.agent_mode)

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
            summary = self.llm_runner.summary()
            Runner.save_file(summary, save_file_name=save_file_name, write='w')
            print(summary)

    def run(self, mode=''):
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
    args = parser.parse_args()

    # main
    runner = Runner(args)
    runner.run(mode=args.run_mode)
