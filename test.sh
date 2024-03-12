# [Upper bound] will use Gemini to be llm
# Run summary with rag
python langchain_applications/runner.py --chain_mode 'rag_map_reduce-stuff' --infolder examples --file RKD.pdf --run_mode summary
# Run chat
python langchain_applications/runner.py --memory_mode 'default'
# Run chat with rag
python langchain_applications/runner.py --memory_mode 'default' --chain_mode 'rag_map_reduce-stuff' --infolder examples --file RKD.pdf --run_mode chat_summary


# [Zephyr]
# Run summary with rag
python langchain_applications/runner.py --chain_mode 'rag_map_reduce-stuff' --infolder examples --file RKD.pdf --llmname Zephyr --run_mode summary
# Run chat
python langchain_applications/runner.py --memory_mode 'default' --llmname Zephyr
# Run chat with rag
python langchain_applications/runner.py --memory_mode 'default' --chain_mode 'rag_map_reduce-stuff' --infolder examples --file RKD.pdf --llmname Zephyr --run_mode chat_summary


# [Check Name]
python langchain_applications/runner.py --llmname Gemini --run_mode name --check_prompt examples/name_template.txt --infolder examples --file name.json --outfolder examples --output_json name.json
python langchain_applications/runner.py --llmname Gemini --run_mode name --check_prompt examples/name_template.txt --infolder examples --file name.json --outfolder examples --output_json name.json --restart  # restart inference whole json
