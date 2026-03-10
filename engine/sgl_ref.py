import sglang as sgl

class SGLangEngine:
    def __init__(self, model_path):
        self.engine = sgl.Engine(model_path=model_path)
        self.max_new_tokens = 2048
        # self.gen_config = dict(
        #     #max_new_tokens=self.max_new_tokens,
        #     temperature=0,
        #     top_p=1,
        #     do_sample=False   
        # )
        # self.gen_config = {
        #     "temperature": 0,
        #     "top_p": 1,
        #     "max_new_tokens": self.max_new_tokens,
        #     "top_k": 1,
        #     "repetition_penalty": 1.5
        # }
        self.gen_config = {
            'top_p': 1.0, 
            'top_k': 100000000,
            'max_new_tokens': 2048,
            'temperature': 0,
            'stop_token_ids': [151643, 151645],
            'ignore_eos': False,
            'skip_special_tokens': True,
            'frequency_penalty': 1.0
        }

    def generate(self, prompt: str):
        # 直接调用 engine.generate()，传入生成参数
        result = self.engine.generate(prompt, self.gen_config)
        # 返回 answer，如果返回是 dict，则根据实际键修改
        return result.get("text", result)

    def trim_overlap(self,existing_text, new_chunk):
        """
        Finds the largest suffix of 'existing_text' that is a prefix of 'new_chunk'
        and removes that overlap from the start of 'new_chunk'.
        """
        max_overlap = 0
        max_possible = min(len(existing_text), len(new_chunk))
        for i in range(max_possible, 0, -1):
            if existing_text.endswith(new_chunk[:i]):
                max_overlap = i
                break
        return new_chunk[max_overlap:]

    def stream_generate(self, prompt: str):
        # 直接调用 engine.stream()
        final_text = ""
        for chunk in self.engine.generate(prompt, self.gen_config, stream=True):
            chunk_text = chunk["text"]
            cleaned_chunk = self.trim_overlap(final_text, chunk_text)
            print(cleaned_chunk, end='', flush=True)  # 实时输出增量文本
            final_text += cleaned_chunk
        return final_text
        

def test_sglang_engine():
    model_path = "/workspace/models/Qwen2.5-1.5B-Instruct"
    engine = SGLangEngine(model_path)
    prompt = "What is the capital of France?"
    response = engine.generate(prompt)

    streamed_response = engine.stream_generate(prompt)
    print('\n')
    print('-----------------------------')
    print(f'Response: {response}')
    print('\n')
    print(f'Streamed Response: {streamed_response}')
    print('-----------------------------')
    # # Compare the two responses
    # check_result = check_output(response, streamed_response)
    # print(f'Comparison Result: {check_result}')
    # token_match = token_match_rate(engine.engine.tokenizer, response, streamed_response)
    # print(f'Token Match Rate: {token_match:.2%}')
    # first_diff = compare_tokens(engine.engine.tokenizer, response, streamed_response)
    # if not first_diff["match"]:
    #     print(f'First difference at token position {first_diff["diff_position"]}:')
    #     print(f'Token in response: {first_diff.get("token1", "N/A")}')
    #     print(f'Token in streamed response: {first_diff.get("token2", "N/A")}')
    # else:
    #     print("The two responses are identical in tokens.")
    # debug_token_diff(engine.engine.tokenizer, response, streamed_response)

if __name__ == "__main__":
    test_sglang_engine()
