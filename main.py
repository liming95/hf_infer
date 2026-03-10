from engine.core import test_engine, HFEngine
from engine.sgl_ref import SGLangEngine
from engine.tool import check_output, token_match_rate, debug_token_diff, compare_tokens

def compare_engines():
    model_path = "/workspace/models/Qwen2.5-1.5B-Instruct"
    engine = HFEngine(model_path)
    prompt = "What is the capital of France?"
    response = engine.generate(prompt)
    
    sglang_engine = SGLangEngine(model_path)
    sgl_response = sglang_engine.generate(prompt)
    print('\n')
    print('-----------------------------')
    print(f'Response: {response}')
    print('\n')
    print(f'SGLang Response: {sgl_response}')
    print('-----------------------------')
    # Compare the two responses
    check_result = check_output(response, sgl_response)
    print(f'Comparison Result: {check_result}')
    token_match = token_match_rate(engine.tokenizer, response, sgl_response)
    print(f'Token Match Rate: {token_match:.2%}')
    first_diff = compare_tokens(engine.tokenizer, response, sgl_response)
    if not first_diff["match"]:
        print(f'First difference at token position {first_diff["diff_position"]}:')
        print(f'Token in response: {first_diff.get("token1", "N/A")}')
        print(f'Token in sgl_response: {first_diff.get("token2", "N/A")}')
    else:
        print("The two responses are identical in tokens.")
    debug_token_diff(engine.tokenizer, response, sgl_response)

if __name__ == "__main__":
    # test_engine()
    # test_sglang_engine()
    compare_engines()