from engine.core import HFEngine
from engine.tool import check_output, token_match_rate, debug_token_diff, compare_tokens

def test_engine():
    model_path = "/workspace/models/Qwen2.5-1.5B-Instruct"
    engine = HFEngine(model_path)
    prompt = "What is the capital of France?"
    response = engine.generate(prompt)
    streamed_response = engine.stream_generate(prompt)
    print('\n')
    print('-----------------------------')
    print(f'Response: {response}')
    print('\n')
    print(f'Streamed Response: {streamed_response}')
    print('-----------------------------')
    # Compare the two responses
    check_result = check_output(response, streamed_response)
    print(f'Comparison Result: {check_result}')
    token_match = token_match_rate(engine.tokenizer, response, streamed_response)
    print(f'Token Match Rate: {token_match:.2%}')
    first_diff = compare_tokens(engine.tokenizer, response, streamed_response)
    if not first_diff["match"]:
        print(f'First difference at token position {first_diff["diff_position"]}:')
        print(f'Token in response: {first_diff.get("token1", "N/A")}')
        print(f'Token in streamed response: {first_diff.get("token2", "N/A")}')
    else:
        print("The two responses are identical in tokens.")
    debug_token_diff(engine.tokenizer, response, streamed_response)

if __name__ == "__main__":
    test_engine()