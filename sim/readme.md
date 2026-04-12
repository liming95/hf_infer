# introduction
This is a simulator to research the influence  for the next problem:
under the continuous batching, what will happend when introduce SD into system 
a queueing system with "memory constraint + probabilistic rejection + dynamic batching" in discrete time
## key variables
batch size, seq len, chunk size, accept rate
## relations
- batch x seq_len: compute cost, KV pressure
- chunk x SD : verification granularity, rejection waste
- accept rate x batch: effective throughput, wasted compute
- kv growth x scheduling -> system saturation point
## performance target
- throughput suface: (batch, seq_len, chunk) -> tokens/sec
- SD gain curve: (accept rate, chunk) -> speedup
- stability boundary: stable region vs unstable regine
- packing_efficiency, GPU utilization (SM, memory)
# flow
1. scheduler picks active requests
2. form batch
3. run HF compute model
4. run SD logic (accept/reject)
5. update KV cache
6. update queue

# component
- request generator: generate requests like rollout in RLHF.(prompt, id)
- batch scheduler: the requests generated in the last step and the the requests processed in pre will be selected from the queue according to the kv buffer.
- hf executor:
- sd accepter:  accept the next tokens by accept rate

# function
## workload generator
request:
- prompt_len
- max_new_tokens
- arrival_time
- draft quality（ accept rate）

generator type:
- Poisson arrival（LLM serving）
- batch rollout（RL）
- mixture workload（long & short）

## scheduler + kv simulator
### continuous batching
queue -> select active requests -> form batch

### kv growth tracking
allocat kv cache by page size
for each request:
seq_len(t) += generated tokens
KV_memory += f(seq_len)

### memory budget control
if KV_total > GPU_budget:
    block new requests

### note: budget size and request size and chunk size
in this part, chunk size need to be consider, which will affect the kv occupancy.
kv budget size: gpu memory - model weights - activation buffer - runtime overhead (50% ~ 70% gpu memory)

request memory size = kv_size(seq_len) = batch * seq_len * kv_size_per_token
kv_size_per_token = 2 * layers * hidden_size x dtype

batch_max = kv_budget/ (avg_seq_len * kv_per_token)


## compute model(HF proxy)
input: (batch_size, seq_len) -> (batch_size, seq_len+chunk size)
output: (latency, throughput, memory, gpu utilization)
start kv
### flow
- prefill
'''
outputs = model(
    input_ids=prompt_ids,
    use_cache=True,
)
past_key_values = outputs.past_key_values
'''
- decode
'''
outputs = model(
    input_ids=new_tokens,          # shape: (batch, chunk_size)
    past_key_values=past_key_values,
    use_cache=True,
)
past_key_values = outputs.past_key_values
'''
### performance metric
- attention cost vs seq_len
- batch scaling curve
- chunk overhead

### note
compute scaling: cost ∝ batch * seq_len * chunk
compute model: pad-based batching & batch packing inefficiency
unused capacity = capacity - packed_tokens
packing efficiency = f(seq_len distribution, chunk size, admission policy)

'''
active_requests = [...]

<!-- for each step:
    select subset s.t.
        KV_total ≤ budget
        maximize tokens processed -->
        active_requests = select_requests()

for r in active_requests:
    compute += f(seq_len[r], chunk_size)
'''

## Speculative Decoding Engine
- generate candidator tokens: draft model (history data) -> propose k tokens
- verify: accept with probability α
- reject: rollback + recompute

note: probability model α = f(chunk size, seq_len, model quality)