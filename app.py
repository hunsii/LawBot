from __future__ import annotations

import sys
sys.path.insert(0, "polyglot-jax-inference/src")

from modeling import Transformer
from miscellaneous import get_conversion_rules, convert_weights, get_sharding_rules
from transformers import AutoTokenizer, AutoModelForCausalLM

import jax
import jax.numpy as jnp
import chex
from jax.sharding import Mesh, NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit

import gradio as gr
# import uvicorn
import pandas as pd
df = pd.read_csv("../data.csv")

mesh = Mesh(mesh_utils.create_device_mesh((1, 8)), ("dp", "mp"))

tokenizer = AutoTokenizer.from_pretrained("KRAFTON/KORani-v1-13B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model_hf = AutoModelForCausalLM.from_pretrained("KRAFTON/KORani-v1-13B")

model = Transformer(
    vocab_size=model_hf.config.vocab_size,
    layers=model_hf.config.num_hidden_layers,
    dim=model_hf.config.hidden_size,
    heads=model_hf.config.num_attention_heads,
    hidden=model_hf.config.intermediate_size,
    rotary=int(model_hf.config.rotary_pct * model_hf.config.hidden_size // model_hf.config.num_attention_heads),
    eps=model_hf.config.layer_norm_eps,
)
params = jax.tree_map(
    lambda param, spec: jax.device_put(param, NamedSharding(mesh, spec)),
    convert_weights(model_hf.state_dict(), get_conversion_rules(model)),
    get_sharding_rules(model)
)

# temperature = 0.8
# max_length = 1024

@pjit
def generate(x: chex.Array, mask: chex.Array, params: chex.ArrayTree, rng: chex.PRNGKey, temperature: float, max_length: int) -> chex.Array:
    rng, new_rng = jax.random.split(rng)
    generated = jnp.zeros((x.shape[0], 1024), dtype=jnp.int32)

    logits, variables = model.apply({"params": params}, x, mask, mutable=["cache"])
    new_tokens = jax.random.categorical(rng, logits[:, -1, :] / 0.8)
    generated = jnp.roll(generated, -1, 1).at[:, -1].set(new_tokens)
    
    def body_fn(_: int, state: tuple[chex.Array, ...]):
        x, cache, rng, generated = state
        rng, new_rng = jax.random.split(rng)

        logits, variables = model.apply({"params": params, "cache": cache}, x[:, None], mutable=["cache"])
        new_tokens = jax.random.categorical(rng, logits[:, -1, :] / 0.8)
        generated = jnp.roll(generated, -1, 1).at[:, -1].set(new_tokens)
        return new_tokens, variables["cache"], new_rng, generated
    
    state = (new_tokens, variables["cache"], new_rng, generated)
    state = jax.lax.fori_loop(0, 1024 - 1, body_fn, init_val=state)
    return state[3]

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load model from HuggingFace Hub
sbert_tokenizer = AutoTokenizer.from_pretrained('jhgan/ko-sroberta-multitask')
sbert_model = AutoModel.from_pretrained('jhgan/ko-sroberta-multitask')

import pickle
import numpy as np
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm

def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

def get_similiar(sentences):
    encoded_input = sbert_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = sbert_model(**encoded_input)

    # Perform pooling. In this case, mean pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    vecs1 = sentence_embeddings.numpy()

    with open("../precedent_embedding_dict.pickle","rb") as f:
        precedent_dict = pickle.load(f)
    precedent_name_list = list(precedent_dict.keys())

    # print(f"입력한 사례\n{sentence}")
    # print('-'*80)

    cs_list = []

    for key2, vecs2 in tqdm(precedent_dict.items(), leave=False):
        sub_list = [-1]
        for v1 in vecs1:
            for v2 in vecs2:
                cs = cos_sim(v1, v2)
                sub_list.append(cs)
        cs_list.append(np.max(sub_list))

    cs_list = np.array(cs_list)

    # 가장 큰 값과 그 값의 인덱스 뽑아내기
    SORT_LENGTH = 100
    sorted_indices = np.argsort(cs_list)
    largest_values = cs_list[sorted_indices[-SORT_LENGTH:]][::-1]
    largest_indices = sorted_indices[-SORT_LENGTH:][::-1]

    MAX_RESULT = 10
    result_list = []
    for i in range(SORT_LENGTH):
        # print(i)
        similar_precedent_score = largest_values[i]#np.max(cs_list)
        similar_precedent_index = largest_indices[i]#np.argmax(cs_list)
        similar_precedent_number = precedent_name_list[similar_precedent_index]
        
        with open(f"../precedent/judgment/{list(precedent_name_list)[similar_precedent_index]}.txt", "r") as f:
            similar_precedent = f.read()
        
        with open(f"../precedent/summary/{similar_precedent_number}.txt", "r") as f:
            similar_precedent_summary = f.read()

        with open(f"../precedent/reference/{similar_precedent_number}.txt", "r") as f:
            similar_precedent_reference = f.read()

        if similar_precedent == '':
            continue
        if similar_precedent_summary == '':
            continue
        # print(f"유사한 판례: {similar_precedent_number}, {similar_precedent_score:.2f}\n{similar_precedent}")
        # print("- - - "*10)
        data_dict = {
            'id': similar_precedent_number, 
            'score': similar_precedent_score, 
            'similar_precedent': similar_precedent,
            'summary': similar_precedent_summary, 
            'reference': similar_precedent_reference
        }
        result_list.append(data_dict)
        if i >= MAX_RESULT:
            break
        # print('-'*80)
    return result_list

import json # import json module

json_file_path = "../legalqa.jsonlines.txt"
# with statement
json_data = []
with open(json_file_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        json_data.append(data)

def text_preprocessing(text):
    sentences = []
    for line in text.splitlines():
        sentences.append(line.strip())
    if len(sentences) == 0:
        return ""
    return ' '.join(sentences)

def process_chat(prompt, context, temperature, max_length, seed):
    prompt = prompt.strip()
    # sentences = prompt[-1][0]
    # print(type(sentences))
    # print(prompt)
    similiar_list = get_similiar([prompt])
    
    inner_count = 1
    summary1 = text_preprocessing(similiar_list[0]['summary'])
    score1 = similiar_list[0]['score']
    id1 = similiar_list[0]['id']
    # inner_count = 1
    while len(summary1) == 0:
        # print(inner_count)
        summary1 = text_preprocessing(similiar_list[inner_count]['summary'])
        score1 = similiar_list[inner_count]['score']
        id1 = similiar_list[inner_count]['id']
        inner_count += 1

    summary2 = text_preprocessing(similiar_list[inner_count]['summary'])
    score2 = similiar_list[inner_count]['score']
    id2 = similiar_list[inner_count]['id']
    # inner_count = 1
    while len(summary2) == 0:
        # print(inner_count)
        summary2 = text_preprocessing(similiar_list[inner_count]['summary'])
        score2 = similiar_list[inner_count]['score']
        id2 = similiar_list[inner_count]['id']
        inner_count += 1
        
    
    text = f"""# Document\n{summary1}" # 한글 2문장으로 위의 Document를 요약해줘. 그리고 Document에 근거해서 다음 질문에 한문장으로 답해주세요.:"{prompt}"\n### Assistant: 1)"""
    print(text)
    print('-'*80)
    encodings = tokenizer(text, max_length=2048, padding="max_length", truncation=True, return_tensors="np")

    with mesh:
        generated = generate(
            jnp.asarray(encodings.input_ids, dtype=jnp.int32),
            jnp.asarray(encodings.attention_mask, dtype=jnp.bool_),
            params,
            jax.random.PRNGKey(seed),
            temperature, 
            max_length
        )
        generated.block_until_ready()

    # print(tokenizer.decode(generated[0].tolist()).split("###")[0])

    # output_list.append(tokenizer.decode(generated[0].tolist()).split("###")[0])
    data_dict = {}
    data_dict['score1'] = score1
    data_dict['score2'] = score2
    data_dict['id1'] = id1
    data_dict['id2'] = id2
    data_dict['output'] = "1)" +tokenizer.decode(generated[0].tolist()).split("###")[0]

    
    precednet = df[df['판례일련번호'] == int(id1)].iloc[0]
    output = f"[{precednet.법원명}] {precednet.선고일자}, {precednet.선고}, {precednet.사건번호}, {precednet.판결유형}"
    output += f"\n{precednet.사건명}"
    output += f"\n상세링크: https://www.law.go.kr{precednet.판례상세링크}"
    output += f"\n답변:\n{data_dict['output']}"

    
    precednet = df[df['판례일련번호'] == int(id2)].iloc[0]
    output += '\n\n\n그 외 유사한 판례' + '-'*80
    output += f"\n[{precednet.법원명}] {precednet.선고일자}, {precednet.선고}, {precednet.사건번호}, {precednet.판결유형}"
    output += f"\n{precednet.사건명}"
    
    output += f"\n상세링크: https://www.law.go.kr{precednet.판례상세링크}"
    return output, context

with gr.Blocks(analytics_enabled=False, title='EasyLM Chat') as gradio_chatbot:
    gr.Markdown('# Law문철')
    gr.Markdown("2023-1 텍스트마이닝 기말 프로젝트")
    chatbot = gr.Chatbot(label='Chat history')
    msg = gr.Textbox(
        placeholder='Type your message here...',
        show_label=False
    )
    with gr.Row():
        send = gr.Button('Send')
        regenerate = gr.Button('Regenerate', interactive=False)
        clear = gr.Button('Reset')

    temp_slider = gr.Slider(
        label='Temperature', minimum=0, maximum=2.0,
        value=0.8
    )

    length_slider = gr.Slider(
        label='Max Length', minimum=0, maximum=2048,
        value=1024
    )

    seed_slider = gr.Slider(
        label='Random Seed', minimum=0, maximum=65535,
        value=76
    )

    context_state = gr.State(['', ''])

    def user_fn(user_message, history, context):
        return {
            msg: gr.update(value='', interactive=False),
            clear: gr.update(interactive=False),
            send: gr.update(interactive=False),
            regenerate: gr.update(interactive=False),
            chatbot: history + [[user_message, None]],
            context_state: [context[1], context[1]],
        }

    def model_fn(history, context, temperature, max_length, seed):
        history[-1][1], new_context = process_chat(
            history[-1][0], context[0], temperature, max_length, seed
        )
        return {
            msg: gr.update(value='', interactive=True),
            clear: gr.update(interactive=True),
            send: gr.update(interactive=True),
            chatbot: history,
            context_state: [context[0], new_context],
            regenerate: gr.update(interactive=True),
        }

    def regenerate_fn():
        return {
            msg: gr.update(value='', interactive=False),
            clear: gr.update(interactive=False),
            send: gr.update(interactive=False),
            regenerate: gr.update(interactive=False),
        }

    def clear_fn():
        return {
            chatbot: None,
            msg: '',
            context_state: ['', ''],
            regenerate: gr.update(interactive=False),
        }

    msg.submit(
        user_fn,
        inputs=[msg, chatbot, context_state],
        outputs=[msg, clear, send, chatbot, context_state, regenerate],
        queue=False
    ).then(
        model_fn,
        inputs=[chatbot, context_state, temp_slider, length_slider, seed_slider],
        outputs=[msg, clear, send, chatbot, context_state, regenerate],
        queue=True
    )
    send.click(
        user_fn,
        inputs=[msg, chatbot, context_state],
        outputs=[msg, clear, send, chatbot, context_state, regenerate],
        queue=False
    ).then(
        model_fn,
        inputs=[chatbot, context_state, temp_slider, length_slider, seed_slider],
        outputs=[msg, clear, send, chatbot, context_state, regenerate],
        queue=True
    )
    regenerate.click(
        regenerate_fn,
        inputs=None,
        outputs=[msg, clear, send, regenerate],
        queue=False
    ).then(
        model_fn,
        inputs=[chatbot, context_state, temp_slider, length_slider, seed_slider],
        outputs=[msg, clear, send, chatbot, context_state, regenerate],
        queue=True
    )
    clear.click(
        clear_fn,
        inputs=None,
        outputs=[chatbot, msg, context_state, regenerate],
        queue=False
    )

gradio_chatbot.queue(concurrency_count=1)
gradio_chatbot.launch(share=True)