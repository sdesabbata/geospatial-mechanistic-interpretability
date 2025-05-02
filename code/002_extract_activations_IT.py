# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

HF_CACHE = 'your_cache_dir'

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")

import warnings
warnings.filterwarnings('ignore')


# Load data from txt file and convert to geodataframe
df = pd.read_csv('storage/geonames_IT.csv')
prov = pd.read_csv('storage/id_provinces_IT.csv', index_col=0)
prov = prov[['SIGLA','DEN_PCM']]
prov.columns = ['admin2 code','admin2 code text']
df = df.merge(prov, how='inner', on=['admin2 code'])
df = df.loc[(df['feature class'] == 'P') & (df['population'] >= 500)]
# df['admin1 code'] = df['admin1 code'].astype(str)
# df = df[df['admin1 code']!= 'nan']

def get_prompt(x):
    return f"{x['name']}, {x['admin2 code text']}"

df['prompt'] = df.progress_apply(get_prompt,axis=1)
places = list(df['prompt'].values)
# print(len(places))
# places[:5]


# Initialize tokenizer and model
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='your_cache_dir')
model = AutoModel.from_pretrained(model_name, cache_dir='your_cache_dir')

if torch.cuda.is_available():
    model.to(device)

# Hook to capture outputs
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Function to perform experiment and record results
def perform_experiment(ids, text, layers):
    # Reset activations
    global activations
    activations = {}

    # Attach hooks to the specified layers' attention outputs
    hooks = []
    for layer_number in layers:
        hook = model.layers[layer_number].post_attention_layernorm.register_forward_hook(get_activation(f'layer_{layer_number}_output'))
        hooks.append(hook)

    # Prepare input text and perform a forward pass
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Collect results for each layer
    results = []
    for layer_number in layers:
        activation_output = activations.get(f'layer_{layer_number}_output')

        # Compute mean and max pooling
        if activation_output is not None:
            activation_output = activation_output.to('cpu')
            mean_pooling = torch.mean(activation_output, dim=1).numpy()
            max_pooling = torch.max(activation_output, dim=1).values.numpy()
        else:
            mean_pooling = None
            max_pooling = None

        # Append results for this layer
        results.append({
            'geonameid': ids,
            'prompt': text,
            'layer': layer_number,
            'activations': activation_output.numpy() if activation_output is not None else None,
            'mean_pooling': mean_pooling,
            'max_pooling': max_pooling
        })

    return results

# List of texts and layers for experiments
texts = [x for x in places]
layers = [7, 15, 31]
identifiers = df['geonameid'].values[:len(texts)]

# Initialize the DataFrame
df = pd.DataFrame(columns=['geonameid','prompt', 'layer', 'activations', 'mean_pooling', 'max_pooling'])

# Perform experiments and collect results
for ids,text in tqdm(zip(identifiers,texts), desc="Texts", total=len(texts)):
    results = perform_experiment(ids,text, layers)
    sub = pd.DataFrame(results)
    df = pd.concat([df, sub])

# Save activations
df.to_pickle('storage/activations_IT.pkl')
