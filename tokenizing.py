from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path("./data/json").glob("**/dataset_*.json")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
    'sp', 'dp', 'ptcl', 'ipv', 'vln', 'tnnl', 'bi_dur', 'bi_pkt', 'bi_byte',
    's2d_dur', 's2d', 's2d_byte', 'd2s_dur', 'd2s', 'd2s_byte', 'bi_min_ps',
    'bi_mean_ps', 'bi_std_ps', 'bi_max_ps', 's2d_min_ps', 's2d_mean_ps',
    's2d_std_ps', 's2d_max_ps', 'd2s_min_ps', 'd2s_mean_ps', 'd2s_std_ps',
    'd2s_max_ps', 'bi_min_pi_ms', 'bi_mean_pi_ms', 'bi_std_pi_ms',
    'bi_max_pi_ms', 's2d_min_pi_ms', 's2d_mean_pi_ms', 's2d_std_pi_ms',
    's2d_max_pi_ms', 'd2s_min_pi_ms', 'd2s_mean_pi_ms', 'd2s_std_pi_ms',
    'd2s_max_pi_ms', 'bi_syn', 'bi_cwr', 'bi_ece', 'bi_urg', 'bi_ack',
    'bi_psh', 'bi_rst', 'bi_fin', 's2d_syn', 's2d_cwr', 's2d_ece',
    's2d_urg', 's2d_ack', 's2d_psh', 's2d_rst', 's2d_fin', 'd2s_syn',
    'd2s_cwr', 'd2s_ece', 'd2s_urg', 'd2s_ack', 'd2s_psh', 'd2s_rst',
    'd2s_fin', 'app_name', 'app_cat', 'req_server_name',
    'client_fingerprint', 'server_fingerprint', 'content_type', 'category',
])

tokenizer.save_model("models/tokenizer_iot")

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


tokenizer = ByteLevelBPETokenizer(
    "./models/tokenizer_iot/vocab.json",
    "./models/tokenizer_iot/merges.txt",
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=576)

