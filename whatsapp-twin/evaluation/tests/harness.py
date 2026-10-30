import json, os, sys, types, re, random, tempfile, shutil, builtins  # noqa: F401 (re-exported to `from harness import *`)
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, BatchEncoding
from peft import LoraConfig, get_peft_model

HERE = os.path.dirname(os.path.abspath(__file__))
NB = json.load(open(os.path.join(HERE, "..", "notebooks", "evaluation.ipynb"), encoding="utf-8"))
CELLS = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]

# ---------- mock tokenizer (char-level, real chat-template format) ----------
class MockTok:
    chat_template = "qwen"
    def __init__(self):
        self.vocab = {"<|im_end|>": 1, "<|im_start|>": 2, "<unk>": 0}
        self.inv = {v: k for k, v in self.vocab.items()}
        self.eos_token, self.eos_token_id, self.unk_token_id, self.pad_token_id = "<|im_end|>", 1, 0, None
    def _id(self, ch):
        if ch not in self.vocab:
            i = len(self.vocab) % 4000 + 3 if len(self.vocab) >= 4000 else len(self.vocab)
            self.vocab[ch] = i; self.inv.setdefault(i, ch)
        return self.vocab[ch]
    def encode(self, text):
        out, i = [], 0
        while i < len(text):
            for sp in ("<|im_end|>", "<|im_start|>"):
                if text.startswith(sp, i): out.append(self.vocab[sp]); i += len(sp); break
            else:
                out.append(self._id(text[i])); i += 1
        return out
    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        texts = text if isinstance(text, list) else [text]
        ids = [self.encode(t) for t in texts]
        if return_tensors == "pt":
            return BatchEncoding({"input_ids": torch.tensor(ids), "attention_mask": torch.ones(len(ids), len(ids[0]), dtype=torch.long)})
        return BatchEncoding({"input_ids": ids[0] if not isinstance(text, list) else ids})
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return s + ("<|im_start|>assistant\n" if add_generation_prompt else "")
    def convert_tokens_to_ids(self, t): return self.vocab.get(t, 0)
    def decode(self, ids, skip_special_tokens=True):
        ids = ids.tolist() if hasattr(ids, "tolist") else ids
        return "".join(self.inv.get(i, "") for i in ids if not (skip_special_tokens and i in (1, 2)))

TOK = MockTok()
CFG = Qwen2Config(vocab_size=4100, hidden_size=48, intermediate_size=96, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)

def make_base():
    torch.manual_seed(0)
    return Qwen2ForCausalLM(CFG).eval()

fake_unsloth = types.ModuleType("unsloth")
class FastLanguageModel:
    @staticmethod
    def from_pretrained(model_name, **kw): return make_base(), TOK
    @staticmethod
    def for_inference(m): m.eval()
fake_unsloth.FastLanguageModel = FastLanguageModel
sys.modules["unsloth"] = fake_unsloth

# ---------- fake judge API ----------
class FakeJudge:
    calls = 0; fail_ids = set(); mode = "prefers_dpo"
    def __init__(self, **kw): self.chat = self; self.completions = self
    def create(self, model, messages, **kw):
        FakeJudge.calls += 1
        p = messages[0]["content"]
        if any(f"[[{i}]]" in p for i in FakeJudge.fail_ids):
            raise RuntimeError("Error code: 429 - rate limit")
        a = re.search(r"CANDIDATE A: (.*?)\n\nCANDIDATE B", p, re.S).group(1)
        _b = re.search(r"CANDIDATE B: (.*?)\n\nAnswer:", p, re.S).group(1)  # unused: only checks B is in the prompt
        dpo_texts = FakeJudge.dpo_texts
        rnd = random.Random(hash(p) % 1000)
        if FakeJudge.mode == "prefers_dpo":
            ans = "A" if a in dpo_texts else "B"
            if rnd.random() < 0.15: ans = "A"                      # some position bias
            ans = rnd.choice([ans, f"Candidate {ans}", f"**{ans}**", ans + "."])  # messy formatting
        else:
            ans = "TIE"
        msg = types.SimpleNamespace(content=ans)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])
import openai; openai.OpenAI = FakeJudge
import getpass as gp; gp.getpass = lambda prompt="": "fake-key"

def synthetic_conversations(n, seed=0):
    # Synthetic Egyptian-Arabic-style chats: the tests never need (or touch) real private data.
    rng = random.Random(seed)
    them = ["ازيك عامل ايه", "خلصت الشغل؟", "هتيجي امتى", "شوفت الفيديو؟", "تمام", "طب ماشي", "ايه رأيك", "معلش اتأخرت"]
    me = ["الحمد لله تمام", "لسه شويه", "بكره ان شاء الله", "اه شوفته جامد", "ماشي", "اوكي", "معلش مشغول", "خلاص تمام"]
    sys_msg = {"role": "system", "content": "Continue the conversation as Ahmed."}
    rows = []
    for i in range(n):
        turns = [sys_msg]
        for t in range(rng.randint(1, 4)):
            turns.append({"role": "user", "content": f"{rng.choice(them)} {i}-{t}"})
            turns.append({"role": "assistant", "content": f"{rng.choice(me)} {i}-{t}"})
        rows.append({"id": f"conv{i:05d}", "messages": turns})
    return rows

def setup_drive(root, n_valid=1500):
    os.makedirs(f"{root}/ahmed_twin", exist_ok=True); os.makedirs(f"{root}/DPO/runs/dpo_v7_rpo", exist_ok=True)
    rows = synthetic_conversations(n_valid)
    with open(f"{root}/ahmed_twin/validation.jsonl", "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # DPO dev: 40 prompts built exactly like the builder does
    with open(f"{root}/DPO/dpo_eval_hard.jsonl", "w", encoding="utf-8") as f:
        for r in rows[100:140]:
            f.write(json.dumps({"prompt": TOK.apply_chat_template(r["messages"][:-1]), "chosen": "x", "rejected": "y"}, ensure_ascii=False) + "\n")
    # Fictional placeholder names only -- real names must never appear in this public repo.
    open(f"{root}/ahmed_twin/private_names.txt", "w", encoding="utf-8").write(
        "# comment lines must be ignored\nفلانه\nعلانه\nرح\nZed\n")
    # real LoRA adapter + DPO-style run_summary with its fingerprint
    peft_m = get_peft_model(make_base(), LoraConfig(r=4, lora_alpha=4, target_modules=["q_proj","v_proj","o_proj","up_proj","down_proj","gate_proj","k_proj"]))
    torch.manual_seed(5)
    with torch.no_grad():
        for n, p in peft_m.named_parameters():
            if "lora_" in n: p.add_(torch.randn_like(p) * 0.05)
    fp = sum(p.detach().float().abs().sum().item() for n, p in peft_m.named_parameters() if "lora_" in n)
    peft_m.save_pretrained(f"{root}/DPO/runs/dpo_v7_rpo/adapter")
    json.dump({"lora_fingerprint": {"after": fp}}, open(f"{root}/DPO/runs/dpo_v7_rpo/run_summary.json", "w"))
    return fp

def prep(src, root, overrides):
    if src.startswith("!pip") or "drive.mount" in src: return None
    src = src.replace('DRIVE_ROOT = "/content/drive/MyDrive"', f'DRIVE_ROOT = "{root}"')
    src = src.replace('"cuda"', '"cpu"').replace("assert torch.cuda.is_available()", "assert True or torch.cuda.is_available()")
    src = src.replace('print("GPU:", torch.cuda.get_device_name(0))', 'print("GPU: (cpu test)")')
    for k, v in overrides.items():
        src = re.sub(rf"^{k} = .*$", f"{k} = {v}", src, flags=re.M)
    return src

def run_all(root, ns=None, overrides=None, stop_before=None):
    ns = ns if ns is not None else {}
    for i, c in enumerate(CELLS):
        if stop_before and stop_before in c: break
        src = prep(c, root, overrides or {})
        if src is None: continue
        exec(compile(src, f"<cell {i}>", "exec"), ns)
    return ns
