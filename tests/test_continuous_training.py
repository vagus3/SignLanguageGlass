"""CTC repeats, padded batches, optimization, checkpoint parity and split leakage."""
import _path  # noqa
import sys
try:
    import torch
except ImportError:
    sys.exit(77)
import tempfile
from pathlib import Path
import numpy as np
from src.pose_sequence import PoseSequence
from src.continuous_sign import (CtcRecognizer, collate, greedy_decode, split_dataset,
                                 save_checkpoint, Recognizer)
from src.train_continuous import feature_cache, fit
from tools.evaluate_continuous import choose_threshold, negative_duration, summarize

torch.set_num_threads(1)
torch.manual_seed(0)
labels = ["<blank>","A","B"]
p = np.eye(3)[[1,1,0,1,2,2]]
assert greedy_decode(p,labels)[0] == ["A","A","B"]
rng = np.random.default_rng(4)
rows = []
for i,n in enumerate([12,17,14]):
    sequence = PoseSequence(np.arange(n)/30, rng.uniform(.1,.8,(n,49,2)),
                            np.ones((n,49),bool),(640,480))
    rows.append({"id":str(i),"sequence":sequence,"glosses":["A","A","B"],
                 "signer_id":f"s{i}","session_id":"day1","group_id":str(i)})
model = CtcRecognizer(3,hidden=16)
before = next(model.parameters()).detach().clone()
fit(model,rows,labels,epochs=1,batch=2)
assert not torch.equal(before,next(model.parameters()).detach())
# The feature cache changes cost, not results: identical loss history from the same seed.
histories=[]
for cache in (None,{}):
    torch.manual_seed(1)
    histories.append(fit(CtcRecognizer(3,hidden=16),rows,labels,epochs=3,batch=2,cache=cache))
assert histories[0]==histories[1] and len(cache)==len(rows)
assert set(cache)=={(r["id"],"2d") for r in rows}
assert feature_cache(rows,"2d")=={} and feature_cache(rows,"2d",limit=1) is None
model.eval()
with torch.inference_mode():
    x,lengths,_,_ = collate(rows,labels)
    batched = model(x,lengths)
    singleton = model(x[:1,:12],lengths[:1])
    assert torch.allclose(singleton[:,0],batched[:12,0],atol=1e-6)
with tempfile.TemporaryDirectory() as td:
    path=Path(td)/"ctc.pt"
    save_checkpoint(path,model,labels)
    loaded=Recognizer(path)
    assert loaded.predict(rows[0]["sequence"]) == Recognizer(path).predict(rows[0]["sequence"])
    assert all(torch.equal(a,b) for a,b in zip(model.parameters(),loaded.model.parameters()))
assert all(split_dataset(rows).values())
leaking=[{**rows[0],"split":"train"},{**rows[1],"split":"val","signer_id":"s0"},
         {**rows[2],"split":"test"}]
try:
    split_dataset(leaking)
    raise AssertionError("Signer leakage was allowed")
except ValueError:
    pass
reused=[{**rows[0],"split":"train"},{**rows[1],"split":"val","group_id":"0"},
        {**rows[2],"split":"test"}]
try:
    split_dataset(reused)
    raise AssertionError("Motion leakage was allowed")
except ValueError:
    pass
results=[{"id":"p","reference":["A"],"prediction":["A"],"score":.9,"llm_text":"가"},
         {"id":"n","reference":[],"prediction":["B"],"score":.8,"negative":True}]
review={"reviewer":"fixture","clips":{"p":{"prediction":["A"],"text":"가","meaning_preserved":True}}}
assert not choose_threshold(results,599,review)["enabled"]
assert choose_threshold(results,600,review)["enabled"]
assert not choose_threshold(results,600,{})["enabled"]
summary=summarize({"clips":results},[],{},review)
assert summary["recognition_coverage"]==1 and summary["meaning_reviewed_clips"]==1
assert summary["false_recognition_outputs"]==1
overlap=[{**rows[0],"negative":True,"glosses":[],"capture_start":0},
         {**rows[0],"id":"duplicate","negative":True,"glosses":[],"capture_start":0}]
try:
    negative_duration(overlap)
    raise AssertionError("Overlapping negative intervals counted twice")
except ValueError:
    pass
print("[OK] variable-length CTC optimization, repeats, reload and safety calibration")
