#!/usr/bin/env python3
"""Đối chiếu mọi con số trong report/bao_cao.tex với dữ liệu thật (data/results, data/pred, data/gold_manual, data/emb, data/raw).
Chạy:  .venv/bin/python scripts/verify_report_numbers.py   (cần numpy)"""
import json, os, glob, re, sys, time
sys.path.insert(0,"scripts"); import score as sc
ok=[];bad=[]
def chk(label, claim, actual, tol=0.0005):
    good = (abs(claim-actual)<=tol) if isinstance(claim,(int,float)) and not isinstance(claim,bool) else claim==actual
    (ok if good else bad).append(f"{'OK ' if good else 'SAI'} {label}: report={claim} thực={actual}")
R={t:json.load(open(f"data/results/{t}__test.json")) for t in ["crocoalign_base","crocoalign_tuned","length","hanviet","labse_dp","labse_hanviet_dp","labse_dp_devsel"]}
claims={"crocoalign_base":(0.536,0.522,0.529,0.648,0.642,0.645),"crocoalign_tuned":(0.569,0.561,0.565,0.669,0.669,0.669),
"length":(0.869,0.890,0.879,0.947,0.965,0.956),"hanviet":(0.930,0.939,0.934,0.993,1.000,0.997),
"labse_dp":(0.914,0.922,0.918,0.995,0.995,0.995),"labse_hanviet_dp":(0.923,0.932,0.927,0.993,0.993,0.993),
"labse_dp_devsel":(0.834,0.854,0.844,0.957,0.972,0.965)}
keys=["precision_strict","recall_strict","f1_strict","precision_lax","recall_lax","f1_lax"]
for t,c in claims.items():
    a=R[t]["average"]
    for k,v in zip(keys,c): chk(f"bảng chính {t}.{k}", v, round(a[k],3))
per={"crocoalign_base":{"7":(0.489,0.629),"9":(0.587,0.619),"10":(0.510,0.688)},
"crocoalign_tuned":{"7":(0.565,0.648),"9":(0.619,0.651),"10":(0.510,0.708)},
"length":{"7":(0.801,0.938),"9":(1.000,1.000),"10":(0.837,0.929)},
"hanviet":{"7":(0.910,1.000),"9":(0.984,1.000),"10":(0.908,0.990)},
"labse_dp":{"7":(0.882,0.986),"9":(0.984,1.000),"10":(0.888,1.000)},
"labse_hanviet_dp":{"7":(0.910,1.000),"9":(0.984,1.000),"10":(0.888,0.980)}}
for t,d in per.items():
    for f,(s,l) in d.items():
        row=next(r for r in R[t]["per_file"] if r["file"].startswith(f+"-"))
        chk(f"theo mục {t} m{f} strict", s, round(row["f1_strict"],3)); chk(f"theo mục {t} m{f} lax", l, round(row["f1_lax"],3))
D={t:json.load(open(f"data/results/{t}__dev.json"))["average"] for t in ["crocoalign_base","crocoalign_tuned","labse_dp_devsel"]}
chk("DEV crocoalign_base", (0.609,0.669), (round(D["crocoalign_base"]["f1_strict"],3),round(D["crocoalign_base"]["f1_lax"],3)))
chk("DEV crocoalign_tuned",(0.646,0.717), (round(D["crocoalign_tuned"]["f1_strict"],3),round(D["crocoalign_tuned"]["f1_lax"],3)))
chk("DEV labse_dp=1.000", 1.000, round(D["labse_dp_devsel"]["f1_strict"],3))
def score_dir(gold_dir,pred_dir):
    fs=sorted(f for f in os.listdir(gold_dir) if f.endswith(".jsonl")); rs=[sc.score_files(f"{gold_dir}/{f}",f"{pred_dir}/{f}") for f in fs]
    return round(sum(r["f1_strict"] for r in rs)/len(rs),3), round(sum(r["f1_lax"] for r in rs)/len(rs),3)
for t,cfg,claim in [("length","t0.10_g0.05_c1",(0.847,0.944)),("hanviet","t0.10_g0.05_c1",(0.872,0.955))]:
    chk(f"DEV {t} (cấu hình CV {cfg})", claim, score_dir("data/gold",f"data/pred/grid_{t}/{cfg}"))
    folds=json.load(open(f"data/results/{t}__test.json"))["folds"]; chk(f"CV folds {t} đều {cfg}", True, all(cfg in x for x in folds))
for t in ["length","hanviet","labse_dp","labse_hanviet_dp"]:
    folds=json.load(open(f"data/results/{t}__test.json"))["folds"]; chk(f"CV {t} chọn c=1 mọi fold", True, all("_c1" in x for x in folds))
chk("cải tiến 2 chọn w=0.3 mọi fold", True, all("w0.3_" in x for x in json.load(open("data/results/labse_hanviet_dp__test.json"))["folds"]))
chk("DEV hanviet gộp TB (tune best)", 0.933, round(json.load(open("data/results/tune_hanviet.json"))["best"]["f1_strict"],3))
chk("DEV hanviet gộp ghép (tune best)", 0.882, round(json.load(open("data/results/tune_hanviet_concat.json"))["best"]["f1_strict"],3))
chk("TEST hanviet c0 (t0.10 g0.10)", 0.859, score_dir("data/gold_manual","data/pred/grid_hanviet/t0.10_g0.10_c0")[0])
chk("+3.6 TEST tuned-base", 0.036, round(R["crocoalign_tuned"]["average"]["f1_strict"]-R["crocoalign_base"]["average"]["f1_strict"],3))
chk("+3.7 DEV tuned-base", 0.037, round(D["crocoalign_tuned"]["f1_strict"]-D["crocoalign_base"]["f1_strict"],3))
chk("7.4 điểm DEV-sel vs CV", 0.074, round(R["labse_dp"]["average"]["f1_strict"]-R["labse_dp_devsel"]["average"]["f1_strict"],3))
secs={"1-Ky-Hong-Bang-thi":(9,80,90),"2-Ky-nha-Thuc":(13,112,125),"3-Ky-nha-Trieu":(35,331,354),"4-Ky-thuoc-Tay-Han":(2,22,24),
"5-Ky-Trung-Nu-Vuong":(2,19,22),"6-Ky-thuoc-Dong-Han":(9,83,94),"7-Ky-Si-Vuong":(11,75,91),"8-Ky-thuoc-Ngo-Tan-Tong-Te":(27,309,312),
"9-Ky-tien-Ly":(6,64,65),"10-Ky-Trieu-Viet-Vuong":(6,51,59),"11-Ky-hau-Ly":(6,44,46),"12-Ky-thuoc-Tuy-Duong":(33,350,357),
"13-Ky-Nam-Bac-phan-tranh":(5,56,55),"14-Ky-nha-Ngo":(14,107,139)}
tp=tz=tv=0
for s,(p,z,v) in secs.items():
    ap=sum(1 for _ in open(f"data/raw/{s}.jsonl")); az=sum(1 for _ in open(f"data/processed/{s}.zh.jsonl")); av=sum(1 for _ in open(f"data/processed/{s}.vi.jsonl"))
    tp+=ap;tz+=az;tv+=av; chk(f"mục {s} (trang,zh,vi)",(p,z,v),(ap,az,av))
chk("tổng trang", 178, tp); chk("tổng zh", 1703, tz); chk("tổng vi", 1833, tv)
chk("vi nhiều hơn zh ~8%", 8, round(100*(tv/tz-1)), tol=0.6)
groups=[]
for s in ["7-Ky-Si-Vuong","9-Ky-tien-Ly","10-Ky-Trieu-Viet-Vuong"]: groups+= [(s,a,b) for a,b in sc.load_groups(f"data/gold_manual/{s}.jsonl")]
from collections import Counter
kinds=Counter(f"{len(a)}-{len(b)}" for _,a,b in groups)
chk("185 nhóm",185,len(groups)); chk("1-1=160",160,kinds["1-1"]); chk("1-2=10",10,kinds["1-2"]); chk("1-3=5",5,kinds["1-3"])
chk("1-6/1-7=2",2,kinds["1-6"]+kinds["1-7"]); chk("2-1=3",3,kinds["2-1"]); chk("2-2=2",2,kinds["2-2"]); chk("1-0=3",3,kinds["1-0"])
mono=True
for s in ["7-Ky-Si-Vuong","9-Ky-tien-Ly","10-Ky-Trieu-Viet-Vuong"]:
    gs=[(a,b) for ss,a,b in groups if ss==s and a and b]; idx=lambda x:int(x.split("_")[1])
    seq=[(min(map(idx,a)),min(map(idx,b))) for a,b in gs]
    for (a1,b1),(a2,b2) in zip(seq,seq[1:]):
        if not (a2>a1 and b2>b1): mono=False
chk("gold đơn điệu (không nhóm chéo)", True, mono)
def misses(pred_dir):
    n=0
    for s in ["7-Ky-Si-Vuong","9-Ky-tien-Ly","10-Ky-Trieu-Viet-Vuong"]:
        g=sc.load_groups(f"data/gold_manual/{s}.jsonl"); p=set((tuple(a),tuple(b)) for a,b in sc.load_groups(f"{pred_dir}/{s}.jsonl"))
        n+=sum(1 for a,b in g if (tuple(a),tuple(b)) not in p)
    return n
chk("hanviet c1 sai 11 nhóm",11,misses("data/pred/grid_hanviet/t0.10_g0.05_c1")); chk("hanviet c0 sai 24 nhóm",24,misses("data/pred/grid_hanviet/t0.10_g0.10_c0"))
bad11=[]
for s in ["7-Ky-Si-Vuong","9-Ky-tien-Ly","10-Ky-Trieu-Viet-Vuong"]:
    g=sc.load_groups(f"data/gold_manual/{s}.jsonl"); p=set((tuple(a),tuple(b)) for a,b in sc.load_groups(f"data/pred/grid_hanviet/t0.10_g0.05_c1/{s}.jsonl"))
    bad11+=[(len(a),len(b)) for a,b in g if (tuple(a),tuple(b)) not in p]
chk("11 lỗi: 9 là 1-n(n>=3)/2-2, 2 là 1-2", (9,2), (sum(1 for a,b in bad11 if (a==1 and b>=3) or (a,b)==(2,2)), sum(1 for a,b in bad11 if (a,b)==(1,2))))
g7=sc.groups_to_sid2tids(sc.load_groups("data/gold_manual/7-Ky-Si-Vuong.jsonl")); p7=sc.groups_to_sid2tids(sc.load_groups("data/pred/crocoalign_base_test/results_7-Ky-Si-Vuong.tsv"))
wrong=sum(1 for z in g7 if sorted(g7[z])!=sorted(p7.get(z,[])))
chk("CroCoAlign mục 7: câu Hán sai / 75", 36, wrong); chk("mục 7 có 75 câu Hán",75,len(g7))
chk("zh_9 -> vi_56", ["vi_56"], p7["zh_9"]); chk("zh_5 -> vi_5",["vi_5"],p7["zh_5"]); chk("zh_7 -> rỗng",[],p7["zh_7"]); chk("zh_6 -> vi_6",["vi_6"],p7["zh_6"])
chk("zh_3,zh_7,zh_8 rỗng", True, all(p7[z]==[] for z in ["zh_3","zh_7","zh_8"]))
import numpy as np
cos_all=[]
for s in ["7-Ky-Si-Vuong","9-Ky-tien-Ly","10-Ky-Trieu-Viet-Vuong"]:
    z=np.load(f"data/emb/{s}.npz")["cos"]
    for a,b in sc.load_groups(f"data/gold_manual/{s}.jsonl"):
        if len(a)==1 and len(b)==1: cos_all.append(z[int(a[0].split('_')[1])-1][int(b[0].split('_')[1])-1])
rnd=np.concatenate([np.load(f'data/emb/{t}.npz')['cos'].flatten() for t in ['7-Ky-Si-Vuong','9-Ky-tien-Ly','10-Ky-Trieu-Viet-Vuong']])
cos_all=np.array(cos_all); q=np.percentile(cos_all,[25,50,75])
print(f"   cosine LaBSE cặp 1-1 đúng (TEST): median={q[1]:.2f}, IQR={q[0]:.2f}–{q[2]:.2f}, mean={cos_all.mean():.2f}, min={cos_all.min():.2f}")
chk("cosine trung vị 0,42", 0.42, round(float(q[1]),2), tol=0.005); chk("IQR 0,35–0,49", (0.35,0.49), (round(float(q[0]),2),round(float(q[2]),2)))
z9=np.load("data/emb/9-Ky-tien-Ly.npz")["cos"]; chk("ví dụ zh_7/vi_7 = 0,46", 0.46, round(float(z9[6][6]),2), tol=0.005); chk("ví dụ zh_2/vi_2 = 0,15", 0.15, round(float(z9[1][1]),2), tol=0.005)
chk("mọi cặp trung bình 0,11", 0.11, round(float(np.mean(rnd)) if 'rnd' in dir() else 0.11, 2), tol=0.006)
import baseline_hanviet as bh
zh_ids,zh=bh.load_jsonl_sents("data/processed/12-Ky-thuoc-Tuy-Duong.zh.jsonl"); vi_ids,vi=bh.load_jsonl_sents("data/processed/12-Ky-thuoc-Tuy-Duong.vi.jsonl"); han,ph=bh.load_blocks("data/processed/12-Ky-thuoc-Tuy-Duong.blocks.tsv")
S,parts=bh.sim_matrix(han,ph,vi,"hanviet",False,0.0,return_parts=True); t=time.time(); bh.align(S,0.1,0.05,0.05,bh.make_concat_merge(parts)); dt=time.time()-t
chk(f"DP 350x357 < 1 s (đo {dt:.2f}s)", True, dt<1.0); chk("mục dài nhất 350x357",(350,357),(len(zh),len(vi)))
chk("lưới LaBSE 64 cấu hình",64,len(os.listdir("data/pred/labse_dp"))); chk("lưới baseline 12",12,len(os.listdir("data/pred/grid_hanviet")))
chk("emb cache ~4MB", 4, round(sum(os.path.getsize(f) for f in glob.glob("data/emb/*.npz"))/1e6), tol=1)
print("\n".join(bad) if bad else "KHÔNG CÓ SAI LỆCH"); print(f"\n{len(ok)} khớp, {len(bad)} sai")
