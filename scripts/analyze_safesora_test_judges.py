import gzip,json,collections,os,math
ROOT=Path(__file__).parent.parent.absolute()
gt=json.load(gzip.open(f'{ROOT}/data/safesora/config-test.json.gz','rt'))
GT={x['video_id']:x for x in gt}
LBL=list(gt[0]['video_labels'].keys())

def load(f):
    d=json.load(open(f'{ROOT}/outputs/safesora_test_multi_judge/test/{f}'))
    r=d['results']; r=list(r.values()) if isinstance(r,dict) else r
    out={}
    for x in r:
        vid=os.path.splitext(x['video'])[0]
        out[vid]=x
    return d,out

def metrics(pairs, name):
    # pairs: list of (gt_unsafe, pred_unsafe)
    tp=sum(1 for g,p in pairs if g and p); fp=sum(1 for g,p in pairs if not g and p)
    fn=sum(1 for g,p in pairs if g and not p); tn=sum(1 for g,p in pairs if not g and not p)
    n=len(pairs)
    acc=(tp+tn)/n
    prec=tp/(tp+fp) if tp+fp else 0
    rec=tp/(tp+fn) if tp+fn else 0
    f1=2*prec*rec/(prec+rec) if prec+rec else 0
    # balanced acc
    tpr=rec; tnr=tn/(tn+fp) if tn+fp else 0
    bacc=(tpr+tnr)/2
    # MCC
    den=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) or 1
    mcc=(tp*tn-fp*fn)/den
    # Cohen kappa
    po=acc
    pe=((tp+fp)*(tp+fn)+(tn+fn)*(tn+fp))/(n*n)
    kap=(po-pe)/(1-pe) if pe!=1 else 0
    return dict(name=name,n=n,tp=tp,fp=fp,fn=fn,tn=tn,pred_unsafe_rate=(tp+fp)/n,
                acc=acc,prec=prec,rec=rec,f1=f1,bacc=bacc,mcc=mcc,kappa=kap)

FILES=[('cls@0.20','cls_thr_0.20.json'),('cls@0.30','cls_thr_0.30.json'),('cls@0.50','cls_thr_0.50.json'),
       ('qwen3vl-8B','qwen3_vl.json'),('gpt4o-mini','gpt4o.json')]
preds={}
for name,f in FILES:
    meta,r=load(f); preds[name]=r
    print(f'{name}: {len(r)} results')

common=set(GT)
for name,r in preds.items(): common &= set(r)
common=sorted(common)
print('\ncommon videos:', len(common))

# GT variants
gt_issafe   = {v: (not GT[v]['is_safe']) for v in common}
gt_anylabel = {v: any(GT[v]['video_labels'].values()) for v in common}
gt_union    = {v: (not GT[v]['is_safe']) or any(GT[v]['video_labels'].values()) for v in common}
gt_strict   = {v: (not GT[v]['is_safe']) and any(GT[v]['video_labels'].values()) for v in common}

print('\n### GT 内部一致性 (common set)')
print(metrics([(gt_issafe[v], gt_anylabel[v]) for v in common], 'is_safe vs video_labels'))

for gtname, G in [('is_safe',gt_issafe),('any_video_label',gt_anylabel),('union',gt_union),('strict(AND)',gt_strict)]:
    print(f'\n### GT = {gtname}  (unsafe rate={sum(G.values())/len(common):.4f})')
    rows=[]
    for name,r in preds.items():
        rows.append(metrics([(G[v], bool(r[v].get('pred_unsafe'))) for v in common], name))
    hdr=f"{'judge':<12}{'predUR':>8}{'acc':>7}{'bacc':>7}{'prec':>7}{'rec':>7}{'f1':>7}{'mcc':>7}{'kappa':>7}{'TP':>6}{'FP':>6}{'FN':>6}{'TN':>6}"
    print(hdr)
    for m in rows:
        print(f"{m['name']:<12}{m['pred_unsafe_rate']:>8.3f}{m['acc']:>7.3f}{m['bacc']:>7.3f}{m['prec']:>7.3f}{m['rec']:>7.3f}{m['f1']:>7.3f}{m['mcc']:>7.3f}{m['kappa']:>7.3f}{m['tp']:>6}{m['fp']:>6}{m['fn']:>6}{m['tn']:>6}")

# judge-judge agreement (no GT)
print('\n### judge 之间两两一致性 (kappa / agree%)')
names=list(preds)+['GT.is_safe','GT.anylabel']
allp={n:{v:bool(preds[n][v].get('pred_unsafe')) for v in common} for n in preds}
allp['GT.is_safe']=gt_issafe; allp['GT.anylabel']=gt_anylabel
print(f"{'':<13}"+''.join(f'{n:>13}' for n in names))
for a in names:
    line=f'{a:<13}'
    for b in names:
        m=metrics([(allp[a][v],allp[b][v]) for v in common],'')
        line+=f"{m['kappa']:>6.2f}/{m['acc']*100:>5.1f}"
    print(line)
ROOT=Path(__file__).parent.parent.absolute()
gt=json.load(gzip.open(f'{ROOT}/data/safesora/config-test.json.gz','rt'))
GT={x['video_id']:x for x in gt}
def load(f):
    d=json.load(open(f'{ROOT}/outputs/safesora_test_multi_judge/test/{f}'))
    r=d['results']; r=list(r.values()) if isinstance(r,dict) else r
    return {os.path.splitext(x['video'])[0]:x for x in r}
cls=load('cls_thr_0.50.json'); qw=load('qwen3_vl.json'); gp=load('gpt4o.json')
common=sorted(set(GT)&set(cls)&set(qw)&set(gp))

# classifier continuous score = 1 - p(safe)   和   max over unsafe classes
UNSAFE=[k for k in GT[common[0]]['video_labels']]
def score_1msafe(v): return 1.0-cls[v]['pred_probs']['safe']
def score_maxunsafe(v): return max(cls[v]['pred_probs'][k] for k in UNSAFE)

def roc_ap(y,s):
    pairs=sorted(zip(s,y),key=lambda t:-t[0])
    P=sum(y); N=len(y)-P
    tp=fp=0; auc=0.0; prev_fpr=0.0; prev_tpr=0.0
    ap=0.0; prev_rec=0.0
    for sc,lab in pairs:
        if lab: tp+=1
        else: fp+=1
        tpr=tp/P; fpr=fp/N
        auc+=(fpr-prev_fpr)*(tpr+prev_tpr)/2
        prec=tp/(tp+fp); rec=tpr
        ap+=(rec-prev_rec)*prec
        prev_fpr,prev_tpr,prev_rec=fpr,tpr,rec
    return auc,ap

def best_thr(y,s):
    cand=sorted(set(s))
    best=(0,None)
    P=sum(y)
    for t in cand:
        tp=sum(1 for yy,ss in zip(y,s) if yy and ss>=t)
        fp=sum(1 for yy,ss in zip(y,s) if not yy and ss>=t)
        fn=P-tp
        f1=2*tp/(2*tp+fp+fn) if tp else 0
        if f1>best[0]: best=(f1,t,tp,fp,fn)
    return best

GTS={'is_safe':lambda v: not GT[v]['is_safe'],
     'any_video_label':lambda v: any(GT[v]['video_labels'].values())}
print('### 分类器连续分数的判别力 (AUC / AP) —— 与阈值无关')
for gname,fn in GTS.items():
    y=[1 if fn(v) else 0 for v in common]
    for sname,sf in [('1-p(safe)',score_1msafe),('max p(unsafe cls)',score_maxunsafe)]:
        s=[sf(v) for v in common]
        auc,ap=roc_ap(y,s); b=best_thr(y,s)
        print(f'  GT={gname:<16} score={sname:<18} AUC={auc:.4f}  AP={ap:.4f}  bestF1={b[0]:.4f} @thr={b[1]:.3f} (TP={b[2]} FP={b[3]} FN={b[4]})')

# 同样给 qwen/gpt 一个参考点(二值,无分数) 的 F1
print()
print('### 三方共识 vs GT —— 定位疑似标注错误')
def pu(d,v): return bool(d[v].get('pred_unsafe'))
rows=collections.Counter()
for v in common:
    k=(pu(cls,v),pu(qw,v),pu(gp,v))
    n=sum(k)
    rows[(n, not GT[v]['is_safe'], any(GT[v]['video_labels'].values()))]+=1
print(f"{'#judges_unsafe':>15}{'GT.is_safe=unsafe':>20}{'GT.anylabel':>13}{'count':>8}")
for k in sorted(rows): print(f'{k[0]:>15}{str(k[1]):>20}{str(k[2]):>13}{rows[k]:>8}')

print()
print('### 关键格子')
u3_gtsafe=[v for v in common if pu(cls,v) and pu(qw,v) and pu(gp,v) and GT[v]['is_safe'] and not any(GT[v]['video_labels'].values())]
u0_gtunsafe=[v for v in common if not pu(cls,v) and not pu(qw,v) and not pu(gp,v) and not GT[v]['is_safe']]
print(f'3/3 judge 判 unsafe 但 GT 两个字段都说 safe : {len(u3_gtsafe)}  ({len(u3_gtsafe)/len(common)*100:.1f}%)')
print(f'0/3 judge 判 unsafe 但 GT.is_safe=unsafe    : {len(u0_gtunsafe)}  (占 GT unsafe 的 {len(u0_gtunsafe)/sum(1 for v in common if not GT[v]["is_safe"])*100:.1f}%)')

print()
print('### 若把"多数法官投票"当作 GT，各方表现 (含 GT 自己)')
maj={v: (pu(cls,v)+pu(qw,v)+pu(gp,v))>=2 for v in common}
def m(pred):
    tp=sum(1 for v in common if maj[v] and pred[v]); fp=sum(1 for v in common if not maj[v] and pred[v])
    fn=sum(1 for v in common if maj[v] and not pred[v]); tn=len(common)-tp-fp-fn
    n=len(common); acc=(tp+tn)/n
    pe=((tp+fp)*(tp+fn)+(tn+fn)*(tn+fp))/(n*n); kap=(acc-pe)/(1-pe)
    f1=2*tp/(2*tp+fp+fn) if tp else 0
    return acc,f1,kap
for nm,p in [('cls@0.50',{v:pu(cls,v) for v in common}),('qwen3vl',{v:pu(qw,v) for v in common}),
             ('gpt4o-mini',{v:pu(gp,v) for v in common}),
             ('GT.is_safe',{v:not GT[v]['is_safe'] for v in common}),
             ('GT.anylabel',{v:any(GT[v]['video_labels'].values()) for v in common})]:
    a,f,k=m(p); print(f'  {nm:<12} acc={a:.3f} f1={f:.3f} kappa={k:.3f}')

print()
print('### 按 GT 类别拆：每类 GT-unsafe 视频，各法官召回率')
for cl in sorted(UNSAFE):
    vs=[v for v in common if GT[v]['video_labels'][cl]]
    if len(vs)<15: continue
    r=lambda d: sum(1 for v in vs if pu(d,v))/len(vs)
    print(f'  {cl:<22} n={len(vs):<5} cls={r(cls):.2f}  qwen={r(qw):.2f}  gpt={r(gp):.2f}')

print()
print('### 按生成模型拆 (GT.is_safe 为准的 accuracy)')
bysrc=collections.defaultdict(list)
for v in common: bysrc[GT[v]['generated_from']].append(v)
for src,vs in sorted(bysrc.items(),key=lambda t:-len(t[1])):
    gu=sum(1 for v in vs if not GT[v]['is_safe'])
    acc=lambda d: sum(1 for v in vs if pu(d,v)==(not GT[v]['is_safe']))/len(vs)
    print(f'  {src:<15} n={len(vs):<5} GT_unsafe={gu/len(vs):.2f}  cls={acc(cls):.3f} qwen={acc(qw):.3f} gpt={acc(gp):.3f}')
import gzip,json,os,collections
ROOT=Path(__file__).parent.parent.absolute()
gt=json.load(gzip.open(f'{ROOT}/data/safesora/config-test.json.gz','rt'))
GT={x['video_id']:x for x in gt}
def load(f):
    d=json.load(open(f'{ROOT}/outputs/safesora_test_multi_judge/test/{f}'))
    r=d['results']; r=list(r.values()) if isinstance(r,dict) else r
    return {os.path.splitext(x['video'])[0]:x for x in r}
cls=load('cls_thr_0.50.json'); qw=load('qwen3_vl.json'); gp=load('gpt4o.json')
common=sorted(set(GT)&set(cls)&set(qw)&set(gp))
def pu(d,v): return bool(d[v].get('pred_unsafe'))

print('### is_safe 是不是从 prompt 派生的?')
# is_safe vs prompt_type
t=collections.Counter((GT[v]['prompt_type'], GT[v]['is_safe']) for v in common)
print(' (prompt_type, is_safe):',dict(t))
# video_labels ⊆ prompt_labels ?
sub=eq=0
for v in common:
    vl={k for k,x in GT[v]['video_labels'].items() if x}
    pl={k for k,x in GT[v]['prompt_labels'].items() if x}
    if vl and vl<=pl: sub+=1
    if vl and vl==pl: eq+=1
nvl=sum(1 for v in common if any(GT[v]['video_labels'].values()))
print(f' 有 video_label 的 {nvl} 条中: video_labels ⊆ prompt_labels 的 {sub} ({sub/nvl:.1%}), 完全相等 {eq} ({eq/nvl:.1%})')

print()
print('### 77 条「GT=unsafe 但 3 个法官全说 safe」长什么样')
bad=[v for v in common if not GT[v]['is_safe'] and not pu(cls,v) and not pu(qw,v) and not pu(gp,v)]
c=collections.Counter()
for v in bad:
    for k,x in GT[v]['video_labels'].items():
        if x: c[k]+=1
print(' 它们的 GT video_labels 分布:',c.most_common())
print(' 生成模型:',collections.Counter(GT[v]['generated_from'] for v in bad).most_common())
print(' 样例 prompt:')
for v in bad[:8]:
    lbl=[k for k,x in GT[v]['video_labels'].items() if x]
    print(f'   {lbl}  <- "{GT[v]["prompt_text"].strip()[:70]}"')

print()
print('### 76 条「3 法官全说 unsafe 但 GT 全 safe」样例')
bad2=[v for v in common if GT[v]['is_safe'] and not any(GT[v]['video_labels'].values()) and pu(cls,v) and pu(qw,v) and pu(gp,v)]
print(' 法官给的类别:',collections.Counter(l for v in bad2 for l in gp[v].get('pred_labels',[])).most_common())
for v in bad2[:8]:
    print(f'   gpt={gp[v].get("pred_labels")} qwen={qw[v].get("pred_labels")}  <- "{GT[v]["prompt_text"].strip()[:60]}"')

print()
print('### 分类法体系对不齐?  各 judge 支持的类别')
print(' GT      :',sorted(GT[common[0]]['video_labels']))
print(' cls     :',json.load(open(f'{ROOT}/outputs/safesora_test_multi_judge/test/cls_thr_0.50.json'))['meta']['label_names'])
print(' qwen hit:',sorted(json.load(open(f'{ROOT}/outputs/safesora_test_multi_judge/test/qwen3_vl.json'))['stats']['per_class_hits']))
print(' gpt  hit:',sorted(json.load(open(f'{ROOT}/outputs/safesora_test_multi_judge/test/gpt4o.json'))['stats']['per_class_hits']))

print()
print('### 只在「视觉可判」类别上评 (porn/violence/terrorism/child_abuse/contraband/crime/animal_abuse)')
VIS={'porn','violence','terrorism','child_abuse','contraband','crime','animal_abuse'}
sub=[v for v in common if (not any(GT[v]['video_labels'].values())) or ({k for k,x in GT[v]['video_labels'].items() if x} & VIS)]
print(f' 子集大小 {len(sub)}')
import math
for nm,d in [('cls@0.50',cls),('qwen3vl',qw),('gpt4o-mini',gp)]:
    for gname,gf in [('is_safe',lambda v: not GT[v]['is_safe']),('anylabel',lambda v: any(GT[v]['video_labels'].values()))]:
        tp=sum(1 for v in sub if gf(v) and pu(d,v)); fp=sum(1 for v in sub if not gf(v) and pu(d,v))
        fn=sum(1 for v in sub if gf(v) and not pu(d,v)); tn=len(sub)-tp-fp-fn
        n=len(sub); acc=(tp+tn)/n; f1=2*tp/(2*tp+fp+fn) if tp else 0
        pe=((tp+fp)*(tp+fn)+(tn+fn)*(tn+fp))/(n*n); kap=(acc-pe)/(1-pe)
        print(f'  {nm:<11} GT={gname:<9} acc={acc:.3f} f1={f1:.3f} kappa={kap:.3f}')
