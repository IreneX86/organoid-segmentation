import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
A=json.load(open("out/a2_msc.json")); C2=json.load(open("out/c2_within_expressing.json")); F=json.load(open("out/f_oracle_bound.json"))
S=list(F); bands=["≥32","16–32","8–16","4–8","2.5–4"]
fig,ax=plt.subplots(1,3,figsize=(16,4.6))
for key,c,lab in (("EE1","#2a6fdb","ER(z) vs ER(z+1)  [reference]"),("FE","#d9480f","Fluo(z) vs ER(z)"),("FEnull","#868e96","Fluo vs rotated ER  [null]")):
    v=np.array([A["results"][s]["msc"][key] for s in S]); ax[0].plot(bands,v.mean(0),"o-",color=c,label=lab); ax[0].fill_between(bands,v.min(0),v.max(0),color=c,alpha=.15)
ax[0].set_xlabel("spatial period (px)"); ax[0].set_ylabel("magnitude-squared coherence"); ax[0].set_title("A. Shared linear information by scale\n(5 specimens, band = min–max)"); ax[0].legend(fontsize=8,loc="center right"); ax[0].set_ylim(0,1)
x=np.arange(len(S)); w=.27
for i,(arm,c,lab) in enumerate((("er_adj","#2a6fdb","ER(z+1) features [reference]"),("fluo","#d9480f","Fluo features"),("fluo_flip","#868e96","mirrored Fluo [null]"))):
    ax[1].bar(x+(i-1)*w,[C2[s][f"{arm}|t_fine"]["r"] for s in S],w,color=c,label=lab)
ax[1].set_xticks(x,S); ax[1].set_ylabel("Pearson r, held-out specimen"); ax[1].set_title("B. Predicting ER fine detail, leave-one-specimen-out\n(only inside ER-reporter-positive neighbourhoods)"); ax[1].legend(fontsize=8,loc="upper center",bbox_to_anchor=(0.5,-0.12),ncol=3); ax[1].set_ylim(-.05,1)
ax[2].bar(x,[F[s]["oracle_dice"] for s in S],color="#5f3dc4")
ax[2].set_xticks(x,[f"{s}\n{F[s]['batch']}" for s in S]); ax[2].set_ylim(0,1); ax[2].set_ylabel("Dice vs pseudo-label")
ax[2].set_title("C. Pseudo-label Dice ceiling for a model that\nknows ER perfectly in every cell (mosaic reporter)")
plt.tight_layout(); plt.savefig("out/fig_summary.png",dpi=120); print("ok")
