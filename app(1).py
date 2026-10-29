import json
import math
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Smart Scheduler AI Lab", page_icon="⚡", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:Inter,sans-serif}
.stApp{background:linear-gradient(135deg,#f4f6ff,#eefbf7 55%,#fff7ec)}
.block-container{padding-top:1.2rem;max-width:1500px}
.hero{padding:1.6rem 1.8rem;border-radius:24px;color:white;background:linear-gradient(115deg,#302b8f,#536dfe,#00b8a9);box-shadow:0 12px 35px #302b8f22;margin-bottom:1.2rem}
.hero h1{color:white;font-size:2.25rem;font-weight:800;margin:0}.hero p{color:#eaf0ff;font-size:1.02rem;margin:.45rem 0 0}
div[data-testid="stMetric"]{background:white;border:1px solid #e5e9f4;padding:13px 15px;border-radius:16px;box-shadow:0 5px 16px #1923500a}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#f1f3ff,#effbf8)}
.card{background:#ffffffdd;border:1px solid #e4e9f5;padding:1.1rem;border-radius:18px}
h2,h3{color:#1c2550}
</style>
<div class="hero"><h1>⚡ Smart Scheduler AI Lab</h1>
<p>Understand your workload · compare scheduling policies · get a data-driven recommendation</p>
<p>🟦 FIFO &nbsp; 🟩 EDF &nbsp; 🟧 ML-EDF &nbsp; | &nbsp; Interactive analytics</p></div>
""", unsafe_allow_html=True)

REQUIRED = ["task_id","arrival_time","execution_time","deadline","priority","input_size"]

def make_workload(n, seed, difficulty):
    r=np.random.default_rng(seed)
    arrival=np.sort(r.integers(0,max(8,n//3),n))
    size=r.integers(1,101,n); priority=r.integers(1,6,n); typ=r.integers(0,3,n)
    execution=np.maximum(1,np.ceil(size/18+typ*1.2+r.normal(0,1,n))).astype(int)
    slack=r.integers(8,22,n) if difficulty=="Light" else r.integers(3,13,n) if difficulty=="Moderate" else r.integers(1,7,n)
    return pd.DataFrame({"task_id":[f"T{i+1:03d}" for i in range(n)],"arrival_time":arrival,
        "execution_time":execution,"deadline":arrival+execution+slack,"priority":priority,
        "task_type":np.array(["Compute","I/O-like","Mixed"])[typ],"input_size":size})

def validate(df):
    df=df.copy()
    missing=[c for c in REQUIRED if c not in df.columns]
    if missing: return False,"Missing columns: "+", ".join(missing),df
    for c in REQUIRED[1:]: df[c]=pd.to_numeric(df[c],errors="coerce")
    if df[REQUIRED].isna().any().any(): return False,"Required fields contain blanks or non-numeric values.",df
    if (df.arrival_time<0).any() or (df.execution_time<=0).any() or (df.deadline<0).any() or (df.priority<1).any() or (df.input_size<0).any():
        return False,"Arrival/deadline/input size must be ≥ 0; execution time > 0; priority ≥ 1.",df
    if df.task_id.astype(str).duplicated().any(): return False,"task_id values must be unique.",df
    df["task_id"]=df.task_id.astype(str)
    if "task_type" not in df: df["task_type"]="Compute"
    df["task_type"]=df["task_type"].fillna("Compute")
    return True,"Workload is valid.",df.sort_values(["arrival_time","task_id"]).reset_index(drop=True)

def features(df):
    typ=df.task_type.map({"Compute":0,"I/O-like":1,"Mixed":2}).fillna(0).to_numpy()
    rel=np.maximum(.1,df.deadline.to_numpy()-df.arrival_time.to_numpy())
    local=df.execution_time.rolling(4,min_periods=1).sum().to_numpy()
    return pd.DataFrame({"input_size":df.input_size.astype(float),"type_code":typ,"priority":df.priority.astype(float),
                         "rel_deadline":rel,"local_load":local})

@st.cache_resource
def train_models():
    r=np.random.default_rng(2026); n=4500
    size=r.integers(1,101,n); typ=r.integers(0,3,n); pri=r.integers(1,6,n)
    rel=r.integers(2,28,n); load=r.integers(1,35,n)
    execution=np.maximum(1,np.ceil(size/18+typ*1.2+r.normal(0,1,n))).astype(float)
    prob=1/(1+np.exp(-np.clip((execution+load*.18-rel)*.55-(pri-3)*.12,-12,12)))
    labels=r.binomial(1,prob)
    xr=pd.DataFrame({"input_size":size,"type_code":typ,"priority":pri})
    xc=xr.copy(); xc["rel_deadline"]=rel; xc["local_load"]=load
    xr1,xr2,yr1,yr2=train_test_split(xr,execution,test_size=.2,random_state=42)
    xc1,xc2,yc1,yc2=train_test_split(xc,labels,test_size=.2,random_state=42,stratify=labels)
    reg=RandomForestRegressor(n_estimators=80,max_depth=10,min_samples_leaf=2,random_state=42,n_jobs=-1).fit(xr1,yr1)
    clf=RandomForestClassifier(n_estimators=80,max_depth=9,min_samples_leaf=3,class_weight="balanced",random_state=42,n_jobs=-1).fit(xc1,yc1)
    rp=reg.predict(xr2); cp=clf.predict(xc2)
    metrics={"MAE":mean_absolute_error(yr2,rp),"RMSE":math.sqrt(mean_squared_error(yr2,rp)),"R²":r2_score(yr2,rp),
             "Accuracy":accuracy_score(yc2,cp),"Precision":precision_score(yc2,cp,zero_division=0),
             "Recall":recall_score(yc2,cp,zero_division=0),"F1":f1_score(yc2,cp,zero_division=0)}
    return reg,clf,metrics

def simulate(df,policy,reg,clf,risk_weight,cs_cost):
    w=df.copy().reset_index(drop=True); f=features(w)
    w["predicted_execution"]=np.maximum(1,reg.predict(f[["input_size","type_code","priority"]]))
    w["miss_probability"]=clf.predict_proba(f)[:,1]
    remaining=set(range(len(w))); now=float(w.arrival_time.min()); last=None
    start={}; finish={}; trace=[]; idle=0.; switch_time=0.
    while remaining:
        ready=[i for i in remaining if float(w.loc[i,"arrival_time"])<=now]
        if not ready:
            nxt=min(float(w.loc[i,"arrival_time"]) for i in remaining)
            trace.append({"Task":"IDLE","Start":now,"Finish":nxt,"State":"IDLE"}); idle+=nxt-now; now=nxt
            ready=[i for i in remaining if float(w.loc[i,"arrival_time"])<=now]
        def key(i):
            q=w.loc[i]; a=float(q.arrival_time); d=float(q.deadline)
            if policy=="FIFO": return (a,str(q.task_id))
            if policy=="EDF": return (d,a,str(q.task_id))
            return (d-risk_weight*float(q.miss_probability)*float(q.predicted_execution),d,a,str(q.task_id))
        i=min(ready,key=key); q=w.loc[i]
        if last is not None and last!=str(q.task_id) and cs_cost:
            trace.append({"Task":"CONTEXT SWITCH","Start":now,"Finish":now+cs_cost,"State":"CONTEXT_SWITCH"})
            now+=cs_cost; switch_time+=cs_cost
        start[i]=now; finish[i]=now+float(q.execution_time)
        trace.append({"Task":str(q.task_id),"Start":now,"Finish":finish[i],"State":"RUN"})
        now=finish[i]; last=str(q.task_id); remaining.remove(i)
    w["start_time"]=[start[i] for i in range(len(w))]; w["completion_time"]=[finish[i] for i in range(len(w))]
    w["turnaround_time"]=w.completion_time-w.arrival_time
    w["waiting_time"]=w.turnaround_time-w.execution_time
    w["response_time"]=w.start_time-w.arrival_time
    w["tardiness"]=np.maximum(0,w.completion_time-w.deadline)
    w["deadline_missed"]=w.completion_time>w.deadline
    elapsed=max(.0001,float(w.completion_time.max()-w.arrival_time.min()))
    run=float(w.execution_time.sum())
    metrics={"Policy":policy,"Tasks":len(w),"Deadline misses":int(w.deadline_missed.sum()),
       "Miss rate (%)":float(w.deadline_missed.mean()*100),"Avg waiting":float(w.waiting_time.mean()),
       "Avg turnaround":float(w.turnaround_time.mean()),"Avg response":float(w.response_time.mean()),
       "Avg tardiness":float(w.tardiness.mean()),"CPU utilization (%)":min(100,run/elapsed*100),
       "Throughput":len(w)/elapsed,"Context switches":max(0,len(w)-1),"Idle time":idle,
       "Context-switch time":switch_time,"Total time":float(now-w.arrival_time.min())}
    return metrics,w,pd.DataFrame(trace)

def recommend(results,goal):
    if goal=="Minimize deadline misses":
        ranked=results.sort_values(["Deadline misses","Avg tardiness","Avg waiting","Total time"])
        reason="Prioritized fewer deadline misses, then lower tardiness and waiting time."
    elif goal=="Minimize waiting time":
        ranked=results.sort_values(["Avg waiting","Deadline misses","Total time"])
        reason="Prioritized lower average waiting time, then deadline misses."
    else:
        ranked=results.copy()
        ranked["Recommendation score"]=ranked["Miss rate (%)"]/max(1,ranked["Miss rate (%)"].max())*.60+ranked["Avg waiting"]/max(.001,ranked["Avg waiting"].max())*.25+(100-ranked["CPU utilization (%)"])/100*.15
        ranked=ranked.sort_values("Recommendation score")
        reason="Balanced score: deadline-miss rate 60%, average waiting 25%, unused CPU capacity 15%."
    return str(ranked.iloc[0]["Policy"]),reason,ranked

with st.sidebar:
    st.header("🎛️ Simulation controls")
    source=st.radio("Workload source",["Generate sample","Upload CSV"])
    if source=="Generate sample":
        n=st.slider("Number of tasks",10,200,60,5)
        difficulty=st.select_slider("Workload difficulty",["Light","Moderate","Overloaded"],value="Moderate")
        seed=st.number_input("Random seed",0,999999,42)
        raw=make_workload(n,int(seed),difficulty)
    else:
        upload=st.file_uploader("Upload workload CSV",type=["csv"])
        if upload:
            try: raw=pd.read_csv(upload)
            except Exception as e:
                st.error(f"Could not read CSV: {e}"); raw=make_workload(40,42,"Moderate")
        else:
            st.info("Upload a CSV to use your own tasks."); raw=make_workload(40,42,"Moderate")
    st.divider()
    goal=st.selectbox("What matters most?",["Minimize deadline misses","Balance deadline misses + waiting","Minimize waiting time"])
    risk=st.slider("ML risk weight",0.,2.,.8,.1,help="Higher weight gives more influence to predicted risk and predicted execution time.")
    cs=st.slider("Context-switch cost",0.,2.,.2,.1)
    st.caption("Educational single-CPU, non-preemptive simulator.")

ok,msg,workload=validate(raw)
if not ok: st.error("⚠️ "+msg); st.stop()
st.success(f"✅ {msg} {len(workload)} tasks ready.")
reg,clf,ml= train_models()
rows=[]; details={}; traces={}
for policy in ["FIFO","EDF","ML-EDF"]:
    m,d,t=simulate(workload,policy,reg,clf,risk,cs); rows.append(m); details[policy]=d; traces[policy]=t
results=pd.DataFrame(rows); winner,reason,ranking=recommend(results,goal)
wid=str(pd.util.hash_pandas_object(workload[REQUIRED],index=False).sum())[:8]

tabs=st.tabs(["🏠 Overview","📊 Compare","🧠 Recommendation","🧾 Task explorer","🤖 ML health","📥 Export"])
with tabs[0]:
    st.subheader("Your workload at a glance")
    a,b,c,d=st.columns(4)
    a.metric("Tasks loaded",len(workload)); b.metric("Avg execution",f"{workload.execution_time.mean():.1f} units")
    c.metric("Tight-deadline tasks",int(((workload.deadline-workload.arrival_time)<=workload.execution_time+2).sum()))
    d.metric("Workload ID",wid)
    st.markdown(f"""<div class="card"><small>DATA-DRIVEN RECOMMENDATION</small><h2>🏆 {winner}</h2><p>{reason}</p><small>This result depends on the current workload and selected goal; it is not a universal guarantee.</small></div>""",unsafe_allow_html=True)
    st.markdown("### Understand the policies")
    x,y,z=st.columns(3)
    x.info("🟦 **FIFO**\n\nRuns ready tasks in arrival order.")
    y.success("🟩 **EDF**\n\nRuns the ready task with the earliest absolute deadline.")
    z.warning("🟧 **ML-EDF**\n\nUses model-estimated deadline risk and execution time to adjust deadline ranking.")
    st.markdown("### Workload preview")
    st.dataframe(workload.head(12),use_container_width=True,hide_index=True)
    st.download_button("⬇️ Download workload CSV",workload.to_csv(index=False).encode(), "workload.csv","text/csv")
with tabs[1]:
    st.subheader("Same workload, three scheduling policies")
    st.caption("ML-EDF is not automatically the winner. Compare the measured results.")
    cols=["Policy","Tasks","Deadline misses","Miss rate (%)","Avg waiting","Avg turnaround","Avg response","Avg tardiness","CPU utilization (%)","Throughput","Context switches"]
    st.dataframe(results[cols].round(2),use_container_width=True,hide_index=True)
    l,r=st.columns(2)
    with l:
        fig=px.bar(results,x="Policy",y="Deadline misses",color="Policy",text_auto=True,title="Deadline misses — lower is better",
          color_discrete_map={"FIFO":"#6675f5","EDF":"#08a88a","ML-EDF":"#f4a62a"})
        fig.update_layout(showlegend=False); st.plotly_chart(fig,use_container_width=True)
    with r:
        melt=results[["Policy","Avg waiting","Avg response"]].melt("Policy",var_name="Metric",value_name="Time")
        fig=px.bar(melt,x="Policy",y="Time",color="Metric",barmode="group",title="Waiting and response time — lower is better")
        st.plotly_chart(fig,use_container_width=True)
    fig=px.bar(results,x="Policy",y="CPU utilization (%)",color="Policy",text_auto=".1f",range_y=[0,100],title="CPU utilization — higher is not always better")
    fig.update_layout(showlegend=False); st.plotly_chart(fig,use_container_width=True)
    st.info("A high CPU utilization does not automatically mean fewer deadline misses. Use multiple metrics.")
with tabs[2]:
    st.subheader("Why was this scheduler recommended?")
    st.write(f"**Goal:** {goal}"); st.write(f"**Recommendation:** {winner}"); st.write(reason)
    st.dataframe(ranking.round(3),use_container_width=True,hide_index=True)
    st.markdown("### ML-EDF scheduling decisions")
    mld=details["ML-EDF"].copy()
    mld["slack_at_arrival"]=mld.deadline-mld.arrival_time-mld.execution_time
    st.dataframe(mld[["task_id","arrival_time","execution_time","predicted_execution","miss_probability","deadline","slack_at_arrival","start_time","completion_time","deadline_missed"]].sort_values("start_time").head(20).round(3),use_container_width=True,hide_index=True)
    st.caption("Risk probability is an estimate from synthetic training data, not a guarantee of a deadline miss.")
with tabs[3]:
    st.subheader("Task-level explorer")
    policy=st.selectbox("Choose scheduler",["FIFO","EDF","ML-EDF"])
    td=details[policy]
    missed=st.checkbox("Only show missed deadlines")
    if missed: td=td[td.deadline_missed]
    st.dataframe(td[["task_id","task_type","arrival_time","execution_time","predicted_execution","miss_probability","deadline","priority","start_time","completion_time","waiting_time","tardiness","deadline_missed"]].round(3),use_container_width=True,hide_index=True)
    tr=traces[policy]
    run=tr[tr.State=="RUN"]
    fig=px.timeline(run,x_start="Start",x_end="Finish",y="Task",color="Task",title=f"{policy} CPU execution timeline")
    fig.update_yaxes(autorange="reversed"); fig.update_layout(showlegend=False,xaxis_title="Simulation time")
    st.plotly_chart(fig,use_container_width=True)
    with st.expander("Show CPU trace (RUN / IDLE / CONTEXT SWITCH)"): st.dataframe(tr,use_container_width=True,hide_index=True)
    st.download_button("⬇️ Download task outcomes",td.to_csv(index=False).encode(),f"{policy.lower().replace('-','_')}_task_outcomes.csv","text/csv")
with tabs[4]:
    st.subheader("ML model health")
    st.warning("Models train on synthetic demo data at startup. These held-out metrics do not prove real-world or hardware performance.")
    a,b,c,d,e,f,g=st.columns(7)
    a.metric("MAE",f'{ml["MAE"]:.3f}'); b.metric("RMSE",f'{ml["RMSE"]:.3f}'); c.metric("R²",f'{ml["R²"]:.3f}')
    d.metric("Accuracy",f'{ml["Accuracy"]*100:.1f}%'); e.metric("Precision",f'{ml["Precision"]*100:.1f}%')
    f.metric("Recall",f'{ml["Recall"]*100:.1f}%'); g.metric("F1",f'{ml["F1"]*100:.1f}%')
    chart=pd.DataFrame({"Metric":["Accuracy","Precision","Recall","F1"],"Score":[ml[k] for k in ["Accuracy","Precision","Recall","F1"]]})
    fig=px.bar(chart,x="Metric",y="Score",range_y=[0,1],text_auto=".2f",color="Metric",title="Deadline-risk classifier scores")
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("""**Metrics guide**  
- **MAE / RMSE:** execution-time prediction error; lower is better.  
- **R²:** regression fit against held-out synthetic labels.  
- **Precision:** fraction of predicted misses that were misses in the test set.  
- **Recall:** fraction of test-set misses identified.  
- **F1:** balance between precision and recall.  

Classifier labels are generated from a synthetic risk formula, not real operating-system logs.""")
with tabs[5]:
    st.subheader("Export this simulation")
    payload={"project":"Smart Scheduler AI Lab","simulation":"single-CPU, non-preemptive educational simulator","workload_id":wid,
        "objective":goal,"recommendation":winner,"reason":reason,"risk_weight":risk,"context_switch_cost":cs,
        "metrics":results.round(4).to_dict(orient="records"),"ml_metrics":{k:float(v) for k,v in ml.items()}}
    st.download_button("⬇️ Comparison CSV",results.to_csv(index=False).encode(),"scheduler_comparison.csv","text/csv",use_container_width=True)
    st.download_button("⬇️ Experiment JSON",json.dumps(payload,indent=2).encode(),"scheduler_report.json","application/json",use_container_width=True)
    for p in ["FIFO","EDF","ML-EDF"]:
        st.download_button(f"⬇️ {p} timeline CSV",traces[p].to_csv(index=False).encode(),f"{p.lower().replace('-','_')}_timeline.csv","text/csv",key="trace_"+p)
    st.markdown("""<div class="card"><h3>Fair comparison checklist</h3><p>✓ Same workload for all schedulers</p><p>✓ Actual metrics shown beside the recommendation</p><p>✓ User chooses the optimization goal</p><p>✓ ML-EDF is not guaranteed to beat EDF or FIFO</p></div>""",unsafe_allow_html=True)

st.markdown("---")
st.markdown("<center><small>Smart Scheduler AI Lab · Synthetic data · Educational simulator · Not a production RTOS scheduler</small></center>",unsafe_allow_html=True)
