import json
import re
import os
import math
from http.server import BaseHTTPRequestHandler
from rank_bm25 import BM25Okapi

SECRET_KEY = "leemai_studyleem_2025"

CONCEPT_MAP = {
    "ka":["acid dissociation constant","dissociation constant","weak acid"],
    "kb":["base dissociation constant","dissociation constant","weak base"],
    "kc":["equilibrium constant concentration","equilibrium constant"],
    "kp":["equilibrium constant pressure","equilibrium partial pressure"],
    "kw":["water dissociation constant","ionic product water"],
    "ksp":["solubility product","sparingly soluble salt"],
    "ph":["hydrogen ion concentration","acidity","pH scale"],
    "poh":["hydroxide ion concentration","basicity"],
    "le chatelier":["le chatelier principle","equilibrium shift","stress equilibrium"],
    "buffer":["buffer solution","resist pH change","weak acid salt"],
    "common ion":["common ion effect","solubility decrease","ionic equilibrium"],
    "hydrolysis":["salt hydrolysis","acidic basic salt","water reaction"],
    "titration":["acid base titration","neutralisation","equivalence point"],
    "indicator":["acid base indicator","colour change","pH indicator"],
    "oxidation":["oxidation state","electron loss","oxidising agent"],
    "reduction":["reduction electron gain","reducing agent"],
    "redox":["oxidation reduction","electron transfer","electrochemistry"],
    "electrolysis":["electrolytic cell","electrodes","anode cathode"],
    "galvanic":["galvanic cell","electrochemical cell","electrode potential"],
    "mole":["mole concept","avogadro number","molecular mass"],
    "molarity":["molar concentration","moles per litre"],
    "colligative":["colligative properties","boiling point elevation","freezing point depression"],
    "osmosis":["osmotic pressure","semi permeable membrane","solvent movement"],
    "enthalpy":["heat of reaction","delta H","exothermic endothermic"],
    "entropy":["disorder","randomness","delta S"],
    "gibbs":["Gibbs free energy","spontaneous reaction","delta G"],
    "shm":["simple harmonic motion","oscillation","periodic motion"],
    "emf":["electromotive force","potential difference","voltage"],
    "ohm":["Ohm law","resistance","current voltage"],
    "newton":["Newton law","force mass acceleration"],
    "momentum":["linear momentum","conservation momentum","mass velocity"],
    "torque":["moment of force","rotational force"],
    "doppler":["doppler effect","frequency change","moving source"],
    "radioactive":["radioactive decay","half life","nuclear radiation"],
    "fission":["nuclear fission","splitting nucleus","chain reaction"],
    "fusion":["nuclear fusion","combining nuclei","energy release"],
    "dna":["deoxyribonucleic acid","genetic material","double helix"],
    "rna":["ribonucleic acid","protein synthesis","transcription"],
    "mrna":["messenger RNA","transcription","translation"],
    "atp":["adenosine triphosphate","energy currency","cellular energy"],
    "photosynthesis":["light reaction","dark reaction","chlorophyll glucose"],
    "respiration":["cellular respiration","ATP production","glucose oxidation"],
    "mitosis":["cell division","somatic division","daughter cells identical"],
    "meiosis":["sexual reproduction","gamete formation","chromosome halved"],
    "enzyme":["biological catalyst","active site","substrate"],
    "hormone":["chemical messenger","endocrine","target organ"],
    "neuron":["nerve cell","action potential","synaptic transmission"],
    "immunity":["immune response","antibody","antigen"],
    "quadratic":["quadratic equation","ax squared","discriminant"],
    "derivative":["differentiation","rate of change","calculus"],
    "integral":["integration","antiderivative","area under curve"],
    "matrix":["matrices","determinant","inverse matrix"],
    "probability":["chance likelihood","sample space","event"],
    "logarithm":["log","exponent","natural logarithm"],
    "kya hai":["what is","define","meaning"],
    "farq":["difference","compare","distinguish"],
    "misaal":["example","instance"],
    "wajah":["reason","cause","why"],
    "tariqa":["method","process","how"],
    "qanoon":["law","rule","principle"],
    "asool":["principle","rule","law"],
}

QUESTION_PATTERNS = {
    'definition': [r'^what (is|are|was|were)\b',r'^define\b',r'^definition of\b',r'^meaning of\b',r'^what do you mean by\b'],
    'explanation':[r'^how (does|do|is|are|can|could)\b',r'^explain\b',r'^describe\b',r'^elaborate\b'],
    'reason':     [r'^why\b',r'^what (causes|makes|leads|results)\b',r'^give (the )?reason\b'],
    'list':       [r'^list\b',r'^name\b',r'^give.*(types|examples|kinds|uses|properties)\b',
                   r'^what are the (types|kinds|examples|properties|characteristics|uses|factors)\b',r'^mention\b',r'^state\b'],
    'example':    [r'^give.*(example|instance)\b',r'^example of\b',r'^examples of\b'],
    'formula':    [r'\bformula\b',r'\bequation\b',r'\bexpression for\b'],
    'comparison': [r'\bdifference between\b',r'\bcompare\b',r'\bvs\b',r'\bdistinguish\b',r'farq\b'],
    'process':    [r'^how (to|do you|can you)\b',r'\bprocess of\b',r'\bsteps\b',r'\bprocedure\b'],
}

STOPWORDS = {
    'the','a','an','is','are','was','were','in','on','at','to','for','of','and','or','but',
    'with','this','that','it','be','by','from','as','what','how','why','when','where','who',
    'which','do','does','did','have','has','had','will','would','could','should','may','might',
    'shall','can','its','their','our','your','my','his','her','we','they','i','you','he','she',
    'not','no','any','all','some','if','then','so','than','more','most','very','just','also',
    'about','into','out','up','down','over','such','each','both','after','before','please',
    'tell','me','us','give','show','kya','hai','ka','ki','ke','mein','se','ko','aur','ya','nahi',
}

class LeemAIEngine:

    def __init__(self):
        base = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base,'leemai_all_chunks.json'),'r',encoding='utf-8') as f:
            raw = json.load(f)
        self.chunks,self.meta = [],[]
        for item in raw:
            if isinstance(item,str):
                self.chunks.append(item); self.meta.append({})
            elif isinstance(item,dict):
                self.chunks.append(item.get('text','')); self.meta.append({k:v for k,v in item.items() if k!='text'})
        self.bm25 = BM25Okapi([self._tokenize(c) for c in self.chunks])
        print(f"[LeemAI] Ready — {len(self.chunks)} chunks.")

    def _stem(self,t):
        if t.endswith('ing') and len(t)>6:  return t[:-3]
        if t.endswith('tion') and len(t)>7: return t[:-4]
        if t.endswith('ity') and len(t)>6:  return t[:-3]
        if t.endswith('ies') and len(t)>5:  return t[:-3]+'y'
        if t.endswith('ed') and len(t)>5:   return t[:-2]
        if t.endswith('es') and len(t)>4:   return t[:-2]
        if t.endswith('s') and len(t)>4:    return t[:-1]
        return t

    def _tokenize(self,text):
        text = re.sub(r'[^a-z0-9\s]',' ',text.lower())
        return [self._stem(t) for t in text.split() if t not in STOPWORDS and len(t)>=2]

    def _expand(self,query):
        q = query.lower().strip()
        extra = []
        for key in sorted(CONCEPT_MAP,key=len,reverse=True):
            if key in q:
                extra.extend(CONCEPT_MAP[key])
        for word in re.sub(r'[^a-z0-9\s]',' ',q).split():
            if word in CONCEPT_MAP:
                extra.extend(CONCEPT_MAP[word])
        return list(dict.fromkeys(extra))  # deduplicated

    def _clean(self,query):
        q = re.sub(r'^(please |plz |pls )?(tell me |explain me |bata |batao )?','',query.strip(),flags=re.IGNORECASE)
        q = re.sub(r'(ke baray mein|ke baare mein|about|regarding)\s*$','',q,flags=re.IGNORECASE)
        return re.sub(r'\?+$','',q).strip() or query.strip()

    def _classify(self,query):
        q = query.lower().strip()
        for qtype,patterns in QUESTION_PATTERNS.items():
            for pat in patterns:
                if re.search(pat,q): return qtype
        return 'general'

    def _topic(self,query):
        q = re.sub(r'^(what (is|are|was|were|do you mean by)|how (does|do|is|are|can|to)|why (is|are|does)|define|explain|describe|list|name|give|tell me about|what causes|difference between|compare|formula for|equation for|example of|examples of|mention|state|elaborate|bata|batao)\s+','',query.strip(),flags=re.IGNORECASE)
        return re.sub(r'\?+$','',q).strip() or query.strip('?').strip()

    def _retrieve(self,query,top_k=6):
        clean = self._clean(query)
        extra = self._expand(clean)
        expanded = clean + ' ' + ' '.join(extra) if extra else clean

        s1 = self.bm25.get_scores(self._tokenize(clean))
        s2 = self.bm25.get_scores(self._tokenize(expanded)) if extra else s1

        combined = {i: s1[i]*0.55 + s2[i]*0.45 for i in range(len(self.chunks))}
        ranked = sorted(combined.items(),key=lambda x:-x[1])
        results = [(self.chunks[i],score,self.meta[i]) for i,score in ranked[:top_k] if score>=0.3]
        return results, extra

    def _score_sents(self,chunk,qtok,extra_tok):
        parts = re.split(r'(?<=[.!?])\s+|\n+',chunk)
        sents = [(i,s.strip()) for i,s in enumerate(parts) if len(s.strip())>20]
        if not sents: return []
        qset,eset = set(qtok),set(extra_tok or [])
        out = []
        for idx,sent in sents:
            if re.search(r'(press|publisher|journal|vol\.|doi|et al)',sent,re.I): continue
            stok = set(self._tokenize(sent))
            if not stok: continue
            p_ov = len(stok&qset)
            c_ov = len(stok&eset)
            density = (p_ov + c_ov*0.5)/(len(stok)+1)
            wc = len(sent.split())
            lscore = 1.0 if 12<=wc<=50 else (wc/12 if wc<12 else min(50/wc,1.0))
            pscore = 1/math.log(idx+2)
            score = p_ov*3.0 + c_ov*1.5 + density*2.5 + lscore*0.7 + pscore*0.5
            out.append((score,idx,sent))
        out.sort(key=lambda x:-x[0])
        return out

    def _synthesise(self,results,query,extra_tok,n=4):
        qtok = self._tokenize(self._clean(query))
        pool = []
        for rank,(chunk,bscore,_) in enumerate(results):
            boost = math.log(bscore+1)/math.log(rank+2)
            for score,idx,sent in self._score_sents(chunk,qtok,extra_tok):
                pool.append((score*boost,rank,idx,sent))
        pool.sort(key=lambda x:-x[0])
        seen,selected = set(),[]
        for score,rank,idx,sent in pool:
            sig = sent[:70].lower()
            if sig not in seen and score>0.2:
                seen.add(sig); selected.append((rank,idx,sent))
            if len(selected)>=n: break
        selected.sort(key=lambda x:(x[0],x[1]))
        return [s for _,_,s in selected]

    def _list_items(self,results,qtok,extra_tok):
        items = []
        for chunk,_,_ in results[:2]:
            for line in chunk.split('\n'):
                line = line.strip()
                if re.match(r'^(\d+[\.\):]|[-–•*]|\([a-z]\))\s+\w',line) and len(line)>15:
                    items.append(re.sub(r'^(\d+[\.\):]|[-–•*]|\([a-z]\))\s+','',line))
        if not items:
            items = [s for _,_,s in self._score_sents(results[0][0],qtok,extra_tok)[:6]]
        return items[:7]

    def _format(self,qtype,topic,content,meta):
        title  = topic.title() if topic else ''
        chtag  = f'\n\n_Chapter: {meta["chapter"]}_' if meta and meta.get('chapter') else ''
        subjtag= f' ({meta["subject"].title()} — Class {meta["class"]})' if meta and meta.get('subject') and meta.get('class') else ''
        body   = ' '.join(content)
        t = {
            'definition': f'**{title}**{subjtag}\n\n{body}{chtag}',
            'explanation':f'**{title} — Explanation:**\n\n{body}{chtag}',
            'reason':     f'**Why {topic}:**\n\n{body}{chtag}',
            'formula':    f'**Formula — {title}:**\n\n{body}{chtag}',
            'comparison': f'**Comparison — {title}:**\n\n{body}{chtag}',
            'process':    f'**Steps — {title}:**\n\n{body}{chtag}',
            'example':    f'**Examples of {topic}:**\n\n{body}{chtag}',
            'list':       f'**{title}:**\n\n'+'\n'.join(f'• {s}' for s in content)+chtag,
            'general':    body+chtag,
        }
        return t.get(qtype, body+chtag)

    def answer(self,query):
        query = query.strip()
        if not query:
            return {'answer':'Please ask a question.','confidence':0,'type':'error','topic':''}
        topic  = self._topic(query)
        qtype  = self._classify(query)
        qtok   = self._tokenize(self._clean(query))
        if not qtok:
            return {'answer':'Please ask a more specific question.','confidence':0,'type':'error','topic':topic}

        results,extra = self._retrieve(query,top_k=6)
        if not results:
            results,extra = self._retrieve(topic,top_k=3)
        if not results:
            return {'answer':f"I don't have information about **{topic}** yet.\nPlease check your textbook.",'confidence':0,'type':'not_found','topic':topic}

        top_meta = results[0][2]
        if qtype=='list':
            items = self._list_items(results,qtok,extra)
            answer_text = self._format(qtype,topic,items,top_meta)
        else:
            n = 5 if qtype in ('explanation','process','comparison') else 3
            sents = self._synthesise(results,query,extra,n=n)
            answer_text = self._format(qtype,topic,sents or [results[0][0][:500]],top_meta)

        raw = results[0][1]
        conf = min(int((raw/10.0)*80) + min(int(len(results)/6*15),15) + (10 if extra else 0), 95)
        return {'answer':answer_text,'confidence':conf,'type':qtype,'topic':topic,'expanded':bool(extra)}


_engine = None
def get_engine():
    global _engine
    if _engine is None: _engine = LeemAIEngine()
    return _engine
try: get_engine()
except Exception as e: print(f"[LeemAI] Init error: {e}")


class handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(200); self._cors(); self.end_headers()

    def do_GET(self):
        self._send(200,{'status':'LeemAI is running','version':'3.0','engine':'BM25 + Query Expansion + Answer Synthesis'})

    def do_POST(self):
        try:
            if self.headers.get('x-api-key','') != SECRET_KEY:
                self._send(401,{'error':'Unauthorized'}); return
            length = int(self.headers.get('Content-Length',0))
            if not length:
                self._send(400,{'error':'Empty body'}); return
            body = self.rfile.read(length)
            try: data = json.loads(body)
            except: self._send(400,{'error':'Invalid JSON'}); return
            query = (data.get('query') or '').strip()
            if not query: self._send(400,{'error':'No query'}); return
            if len(query)>500: self._send(400,{'error':'Query too long'}); return
            self._send(200, get_engine().answer(query))
        except Exception as e:
            self._send(500,{'error':'Server error','detail':str(e)})

    def _send(self,code,data):
        body = json.dumps(data,ensure_ascii=False).encode('utf-8')
        self.send_response(code); self._cors()
        self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin','*')
        self.send_header('Access-Control-Allow-Methods','GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers','Content-Type, x-api-key')

    def log_message(self,fmt,*args): pass
