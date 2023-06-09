# coding = utf-8

import os, re, sys, random, urllib.parse, json
from collections import defaultdict

def WriteLine(fout, lst):
	fout.write('\t'.join([str(x) for x in lst]) + '\n')

def RM(patt, sr):
	mat = re.search(patt, sr, re.DOTALL | re.MULTILINE)
	return mat.group(1) if mat else ''

try: import requests
except: pass
def GetPage(url, cookie='', proxy='', timeout=5):
	try:
		headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36'}
		if cookie != '': headers['cookie'] = cookie
		if proxy != '': 
			proxies = {'http': proxy, 'https': proxy}
			resp = requests.get(url, headers=headers, proxies=proxies, timeout=timeout)
		else: resp = requests.get(url, headers=headers, timeout=timeout)
		content = resp.content
		headc = content[:min([3000,len(content)])].decode(errors='ignore')
		charset = RM('charset="?([-a-zA-Z0-9]+)', headc)
		if charset == '': charset = 'utf-8'
		content = content.decode(charset, errors='replace')
	except Exception as e:
		print(e)
		content = ''
	return content

def ParseHttpHeaders(headerstr='clip'):
	if headerstr == 'clip':
		import pyperclip
		headerstr = pyperclip.paste()
	return {k:v.strip() for k,v in [x.split(':', 1) for x in headerstr.split('\n') if ':' in x]}

def GetJson(url, cookie='', proxy='', timeout=5.0):
	try:
		headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36'}
		if cookie != '': headers['cookie'] = cookie
		if proxy != '': 
			proxies = {'http': proxy, 'https': proxy}
			resp = requests.get(url, headers=headers, proxies=proxies, timeout=timeout)
		else: resp = requests.get(url, headers=headers, timeout=timeout)
		return resp.json() 
	except Exception as e:
		print(e)
		content = {}
	return content

def FindAllHrefs(url, content=None, regex=''):
	ret = set()
	if content == None: content = GetPage(url)
	patt = re.compile('href="?([a-zA-Z0-9-_:/.%]+)')
	for xx in re.findall(patt, content):
		ret.add( urllib.parse.urljoin(url, xx) )
	if regex != '': ret = (x for x in ret if re.match(regex, x))
	return list(ret)

def Translate(txt):
	postdata = {'from': 'en', 'to': 'zh', 'transtype': 'realtime', 'query': txt}
	url = "http://fanyi.baidu.com/v2transapi"
	try:
		resp = requests.post(url, data=postdata, 
					   headers={'Referer': 'http://fanyi.baidu.com/'})
		ret = resp.json()
		ret = ret['trans_result']['data'][0]['dst']
	except Exception as e:
		print(e)
		ret = ''
	return ret

def IsChsStr(z):
	return re.search('^[\u4e00-\u9fa5]+$', z) is not None

def FreqDict2List(dt):
	return sorted(dt.items(), key=lambda d:d[-1], reverse=True)

def SelectRowsbyCol(fn, ofn, st, num = 0):
	with open(fn, encoding = "utf-8") as fin:
		with open(ofn, "w", encoding = "utf-8") as fout:
			for line in (ll for ll in fin.read().split('\n') if ll != ""):
				if line.split('\t')[num] in st:
					fout.write(line + '\n')

def MergeFiles(dir, objfile, regstr = ".*"):
	with open(objfile, "w", encoding = "utf-8") as fout:
		for file in os.listdir(dir):
			if re.match(regstr, file):
				with open(os.path.join(dir, file), encoding = "utf-8") as filein:
					fout.write(filein.read())

def JoinFiles(fnx, fny, ofn):
	with open(fnx, encoding = "utf-8") as fin:
		lx = [vv for vv in fin.read().split('\n') if vv != ""]
	with open(fny, encoding = "utf-8") as fin:
		ly = [vv for vv in fin.read().split('\n') if vv != ""]
	with open(ofn, "w", encoding = "utf-8") as fout:
		for i in range(min(len(lx), len(ly))):
			fout.write(lx[i] + "\t" + ly[i] + "\n")

				
def RemoveDupRows(file, fobj='*'):
	st = set()
	if fobj == '*': fobj = file
	with open(file, encoding = "utf-8") as fin:
		for line in fin.read().split('\n'):
			if line == "": continue
			st.add(line)
	with open(fobj, "w", encoding = "utf-8") as fout:
		for line in st:
			fout.write(line + '\n')
			
def LoadCSV(fn):
	ret = []
	with open(fn, encoding='utf-8') as fin:
		for line in fin:
			lln = line.rstrip('\r\n').split('\t')
			ret.append(lln)
	return ret

def LoadCSVg(fn):
	with open(fn, encoding='utf-8') as fin:
		for line in fin:
			lln = line.rstrip('\r\n').split('\t')
			yield lln

def SaveCSV(csv, fn):
	with open(fn, 'w', encoding='utf-8') as fout:
		for x in csv:
			WriteLine(fout, x)

def SplitTables(fn, limit=3):
	rst = set()
	with open(fn, encoding='utf-8') as fin:
		for line in fin:
			lln = line.rstrip('\r\n').split('\t')
			rst.add(len(lln))
	if len(rst) > limit: 
		print('%d tables, exceed limit %d' % (len(rst), limit))
		return
	for ii in rst:
		print('%d columns' % ii)
		with open(fn.replace('.txt', '') + '.split.%d.txt' % ii, 'w', encoding='utf-8') as fout:
			with open(fn, encoding='utf-8') as fin:
				for line in fin:
					lln = line.rstrip('\r\n').split('\t')
					if len(lln) == ii:
						fout.write(line)

def LoadSet(fn):
	with open(fn, encoding="utf-8") as fin:
		st = set(ll for ll in fin.read().split('\n') if ll != "")
	return st

def LoadList(fn):
	with open(fn, encoding="utf-8") as fin:
		st = list(ll for ll in fin.read().split('\n') if ll != "")
	return st

def LoadJsonsg(fn): return map(json.loads, LoadListg(fn))
def LoadJsons(fn): return list(LoadJsonsg(fn))

def LoadListg(fn):
	with open(fn, encoding="utf-8") as fin:
		for ll in fin:
			ll = ll.strip()
			if ll != '': yield ll

def LoadDict(fn, func=str):
	dict = {}
	with open(fn, encoding = "utf-8") as fin:
		for lv in (ll.split('\t', 1) for ll in fin.read().split('\n') if ll != ""):
			dict[lv[0]] = func(lv[1])
	return dict

def SaveDict(dict, ofn, output0 = True):
	with open(ofn, "w", encoding = "utf-8") as fout:
		for k in dict.keys():
			if output0 or dict[k] != 0:
				fout.write(str(k) + "\t" + str(dict[k]) + "\n")

def SaveList(st, ofn):
	with open(ofn, "w", encoding = "utf-8") as fout:
		for k in st:
			fout.write(str(k) + "\n")
		
def SaveJsons(st, ofn): return SaveList([json.dumps(x, ensure_ascii=False) for x in st], ofn)
		
def ListDirFiles(dir, filter=None):
	if filter is None: 
		return [os.path.join(dir, x) for x in os.listdir(dir)]
	return [os.path.join(dir, x) for x in os.listdir(dir) if filter(x)]

def ProcessDir(dir, func, param):
	for file in os.listdir(dir):
		print(file)
		func(os.path.join(dir, file), param)

def GetLines(fn):
	with open(fn, encoding = "utf-8", errors = 'ignore') as fin:
		lines = list(map(str.strip, fin.readlines()))
	return lines
				
def SortRows(file, fobj, cid, type=int, rev = True):
	lines = LoadCSV(file)
	dat = []
	for dv in lines:
		if len(dv) <= cid: continue
		dat.append((type(dv[cid]), dv))
	with open(fobj, "w", encoding = "utf-8") as fout:
		for dd in sorted(dat, reverse = rev):
			fout.write('\t'.join(dd[1]) + '\n')

def SampleRows(file, fobj, num):
	zz = list(open(file, encoding='utf-8'))
	num = min([num, len(zz)])
	zz = random.sample(zz, num)
	with open(fobj, 'w', encoding='utf-8') as fout:
		for xx in zz: fout.write(xx)

def SetProduct(file1, file2, fobj):
	l1, l2 = GetLines(file1), GetLines(file2)
	with open(fobj, 'w', encoding='utf-8') as fout:
		for z1 in l1:
			for z2 in l2:
				fout.write(z1 + z2 + '\n')

class TokenList:
	def __init__(self, file, low_freq=2, source=None, func=None, save_low_freq=2, special_marks=[]):
		if not os.path.exists(file):
			tdict = defaultdict(int)
			if special_marks is not None:
				for i, xx in enumerate(special_marks): tdict[xx] = 100000000 - i
			for xx in source:
				for token in func(xx): tdict[token] += 1
			tokens = FreqDict2List(tdict)
			tokens = [x for x in tokens if x[1] >= save_low_freq]
			SaveCSV(tokens, file)
		self.id2t = [x for x,y in LoadCSV(file) if float(y) >= low_freq]
		if special_marks is not None: self.id2t = ['<PAD>', '<UNK>'] + self.id2t
		self.t2id = {v:k for k,v in enumerate(self.id2t)}
	def get_id(self, token): return self.t2id.get(token, 1)
	def get_token(self, ii): return self.id2t[ii]
	def get_num(self): return len(self.id2t)

def CalcF1(correct, output, golden):
	prec = correct / max(output, 1);  reca = correct / max(golden, 1);
	f1 = 2 * prec * reca / max(1e-9, prec + reca)
	pstr = 'Prec: %.4f %d/%d, Reca: %.4f %d/%d, F1: %.4f' % (prec, correct, output, reca, correct, golden, f1)
	return pstr

def Upgradeljqpy(url=None):
	if url is None: url = 'http://kw.fudan.edu.cn/resources/ljqpy.py'
	dirs = [dir for dir in reversed(sys.path) if os.path.isdir(dir) and 'ljqpy.py' in os.listdir(dir)]
	if len(dirs) == 0: raise Exception("package directory no found")
	dir = dirs[0]
	print('downloading ljqpy.py from %s to %s' % (url, dir))
	resp = requests.get(url)
	if b'Upgradeljqpy' not in resp.content: raise Exception('bad file')
	with open(os.path.join(dir, 'ljqpy.py'), 'wb') as fout:
		fout.write(resp.content)
	print('success')

def sql(cmd=''):
	if cmd == '': cmd = input("> ")
	cts = [x for x in cmd.strip().lower()]
	instr = False
	for i in range(len(cts)):
		if cts[i] == '"' and cts[i-1] != '\\': instr = not instr
		if cts[i] == ' ' and instr: cts[i] = "&nbsp;"
	cmds = "".join(cts).split(' ')
	keyw = { 'select', 'from', 'to', 'where' }
	ct, kn = {}, ''
	for xx in cmds:
		if xx in keyw: kn = xx
		else: ct[kn] = ct.get(kn, "") + " " + xx

	for xx in ct.keys():
		ct[xx] = ct[xx].replace("&nbsp;", " ").strip()

	if ct.get('where', "") == "": ct['where'] = 'True'

	if os.path.isdir(ct['from']): fl = [os.path.join(ct['from'], x) for x in os.listdir(ct['from'])]
	else: fl = ct['from'].split('+')

	if ct.get('to', "") == "": ct['to'] = 'temp.txt'

	for xx in ct.keys():
		print(xx + " : " + ct[xx])

	total = 0
	with open(ct['to'], 'w', encoding = 'utf-8') as fout:
		for fn in fl:
			print('selecting ' + fn)
			for xx in open(fn, encoding = 'utf-8'):
				x = xx.rstrip('\r\n').split('\t')
				if eval(ct['where'], {'x':x}):
					if ct['select'] == '*': res = "\t".join(x) + '\n'
					else: res = "\t".join(eval('[' + ct['select'] + ']')) + '\n'
					fout.write(res)
					total += 1

	print('completed, ' + str(total) + " records")

def cmd():
	while True:
		cmd = input("> ")
		sql(cmd)

#计算最终预测三元组的p、r、f1
def predictMetric(tdata,save_dir):
    truth_num,preds_num,pred_correct=0,0,0

    for data in tdata:
        truth=data.get('relationMentions',[])
        preds=data.get('preds',[])

        truth_num+=len(truth)
        preds_num+=len(preds)

        t_list=[[item['label'],item['em1Text'],item['em2Text']] for item in truth]
        print('\n'+'-'*100)
        print(t_list)
        print('rc_pred ',data['rc_pred'])
        print('-'*20)

        for item in preds:
            label,pred_em1Text,pred_em2Text=item['label'],item['em1Text'],item['em2Text']
            if [label,pred_em1Text,pred_em2Text] in t_list:
                print([label,pred_em1Text,pred_em2Text],'\tyes')
                pred_correct+=1
            else:print([label,pred_em1Text,pred_em2Text],'\tno')
    p=pred_correct/preds_num
    r=pred_correct/truth_num
    f1=2*p*r/(p+r) if p+r !=0 else 0
    print(p,r,f1)
    with open(save_dir+'/metrics.txt','a+',encoding='utf-8') as f:
        f.write('precision:{:.4f}\trecall:{:.4f}\tF1:{:.4f}\n'.format(p,r,f1))
        f.write('-' * 50 + '\n')

    return


def modifyData():
    if z.get('ys',[])!=[]:
        z['relationMentions']=[]
        for item in z['ys']:
            p_o=item.get('c')
            if p_o.get('s',[])!=[]:
                for o in item.get('rval',''):
                    spo={'label':p_o.get('p',''),'em1Text':p_o.get('s',''),'em2Text':o}
                    z['relationMentions'].append(spo)
        z.pop('ys')

def getPointer(tokens,out):
    l_tokens=len(tokens)
    out=out[:l_tokens,:l_tokens]
    ents=(out>0).float().nonzero()
    results=[]
    for i in range(ents.size(0)):
        results.append(''.join(tokens[ents[i,0]:ents[i,1]+1]))
    return results

def add_rc_pred(tdata):
    for item in tdata:
        if 'spo_list' in item:
            spo_list = item.get('spo_list', [])
        else:spo_list = item.get('relationMentions', [])
        labels = list(set(x['label'] for x in spo_list))
        item['rc_pred']=labels
