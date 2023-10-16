import csv
import logging
import time
from itertools import chain, combinations
from pathlib import Path
from typing import Any, List, Tuple, Union
from node import TreeNode
from load import *

# 去除掉出現次數不足的產品
def prunCandidate(candidate: dict, minSup: int)->dict:
  result = {key:value for key, value in candidate.items() if value >= minSup} #篩選出count>minSu的item
  del candidate #刪除不要的變數空間
  return {key:value for key, value in sorted(result.items(), key=lambda item:item[1], reverse=True)}

#建HaderTable
def createHaderTable(priority_table: dict)->dict:
  haderTable = {}
  for key, item in priority_table.items():
    haderTable[key] = [item, None, None] #hashTable[0]:總數 hashTable[1]:指標指向第一個TreeNode hashTable[2]指向currentNode節省搜尋時間
  del priority_table #刪除變數空間
  return haderTable

#遞迴更新子樹  
def updateTree(tran, node, haderTable, count):
  if tran[0] in node.children.keys(): 
    node.children[tran[0]].add(count) #如果第一個item有存在node的child，此childNode的count+1
  else: 
    newNode = TreeNode(tran[0], count, node) #反之則新增另一個分支
    node.children[tran[0]] = newNode #並把newNode加在node的children中
    if haderTable[tran[0]][1] == None:  #當haderTable尚未指向第一個tree node 時
      haderTable[tran[0]][1] = newNode  #指向新增的node
    else:
      haderTable[tran[0]][2].next = newNode 
    haderTable[tran[0]][2] = newNode #指向當前newNode
  if len(tran) != 1: #代表該筆transaction後面還有東西
    updateTree(tran[1:], node.children[tran[0]], haderTable, count) #遞迴更新

#建立子樹
def createFPTree(transactions: list[list[str]], haderTble: dict, frequency: list[int]):
  root = TreeNode('{}', 1, None)
  for idx, tran in enumerate(transactions):
    tran = [item for item in tran if item in haderTble.keys()]
    if len(tran) != 0:
        tran = sorted(tran, key=lambda item:haderTble[item][0], reverse=True)
        updateTree(tran, root, haderTble, frequency[idx])
  return root

#print tree
def printTree(node, indent=''):
    if node:
      parent_item = node.parentNode.nodeName if node.parentNode else 'None'
      print(f"{indent}{node.nodeName} ({node.count}) Parent: {parent_item}")
      for child in node.children.values():
          printTree(child, indent + "  ")

#find pattern base and frequency
def conPatternBase(item, headerTable):
  node = headerTable[item][1]
  patten_base = []
  patten_base_freq = []
  while node:
    prefix = []
    parent_node = node.parentNode
    while parent_node.nodeName != "{}":
      prefix.append(parent_node.nodeName)
      parent_node = parent_node.parentNode
    if len(prefix) >= 1:
      patten_base.append(prefix[::-1])
      patten_base_freq.append(node.count)
    node = node.next
  return patten_base, patten_base_freq

#Create conditional FPTree
def createConditionalTree(pattern_base,pattern_base_freq, minSup):
  table = {}
  for idx in range(len(pattern_base)):
    for item in pattern_base[idx]:
      if item in table.keys():
        table[item] += pattern_base_freq[idx]
      else:
        table[item] = pattern_base_freq[idx]
  #去掉table中不符合minSup的item
  table = dict((item, sup) for item, sup in table.items() if sup >= minSup)
  if (len(table) == 0):
    return None, None
  #將table轉為可建tree的形式
  for key in table.keys():
    table[key] = [table[key], None, None]
  #建Tree
  conditionalTree = createFPTree(pattern_base, table, pattern_base_freq)
  return conditionalTree, table

#Mine Tree
def mineTree(headerTable, minSup, prefix, freqItemSet, freqItemsetWithCount):
  sorted_item = [item[0] for item in sorted(list(headerTable.items()), key=lambda p:p[1][0])]
  for item in sorted_item:
    freqSet = prefix.copy()
    freqSet.add(item)
    freqItemSet.append(freqSet)
    freqItemsetWithCount[frozenset(freqSet)] = headerTable[item][0]
    # 找conditional pattern base
    pattern_base, pattern_base_freq = conPatternBase(item, headerTable)
    conditionalTree, newHeaderTable = createConditionalTree(pattern_base,pattern_base_freq, minSup)
    if newHeaderTable != None:
      mineTree(newHeaderTable, minSup, freqSet, freqItemSet, freqItemsetWithCount)
# power set
def powerSet(itemSet):
  return chain.from_iterable(combinations(itemSet, r) for r in range(1, len(itemSet)))

# 計算itemset出現總次數
def countSup(itemSet, transactions):
  count = 0
  for tran in transactions:
    if set(tran) & itemSet == itemSet:
      count +=1
  return count

#association rule
def associationRule(freqItemSet, transactions, minConf, freqItemsetWithCount):
  rules = []
  for itemSet in freqItemSet:
    subSets = powerSet(itemSet)
    set_support = freqItemsetWithCount[frozenset(itemSet)]

    for s in subSets:
      subSet = set(s)
      x = freqItemsetWithCount[frozenset(subSet)]
      y = freqItemsetWithCount[frozenset(itemSet - subSet)]
      confidence = float(set_support / x)
      lift = float(set_support * len(transactions) / (x*y))
      if confidence >= minConf:
        rules.append([subSet, (itemSet - subSet), float(set_support / len(transactions)), confidence, lift])
  return rules


# Apriori functions
def turn_two_array(L1, minSup):
  freq_itemSet = []
  frequency = []
  for key, val in L1.items():
    if val >= minSup:
      freq_itemSet.append({key})
      frequency.append(val)
  return freq_itemSet, frequency

#scan db to count the total number of item
def scanTransaction(transactions:list[list[str]], scanSet:set[str]):
  count = 0
  for tran in transactions:
    if len(set(tran) & scanSet) == len(scanSet):
      count+=1
  return count

#generate candidate
def generateCandidate(transactions, l, level):
  freq_itemSet = []
  frequency = []
  if len(l) <= 1:
    return freq_itemSet, frequency
  for i in range(len(l)):
    for j in range(i+1, len(l)):
      if len(l[i] & l[j]) == level-2 and (l[i] | l[j]) not in freq_itemSet:
        freq_itemSet.append(l[i] | l[j])
        sup = scanTransaction(transactions, (l[i] | l[j]))
        frequency.append(sup)
  return freq_itemSet, frequency

#generate local
def generateLocal(freq_itemSet, frequency, minSup):
  if len(freq_itemSet) == 0:
    return [], []
  new_freq_itemSet = [freq_itemSet[i] for i in range(len(frequency)) if frequency[i] >= minSup]
  new_frequency = [frequency[i] for i in range(len(frequency)) if frequency[i] >= minSup]
  return new_freq_itemSet, new_frequency

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        print(f"Running {func.__name__} ...", end='\r')
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} Done in {end - start:.2f} seconds")
        return result
    return wrapper

@timer
def write_file(data: List[Tuple[Any]], filename: Union[str, Path]) -> None:
    """write_file writes the data to a csv file and
    adds a header row with `antecedent`, `consequent`, `support`, `confidence`, `lift`.
    You can revise the function but please make sure your output is the same as ours
    (See `outputs/example-fp_growth-0.1-0.3.csv` for example).

    Args:
        data (List[Tuple[Any]]): The data to write to the file
        filename (Union[str, Path]): The filename to write to
    """
    proc_data = []
    for rule in data:
        PREC = 2
        a, c, sup, conf, lift = rule
        a, c = list(set(a)), list(set(c))
        sup = round(sup, PREC)
        conf = round(conf, PREC)
        lift = round(lift, PREC)
        proc_data.append([a, c, sup, conf, lift])

    with open(filename, 'w') as f:
        writer = csv.writer(f)
        # rule format: antecedent --> consequent
        writer.writerow(["antecedent", "consequent",
                        "support", "confidence", "lift"])
        writer.writerows(proc_data)

    with open(filename, 'w') as f:
        writer = csv.writer(f)
        # rule format: antecedent --> consequent
        writer.writerow(["antecedent", "consequent",
                        "support", "confidence", "lift"])
        writer.writerows(proc_data)




def setup_logger():
    l = logging.getLogger('l')

    log_dir: Path = Path(__file__).parent / "logs"

    # create log directory if not exists
    log_dir.mkdir(parents=True, exist_ok=True)

    # set log file name
    log_file_name = f"{time.strftime('%Y%m%d_%H%M%S')}.log"

    l.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler(
        filename=log_dir / log_file_name,
        mode='w'
    )
    streamHandler = logging.StreamHandler()

    allFormatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s: %(message)s"
    )

    fileHandler.setFormatter(allFormatter)
    fileHandler.setLevel(logging.INFO)

    streamHandler.setFormatter(allFormatter)
    streamHandler.setLevel(logging.INFO)

    l.addHandler(streamHandler)
    l.addHandler(fileHandler)

    return l

l = setup_logger()

# fpgrowth
def FPG(data_path, minSupRatio, minConfRatio):
    transcations, itemSet = DataLoader(data_path) #read data and turn into transaction(list[list[str]]) and itemSet with total number
    min_sup = len(transcations) * minSupRatio
    one_items = prunCandidate(itemSet, min_sup) # select item which is larger than minsup and sort it
    haderTable = createHaderTable(one_items) # create hader table
    frequency = [1 for i in range(len(transcations))]
    FPtree = createFPTree(transcations, haderTable, frequency)
    freqItemList = []
    freqItemsetWithCount = {}
    mineTree(haderTable, min_sup, set(), freqItemList, freqItemsetWithCount)
    rules = associationRule(freqItemList, transcations, minConfRatio, freqItemsetWithCount)
    print(len(freqItemList))
    print(len(rules))
    return rules

#apriori
def Apriori(data_path, minSupRatio, minConfRatio):
    freqItemList = []
    itemSet_freq = []
    #load data
    transcations, one_itemset = DataLoader(data_path)
    min_sup = len(transcations) * minSupRatio
    #purn candidate
    L1 = prunCandidate(one_itemset, min_sup)
    print(len(L1))
    local, local_freq = turn_two_array(L1, min_sup)
    freqItemList = freqItemList + local
    itemSet_freq = itemSet_freq + local_freq
    level = 2
    while(len(local) != 0):
        candidate, candidate_freq = generateCandidate(transcations, local, level)
        local, local_freq = generateLocal(candidate, candidate_freq, min_sup)
        if len(local) != 0:
            freqItemList = freqItemList + local
            itemSet_freq = itemSet_freq + local_freq
        level += 1
    freqItemList = freqItemList + local
    freqItemsetWithCount = {}
    for freqSet in freqItemList:
      freqItemsetWithCount[frozenset(freqSet)] = countSup(freqSet, transcations)
    rules = associationRule(freqItemList, transcations, minConfRatio, freqItemsetWithCount)
    print(len(freqItemList))
    print(len(rules))
    return rules