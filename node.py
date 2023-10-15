class TreeNode():
  def __init__(self, nodeName, num, parentNode):
    self.nodeName = nodeName
    self.next = None
    self.children = {}
    self.count = num
    self.parentNode = parentNode
  
  def add(self, num):
    self.count += num

