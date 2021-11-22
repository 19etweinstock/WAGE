from builtins import object
import sys
class Log(object):
  isatty = True
  def __init__(self, *args):
    self.f = open(*args)
    sys.stdout = self
    self.buff = ''
    self.isatty = True

  def isatty():
    return True

  def write(self, data):
    self.buff += data
    self.buff = self.buff.split('\r')[-1]
    output = self.buff.rsplit('\n', 1)
    if(len(output) == 2):
      self.f.write(output[0])
      self.f.write('\n')
      self.buff = output[1]
    sys.__stdout__.write(data)

  def flush(self):
    sys.__stdout__.flush()