from builtins import object
import sys
class Log(object):
  def __init__(self, *args):
    self.f = open(*args)
    sys.stdout = self
    self.buff = ''

  def isatty(self):
    return True

  def write(self, data):
    self.buff += data
    self.buff = self.buff.rsplit('\r', 1)[-1]
    output = self.buff.rsplit('\n', 1)
    if(len(output) == 2):
      self.f.write(output[0])
      self.f.write('\n')
      self.buff = output[1]
    sys.__stdout__.write(data)

  def flush(self):
    sys.__stdout__.flush()