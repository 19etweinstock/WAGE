from builtins import object
import sys
class Log(object):
  def __init__(self, *args):
    self.f = open(*args)
    sys.stdout = self
    self.buff = ''

  # trick tf into thinking it is still writing to console
  def isatty(self):
    return True

  def write(self, data):
    # implement 'isatty' behavior for the file
    self.buff += data
    # if there is a line feed clear the buffer
    self.buff = self.buff.rsplit('\r', 1)[-1]
    # if a new line has been printed, we need to update the buffer and print to the file
    output = self.buff.rsplit('\n', 1)
    if(len(output) == 2):
      self.f.write(output[0])
      self.f.write('\n')
      self.buff = output[1]
    sys.__stdout__.write(data)

  def flush(self):
    sys.__stdout__.flush()

  def __del__(self):
    self.f.close()
    sys.stdout=sys.__stdout__