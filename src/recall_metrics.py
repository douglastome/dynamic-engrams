'''
Copyright 2023 Douglas Feitosa Tomé

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import os

class RecallMetrics:
 def __init__(self, tp, fn, fp, tn, matches, samples):
  self.tp = tp
  self.fn = fn
  self.fp = fp
  self.tn = tn
  self.matches = matches
  self.samples = samples

 def add_tp(self, tp):
  self.tp += tp

 def add_fn(self, fn):
  self.fn += fn

 def add_fp(self, fp):
  self.fp += fp

 def add_tn(self, tn):
  self.tn +=  tn

 def add_matches(self, matches):
  self.matches += matches

 def add_samples(self, samples):
  self.samples += samples

 def get_accuracy(self):
  return self.matches / float(self.samples)

 def get_tpr(self):
  return self.tp / float(self.tp + self.fn)

 def has_fpr(self):
  return self.fp + self.tn > 0

 def get_fpr(self):
  return self.fp / float(self.fp + self.tn)

 def print_metrics(self):
  print('accuracy', self.get_accuracy())
  print('tpr', self.get_tpr())
  print('fpr', self.get_fpr())

 def save_file(self, datadir, filename):
  with open(os.path.join(datadir, filename), "w") as f:
   f.write("accuracy = %f\n"%self.get_accuracy())
   f.write("tpr = %f\n"%self.get_tpr())
   if self.has_fpr():
    f.write("fpr = %f\n"%self.get_fpr())
