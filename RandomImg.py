#!/usr/bin/python

import os
import sys
#import math
import random
import shutil

src = sys.argv[1]
dst = sys.argv[2]
count = sys.argv[3]

#random.seed()
sysRand = random.SystemRandom()

for index in range(int(count)):
  img = sysRand.choice(os.listdir(src))
  shutil.copy2(os.path.join(src, img), dst)
#  os.remove(os.path.join(dst, img))
