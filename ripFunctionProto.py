#!/usr/bin/python

import sys

isMultiline = False

string = sys.argv[3]
end = len(string)

with open(sys.argv[1], 'r') as infile:
  with open(sys.argv[2], 'w') as outfile:

    for line in infile:
      if len(line) >= 2 and (line[-2] == ';' or line[-2] == '{'):
        isMultiline = False
        continue
      if (line[0:end] == string or isMultiline):
        outfile.write(line)
        isMultiline = True
