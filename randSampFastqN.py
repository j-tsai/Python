#!/usr/bin/python

import sys
if len(sys.argv) < 5:
	print 'python randSampFastq.py fraction fastq_R1 fastq_R2 out_fastq_R1 out_fastq_R2'
	sys.exit()

import random, itertools
import HTSeq

#try:
#	sys.argv[1:]
#except IndexError:

fraction0 = 1 - float(sys.argv[1])
#fraction = 1 - fraction0
in1_0 = sys.argv[2]
in2_0 = sys.argv[3]
j = 0
while j <= 20:
	in1 = iter(HTSeq.FastqReader(in1_0))
	in2 = iter(HTSeq.FastqReader(in2_0))
	out1_0 = sys.argv[4]+str(fraction0)+ str(j) + '.fastq'
	out2_0 = sys.argv[5]+str(fraction0)+ str(j) + '.fastq'
	out1 = open(out1_0,"w")
	out2 = open(out2_0,"w")
	for read1, read2 in itertools.izip(in1, in2):
		if random.random() < fraction0:
			read1.write_to_fastq_file(out1)
			read2.write_to_fastq_file(out2)
	out1.close()
	out2.close()		
	in1_0 = out1_0
	in2_0 = out2_0
	j+=1
	#fraction -= fraction0

