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

fraction = float(sys.argv[1])

while fraction >= 0.1:
	in1 = iter(HTSeq.FastqReader(sys.argv[2]))
	in2 = iter(HTSeq.FastqReader(sys.argv[3]))
	out1 = open(sys.argv[4]+str(fraction)+'.fastq',"w")
	out2 = open(sys.argv[5]+str(fraction)+'.fastq',"w")
	for read1, read2 in itertools.izip(in1, in2):
		if random.random() < fraction:
			read1.write_to_fastq_file(out1)
			read2.write_to_fastq_file(out2)
	out1.close()
	out2.close()		
	fraction -= 0.1

