#!/bin/bash

# which Q to use?
#$ -q free*

# the qstat job name
#$ -N HQC

# use the real bash shell
#$ -S /bin/bash

# mail me ...
#$ -M dakex@uci.edu

# ... when the job (b)egins, (e)nds
#$ -m e

echo "Job begins"

date

python HQC2.py

date
echo "Job ends"
