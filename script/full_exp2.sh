#!/bin/bash

bash exp2.sh /data5/wuyinjun/mimic-cxr/physionet.org/files/mimic-cxr-jpg/2.0.0/ mimic false 2 > output_mimic_2.txt 2>&1 &

bash exp2.sh /data5/wuyinjun/chexpert/CheXpert-v1.0-small/ chexpert false 2 >  output_chexpert_2.txt 2>&1 

bash exp2.sh /data5/wuyinjun/ retina false 2 > output_retina_2.txt 2>&1 



