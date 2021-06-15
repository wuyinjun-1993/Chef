#!/bin/bash

bash exp1.sh /data5/wuyinjun/mimic-cxr/physionet.org/files/mimic-cxr-jpg/2.0.0/ mimic true 2 > output_mimic.txt 2>&1

bash exp1.sh /data5/wuyinjun/chexpert/CheXpert-v1.0-small/ chexpert true 2 >  output_chexpert.txt 2>&1

bash exp1.sh /data5/wuyinjun/ retina true 2 > output_retina.txt 2>&1



