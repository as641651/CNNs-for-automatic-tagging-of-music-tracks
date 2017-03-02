#!/bin/bash

python py-tools/eval_diff.py --l1 $1 --l2 $2 -i img_diff

#python py-tools/eval_diff.py --l1 models/C5R2M2S/waves_fft4096/02_l.json_results.json --l2 models/C5R2M2S/waves_fft4096/03_l.json_results.json -i img_diff
