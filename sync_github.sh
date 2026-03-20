#!/bin/bash
cd /home/konglz/DiffTumor/STEP1.AutoencoderModel
git add .
git commit -m "更新代码：$(date +%Y-%m-%d_%H:%M:%S)"
git push origin main
