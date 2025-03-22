#!/bin/bash
# {"requirements": {"ram": 2, "vram": 0, "ramp_up_time": 10}}

# Altere para o diretório onde está o docker-compose.yml
cd /raid/aluno_marcospaulo/dreamer_dgx && docker-compose up --build
