#!/bin/bash
{"requirements": {"ram": 3, "vram": 10, "ramp_up_time": 10}}

# Altere para o diretório onde está o docker-compose.yml
cd /raid/aluno_marcospaulo/dreamer_dgx && docker-compose up --build
