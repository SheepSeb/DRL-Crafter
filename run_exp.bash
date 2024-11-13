#!/bin/bash

for i in $(seq 3); do python train.py --logdir logdir/reinforce/$i --steps 100_000 --eval-interval 10_000 --agent_type reinforce & done
for i in $(seq 3); do python train.py --logdir logdir/a2c/$i --steps 100_000 --eval-interval 10_000 --agent_type a2c & done
for i in $(seq 3); do python train.py --logdir logdir/ppo/$i --agent_type ppo_attention & done