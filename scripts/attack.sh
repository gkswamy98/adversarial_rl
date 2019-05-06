#!/bin/bash
box() {
    msg="* $1 *"
    echo "$msg" | sed 's/./\*/g'
    echo "$msg"
    echo "$msg" | sed 's/./\*/g'
}

main() {
    algs=()
    algs+=("a2c")
    algs+=("deepq")
    algs+=("ppo2")
    algs+=("trpo_mpi")

    envs=()
    envs+=("CartPole-v0")
    envs+=("Acrobot-v1")

    attacks=()
    attacks+=("random")
    attacks+=("fgsm")

    norms=()
    norms+=("1")
    norms+=("2")

    eps=()
    eps+=("0.01")
    eps+=("0.1")
    eps+=("0.3")
    eps+=("0.5")

    for env in "${envs[@]}"; do
        for alg in "${algs[@]}"; do
        	for attack in "${attacks[@]}"; do
        		for norm in "${norms[@]}"; do
        			for ep in "${eps[@]}"; do
        				cmd="python3 adv/main.py --alg=${alg} --env=${env} --eps=${ep} --attack=${attack} --attack_ord=${norm} nop"
            			box "${cmd}"
            			bash -c "${cmd}"
            		done
            	done
            done
        done
    done
}

main