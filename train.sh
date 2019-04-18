#!/bin/bash
box() {
    msg="* $1 *"
    echo "$msg" | sed 's/./\*/g'
    echo "$msg"
    echo "$msg" | sed 's/./\*/g'
}


main() {
    algs=()
    algs+=("--alg deepq --num_timesteps=2e6 --save_path=./models/deepq_")
    algs+=("--alg trpo_mpi --num_timesteps=1e5 --save_path=./models/trpo_mpi_")
    algs+=("--alg ppo2 --num_timesteps=3e6 --save_path=./models/ppo2_")
    algs+=("--alg a2c --num_timesteps=3e6 --save_path=./models/a2c_")

    envs=()
    envs+=("CartPole-v0")

    mkdir -p "./models"
    for env in "${envs[@]}"; do
        for alg in "${algs[@]}"; do
            cmd="python3 -m baselines.run --env=${env} ${alg}${env}.pkl"
            box "${cmd}"
            bash -c "${cmd}"
        done
    done
}

main
