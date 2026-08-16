for seed in {1..5}
do
    python -m src.run_experiment navigation maddpg --seeds 1 --seed-start $seed --render
done

for seed in {1..5}
do
    python -m src.run_experiment balance maddpg --seeds 1 --seed-start $seed --render
done

for seed in {1..5}
do
    python -m src.run_experiment buzz_wire maddpg --seeds 1 --seed-start $seed --render
done

for seed in {1..5}
do
    python -m src.run_experiment transport maddpg --seeds 1 --seed-start $seed --render
done