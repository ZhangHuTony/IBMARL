for seed in {1..5}
do
    python -m src.run_experiment navigation maddpg --seed $seed --render
done

for seed in {1..5}
do
    python -m src.run_experiment balance maddpg --seed $seed --render
done

for seed in {1..5}
do
    python -m src.run_experiment buzz_wire maddpg --seed $seed --render
done

for seed in {1..5}
do
    python -m src.run_experiment transport maddpg --seed $seed --render
done