all: generate learn

generate:
		python3 generator/generate_learning_task.py

learn:
		bash run_tasks_docker.sh
