import argparse
import json
from pathlib import Path

import debugpy
from matplotlib import pyplot as plt


if __name__ == "__main__":
    # debugpy.listen(5678)
    # print("Waiting for debugger attach...")
    # debugpy.wait_for_client()
    # print("Debugger attached.")
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Directory containing log files to visualize.",
    )
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    train_json = log_dir / "train.json"
    eval_json = sorted(list(log_dir.glob("eval_*.json")))
    score_json = sorted(list(log_dir.glob("score_*.json")))
    
    csv_dir = log_dir.parent / "csvs"
    plot_dir = log_dir.parent / "plots"
    csv_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    
    train_dict = json.load(open(train_json, "r"))
    eval_dict = [json.load(open(f, "r")) for f in eval_json]
    score_dict = [json.load(open(f, "r")) for f in score_json]
    
    # Prepare data
    iters = list(range(len(train_dict)))
    
    train_returns = {}
    regrets = {}
    novelties = {}
    progresses = {}
    scores = {}
    for i, t, s in zip(iters, train_dict, score_dict):
        train_returns[t["layout"]] = train_returns.get(t["layout"], []) + [(i, t["avg_return"])]
        for layout in s.keys():
            regrets[layout] = regrets.get(layout, []) + [(i, s[layout]["regret"])]
            novelties[layout] = novelties.get(layout, []) + [(i, s[layout]["novelty"])]
            progresses[layout] = progresses.get(layout, []) + [(i, s[layout]["progress"])]
            scores[layout] = scores.get(layout, []) + [(i, s[layout]["score"])]
    
    eval_returns = {}
    for i, e in zip(iters, eval_dict):
        for layout in e.keys():
            eval_returns[layout] = eval_returns.get(layout, []) + [(i, e[layout])]
    
    # Save data as csv
    for layout in eval_returns.keys():
        csv_path = csv_dir / f"eval_{layout}_{log_dir.name}.csv"
        with open(csv_path, "w") as f:
            f.write("iteration,return\n")
            for x, y in eval_returns[layout]:
                f.write(f"{x},{y}\n")
    
    # Plot
    # for layout in eval_returns.keys():
    #     print(f"Plotting eval returns for layout: {layout}")
        
    #     plt.figure(figsize=(10, 6))
    #     data = eval_returns[layout]
    #     x, y = zip(*data)
    #     # x = x[:50]
    #     # y = y[:50]
    #     plt.plot(x, y, label="Eval Return", color='blue')
    #     plt.title(f"Eval Returns for {layout}")
    #     plt.xlabel("Iteration")
    #     plt.ylabel("Return")
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #     plt.savefig(plot_dir / f"eval_{layout}_{log_dir.name}.png")
    #     plt.close()

    # print("Done plotting.")