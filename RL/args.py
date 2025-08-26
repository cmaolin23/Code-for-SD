# args.py
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="A2C for component merging (SD)")

    # IO
    parser.add_argument("--graph", type=str, required=True,
                        help="Edge list file: one edge per line: u v")
    parser.add_argument("--train", type=str, required=True,
                        help="Train queries: one node id per line (raw id)")
    parser.add_argument("--test", type=str, required=True,
                        help="Test queries: one node id per line (raw id)")
    parser.add_argument("--out_dir", type=str, default="results",
                        help="Output directory to save results and model")
    parser.add_argument("--save_path", type=str, default="log",
                        help="Path to save the trained model")

    # problem
    parser.add_argument("--tau", type=int, required=True,
                        help="Threshold tau for a component to be qualified")
    parser.add_argument("--b", type=int, required=True,
                        help="Budget b: maximum number of edge insertions per query")

    # training
    parser.add_argument("--train_epochs", type=int, default=0,
                        help="If >0, train A2C for this many epochs on train queries")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent_coef", type=float, default=0.01,
                        help="Entropy regularization coefficient")
    parser.add_argument("--value_coef", type=float, default=0.5,
                        help="Value loss coefficient")
    parser.add_argument("--n_step", type=int, default=5, help="n-step return for A2C")

    # model
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)

    # misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None,
                        help="torch device, e.g. 'cuda' or 'cpu'")

    args = parser.parse_args()
    return args

# python main.py --graph web-Google.txt --train train.txt --test.txt --tau 5 --b 5