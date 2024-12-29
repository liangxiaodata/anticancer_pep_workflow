import pandas as pd
import sys

def process_csv(input_csv, output_txt):
    # 假设 CSV 文件中有一列名为 'sequence'
    df = pd.read_csv(input_csv)
    sequences = df['sequence'].dropna().tolist()

    # 将序列写入文件
    with open(output_txt, 'w') as f:
        for seq in sequences:
            f.write(f"{seq}\n")

if __name__ == "__main__":
    input_csv = sys.argv[1]   # 输入 CSV 文件
    output_txt = sys.argv[2]  # 输出文本文件
    process_csv(input_csv, output_txt)

