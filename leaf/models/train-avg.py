with open('train.txt', 'r') as f:
    lines = f.readlines()[2:]  # 跳过标题2行
times = [float(line.split(': ')[1].split(' seconds')[0]) for line in lines if line.startswith('Round ')]
avg = sum(times[:100]) / min(100, len(times))
print(f'前100轮平均training time: {avg:.4f} seconds')
# 如需写回文件：
with open('train.txt', 'a+') as f:
    f.write(f'\n=== 前100轮平均training time: {avg:.4f} seconds ===\n')