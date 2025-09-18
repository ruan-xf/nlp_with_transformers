
labels = [
    f'{c}-{i}'
    for c in 'BI' for i in range(1, 54+1) if i not in (27, 45)
]

labels.append('O')